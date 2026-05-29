"""TriangularDoublyMonotoneFlow — non-additive autoregressive doubly-monotone flow.

Coordinate k: r_k = b_k(feat_k; ctx_k) + ∫_{θ_ref}^{θ_k} softplus(α_k(t; ctx_k)
+ β_k(feat_k; ctx_k)) dt,  with ctx_k = (θ_{<k}, feat_{<k}). Each coordinate is
R1 (∂r_k/∂θ_k = softplus(·) > 0) and R2 (∂r_k/∂feat_k > 0) by construction. The
feature-Jacobian is lower-triangular (r_k depends only on feat_{≤k}), so
log|det ∂r/∂feat| = Σ_k log(∂r_k/∂feat_k), computed in closed form by the same
quadrature as the forward pass.

Generalizes TriangularAdditiveFlow (additive special case) and
DoublyMonotoneUMNN (d=1 special case).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cdsbi.flows.base import Flow, Guarantee

_NODES_NP, _WEIGHTS_NP = np.polynomial.legendre.leggauss(12)


def _tanh_mlp(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Sequential:
    assert depth >= 1
    layers: list = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class _CondMonotoneScalarUMNN(nn.Module):
    """Monotone-increasing-in-z scalar map conditioned on a context vector:
        f(z; ctx) = bias(ctx) + ∫_{z_ref}^{z} softplus(MLP([t, ctx])) dt
    derivative(z; ctx) = softplus(MLP([z, ctx])).  context_dim=0 ⇒ unconditioned.
    """

    def __init__(self, context_dim: int, hidden: int = 16, depth: int = 2, z_ref: float = 0.0):
        super().__init__()
        self.context_dim = context_dim
        self.z_ref = z_ref
        self.mlp = _tanh_mlp(in_dim=1 + context_dim, hidden=hidden, out_dim=1, depth=depth)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        if context_dim == 0:
            self.bias_param = nn.Parameter(torch.zeros(1))
            self.bias_net = None
        else:
            self.bias_param = None
            self.bias_net = nn.Linear(context_dim, 1)
        self.register_buffer("_nodes", torch.tensor(_NODES_NP, dtype=torch.float32))
        self.register_buffer("_weights", torch.tensor(_WEIGHTS_NP, dtype=torch.float32))

    def _bias(self, ctx: Optional[torch.Tensor], n: int, device, dtype) -> torch.Tensor:
        if self.context_dim == 0 or ctx is None:
            return self.bias_param.to(device=device, dtype=dtype).view(1, 1).expand(n, 1)
        return self.bias_net(ctx)                              # (n, 1)

    def _integrand(self, t: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        """softplus(MLP([t, ctx])) + 1e-3, strictly positive. t: (m, 1); ctx: (m, c) or None."""
        if self.context_dim == 0 or ctx is None:
            inp = t
        else:
            inp = torch.cat([t, ctx], dim=-1)
        return F.softplus(self.mlp(inp)) + 1e-3

    def derivative(self, z: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        """f'(z; ctx) = softplus(MLP([z, ctx])) + 1e-3. z: (n, 1)."""
        return self._integrand(z, ctx)

    def forward(self, z: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        n = z.shape[0]
        a = self.z_ref
        u = self._nodes.view(1, -1, 1).expand(n, -1, 1)         # (n, K, 1)
        z_exp = z.view(n, 1, 1).expand(-1, u.size(1), -1)
        t = a + 0.5 * (z_exp - a) * (u + 1.0)                   # (n, K, 1)
        K = u.size(1)
        if self.context_dim == 0 or ctx is None:
            ctx_rep = None
        else:
            ctx_rep = ctx.unsqueeze(1).expand(-1, K, -1).reshape(n * K, -1)
        integrand = self._integrand(t.reshape(n * K, 1), ctx_rep).view(n, K, 1)
        weights = self._weights.view(1, -1, 1)
        integral = 0.5 * (z - a) * (weights * integrand).sum(dim=1)   # (n, 1)
        return self._bias(ctx, n, z.device, z.dtype) + integral

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
