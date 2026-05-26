"""Unconstrained Monotonic Neural Network with softplus + 12-pt Gauss-Legendre."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 12-point Gauss-Legendre nodes/weights on [-1, 1]
_NODES_NP, _WEIGHTS_NP = np.polynomial.legendre.leggauss(12)


class UMNNBlock(nn.Module):
    """g(z; c) = bias(c) + integral_0^z softplus(MLP(t, c)) dt, monotone in z by construction.

    Implementation uses 12-point Gauss-Legendre quadrature mapped to [0, z]. The
    Jacobian factor dg/dz is the integrand at z, returned in closed form (no
    recursive autograd).
    """

    def __init__(self, context_dim: int, hidden: int = 32):
        super().__init__()
        self.context_dim = context_dim
        in_dim = 1 + context_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init final layer so g starts near identity-ish at init
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        # Bias network on context (or scalar bias if context_dim == 0)
        if context_dim == 0:
            self.bias_param = nn.Parameter(torch.zeros(1))
            self.bias_net = None
        else:
            self.bias_param = None
            self.bias_net = nn.Linear(context_dim, 1)

        self.register_buffer("_nodes", torch.tensor(_NODES_NP, dtype=torch.float32))
        self.register_buffer("_weights", torch.tensor(_WEIGHTS_NP, dtype=torch.float32))

    def _bias(self, context: Optional[torch.Tensor]) -> torch.Tensor:
        if self.bias_net is not None:
            return self.bias_net(context)
        return self.bias_param

    def _integrand(self, t: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        """softplus(MLP([t, c])) -- strictly positive."""
        if context is None or self.context_dim == 0:
            inputs = t
        else:
            # t: (n, K, 1); context: (n, C) -> broadcast to (n, K, C)
            ctx = context.unsqueeze(1).expand(-1, t.size(1), -1)
            inputs = torch.cat([t, ctx], dim=-1)
        return F.softplus(self.mlp(inputs))

    def forward(self, z: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """g(z; c). z has shape (n, 1); returns (n, 1)."""
        n = z.size(0)
        # Map nodes from [-1, 1] to [0, z]: t = z/2 (u + 1)
        u = self._nodes.view(1, -1, 1).expand(n, -1, 1)  # (n, K, 1)
        z_exp = z.view(n, 1, 1).expand(-1, u.size(1), -1)  # (n, K, 1)
        t = 0.5 * z_exp * (u + 1.0)
        integrand = self._integrand(t, context)  # (n, K, 1)
        weights = self._weights.view(1, -1, 1)  # (1, K, 1)
        # integral_0^z f(t) dt = (z/2) sum w_i f(t_i)
        integral = 0.5 * z * (weights * integrand).sum(dim=1)
        return self._bias(context) + integral

    def jacobian_factor(
        self, z: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """dg/dz evaluated at z -- softplus(MLP([z, c]))."""
        if context is None or self.context_dim == 0:
            inputs = z
        else:
            inputs = torch.cat([z, context], dim=-1)
        return F.softplus(self.mlp(inputs))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
