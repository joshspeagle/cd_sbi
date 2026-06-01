# src/cdsbi/flows/non_monotone_pivot.py
"""NonMonotonePivotFlow — R1-OFF pivot. Per coord k:
    z_k = a_k(θ_{≤k}, ctx) + s_fk·softplus(p_fk(ctx))·feat_k      # a_k UNCONSTRAINED
    r_k = G_k(z_k; ctx)                                           # G_k monotone ↑
so ∂r_k/∂feat_k = s_fk·softplus·G' has a fixed sign (C1 ✓, honest log-det) while
∂r_k/∂θ is free (R1 OFF ✓ — permits r(θ)=r(−θ), hence disconnected C_α). Mirrors
SingleIndexMonotoneFlow but swaps the monotone θ-channel for an unconstrained MLP.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cdsbi.flows.base import Flow, Guarantee

_NODES_NP, _WEIGHTS_NP = np.polynomial.legendre.leggauss(12)


def _tanh_mlp(in_dim, hidden, out_dim, depth):
    layers: list = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class _CtxScalar(nn.Module):
    def __init__(self, context_dim, hidden, depth=2):
        super().__init__()
        self.context_dim = context_dim
        if context_dim == 0:
            self.param = nn.Parameter(torch.zeros(1)); self.net = None
        else:
            self.param = None; self.net = _tanh_mlp(context_dim, hidden, 1, depth)

    def forward(self, ctx, n):
        if self.context_dim == 0 or ctx is None:
            return self.param.view(1, 1).expand(n, 1)
        return self.net(ctx)


class _MonotoneG(nn.Module):
    """G(z;ctx)=∫_0^z softplus(MLP([t,ctx]))dt — monotone ↑ in z, G(0)=0."""
    def __init__(self, context_dim, hidden=32, depth=2):
        super().__init__()
        self.context_dim = context_dim
        self.mlp = _tanh_mlp(1 + context_dim, hidden, 1, depth)
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("_nodes", torch.tensor(_NODES_NP, dtype=torch.float32))
        self.register_buffer("_weights", torch.tensor(_WEIGHTS_NP, dtype=torch.float32))

    def _integrand(self, t, ctx):
        inp = t if (self.context_dim == 0 or ctx is None) else torch.cat([t, ctx], -1)
        return F.softplus(self.mlp(inp)) + 1e-3

    def derivative(self, z, ctx):
        return self._integrand(z, ctx)

    def forward(self, z, ctx):
        n = z.shape[0]; K = self._nodes.shape[0]
        u = self._nodes.view(1, -1, 1).expand(n, -1, 1)
        ze = z.view(n, 1, 1).expand(-1, K, -1)
        t = 0.5 * ze * (u + 1.0)
        ctx_rep = None if (self.context_dim == 0 or ctx is None) else \
            ctx.unsqueeze(1).expand(-1, K, -1).reshape(n * K, -1)
        integ = self._integrand(t.reshape(n * K, 1), ctx_rep).view(n, K, 1)
        return 0.5 * z * (self._weights.view(1, -1, 1) * integ).sum(dim=1)


class NonMonotonePivotFlow(nn.Module, Flow):
    monotonicity_guarantees = frozenset({Guarantee.R2})    # feature-monotone only

    def __init__(self, d: int, feat_signs: Sequence[float], hidden: int = 32,
                 depth: int = 2, theta_hidden: int = 32):
        super().__init__()
        if len(feat_signs) != d:
            raise ValueError(f"feat_signs must have length d={d}")
        self.d = d
        self.register_buffer("_s_feat", torch.tensor([float(s) for s in feat_signs]))
        self._a = nn.ModuleList()        # unconstrained θ-channel a_k(θ_{≤k}, ctx)
        self._p_feat = nn.ModuleList()
        self._G = nn.ModuleList()
        for k in range(d):
            cdim = 2 * k                                   # ctx = (θ_{<k}, feat_{<k})
            self._a.append(_tanh_mlp(cdim + 1, theta_hidden, 1, depth))  # +1 = θ_k
            self._p_feat.append(_CtxScalar(cdim, hidden))
            self._G.append(_MonotoneG(cdim, hidden, depth))

    def forward(self, theta: torch.Tensor, context: Optional[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert context is not None and context.shape[-1] == self.d
        feats = context; n = theta.shape[0]
        r_cols: List[torch.Tensor] = []; logdet: List[torch.Tensor] = []
        for k in range(self.d):
            feat_k = feats[:, k:k + 1]
            ctx_k = None if k == 0 else torch.cat([theta[:, :k], feats[:, :k]], -1)
            a_in = theta[:, :k + 1] if ctx_k is None else torch.cat([theta[:, :k + 1], feats[:, :k]], -1)
            a = self._a[k](a_in)                                       # unconstrained in θ
            w_f = self._s_feat[k] * F.softplus(self._p_feat[k](ctx_k, n))
            z = a + w_f * feat_k
            r_k = self._G[k](z, ctx_k)
            g_prime = self._G[k].derivative(z, ctx_k)
            dr_dfeat = w_f * g_prime                                   # ∂r_k/∂feat_k (signed)
            r_cols.append(r_k)
            logdet.append(torch.log(dr_dfeat.abs().clamp_min(1e-12)).squeeze(-1))
        return torch.cat(r_cols, -1), torch.stack(logdet, -1).sum(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
