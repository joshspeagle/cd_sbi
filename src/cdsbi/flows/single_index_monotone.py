"""SingleIndexMonotoneFlow — single-index monotone autoregressive pivot flow.

Per coordinate k (ctx_k = (θ_{<k}, feat_{<k})):
    z_k = s_θk·softplus(p_θk(ctx_k))·θ_k + s_fk·softplus(p_fk(ctx_k))·feat_k + off_k(ctx_k)
    r_k = G_k(z_k; ctx_k)        # G_k monotone-increasing UMNN of the index z_k
Signs s_θk, s_fk ∈ {+1,-1} are FIXED per coordinate/variable (from the target's
known monotonicity). Then ∂r_k/∂θ_k = s_θk·softplus(p_θk)·G'_k and
∂r_k/∂feat_k = s_fk·softplus(p_fk)·G'_k have fixed signs GLOBALLY (G'>0), so R1/R2
hold with independent signs and no θ_ref restriction. Non-additive (G nonlinear);
ctx-conditioned magnitudes supply multiplicative scale. Feature-Jacobian is
lower-triangular ⇒ log|det| = Σ_k log|s_fk·softplus(p_fk)·G'_k| (closed form).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

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


class _CtxScalar(nn.Module):
    """ctx → scalar (n,1). A learnable parameter when context_dim==0, else an MLP."""
    def __init__(self, context_dim: int, hidden: int, depth: int = 2):
        super().__init__()
        self.context_dim = context_dim
        if context_dim == 0:
            self.param = nn.Parameter(torch.zeros(1)); self.net = None
        else:
            self.param = None; self.net = _tanh_mlp(context_dim, hidden, 1, depth)

    def forward(self, ctx: Optional[torch.Tensor], n: int) -> torch.Tensor:
        if self.context_dim == 0 or ctx is None:
            return self.param.view(1, 1).expand(n, 1)
        return self.net(ctx)


class _MonotoneG(nn.Module):
    """G(z; ctx) = ∫_0^z softplus(MLP([t,ctx])) dt — monotone-increasing in z, G(0)=0."""
    def __init__(self, context_dim: int, hidden: int = 32, depth: int = 2):
        super().__init__()
        self.context_dim = context_dim
        self.mlp = _tanh_mlp(1 + context_dim, hidden, 1, depth)
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("_nodes", torch.tensor(_NODES_NP, dtype=torch.float32))
        self.register_buffer("_weights", torch.tensor(_WEIGHTS_NP, dtype=torch.float32))

    def _integrand(self, t: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        inp = t if (self.context_dim == 0 or ctx is None) else torch.cat([t, ctx], dim=-1)
        return F.softplus(self.mlp(inp)) + 1e-3

    def derivative(self, z: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        return self._integrand(z, ctx)   # G'(z) = softplus(MLP([z,ctx])) + 1e-3 > 0

    def forward(self, z: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        n = z.shape[0]; K = self._nodes.shape[0]
        u = self._nodes.view(1, -1, 1).expand(n, -1, 1)
        ze = z.view(n, 1, 1).expand(-1, K, -1)
        t = 0.5 * ze * (u + 1.0)                      # integrate 0 -> z
        if self.context_dim == 0 or ctx is None:
            ctx_rep = None
        else:
            ctx_rep = ctx.unsqueeze(1).expand(-1, K, -1).reshape(n * K, -1)
        integ = self._integrand(t.reshape(n * K, 1), ctx_rep).view(n, K, 1)
        return 0.5 * z * (self._weights.view(1, -1, 1) * integ).sum(dim=1)   # (n,1)


class SingleIndexMonotoneFlow(nn.Module, Flow):
    monotonicity_guarantees = frozenset({Guarantee.R1, Guarantee.R2})

    def __init__(self, d: int, theta_signs: Sequence[float], feat_signs: Sequence[float],
                 hidden: int = 32, depth: int = 2):
        super().__init__()
        if d < 1:
            raise ValueError(f"d must be ≥ 1, got {d}")
        if len(theta_signs) != d or len(feat_signs) != d:
            raise ValueError(f"theta_signs/feat_signs must have length d={d}")
        self.d = d
        self.register_buffer("_s_theta", torch.tensor([float(s) for s in theta_signs]))
        self.register_buffer("_s_feat", torch.tensor([float(s) for s in feat_signs]))
        self._p_theta = nn.ModuleList(); self._p_feat = nn.ModuleList()
        self._off = nn.ModuleList(); self._G = nn.ModuleList()
        for k in range(d):
            cdim = 2 * k
            self._p_theta.append(_CtxScalar(cdim, hidden))
            self._p_feat.append(_CtxScalar(cdim, hidden))
            self._off.append(_CtxScalar(cdim, hidden))
            self._G.append(_MonotoneG(cdim, hidden, depth))

    def forward(self, theta: torch.Tensor, context: Optional[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert context is not None and context.shape[-1] == self.d, (
            f"expected context (features) width d={self.d}, got "
            f"{None if context is None else context.shape}"
        )
        feats = context; n = theta.shape[0]
        r_cols: List[torch.Tensor] = []; logdet_terms: List[torch.Tensor] = []
        for k in range(self.d):
            theta_k = theta[:, k:k + 1]; feat_k = feats[:, k:k + 1]
            ctx_k = None if k == 0 else torch.cat([theta[:, :k], feats[:, :k]], dim=-1)
            w_th = self._s_theta[k] * F.softplus(self._p_theta[k](ctx_k, n))   # (n,1)
            w_f = self._s_feat[k] * F.softplus(self._p_feat[k](ctx_k, n))      # (n,1)
            off = self._off[k](ctx_k, n)
            z = w_th * theta_k + w_f * feat_k + off
            r_k = self._G[k](z, ctx_k)                                        # (n,1)
            g_prime = self._G[k].derivative(z, ctx_k)                         # (n,1) >0
            dr_dfeat = w_f * g_prime                                          # ∂r_k/∂feat_k
            r_cols.append(r_k)
            logdet_terms.append(torch.log(dr_dfeat.abs().clamp_min(1e-12)).squeeze(-1))
        r = torch.cat(r_cols, dim=-1)
        log_det = torch.stack(logdet_terms, dim=-1).sum(dim=-1)
        return r, log_det

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
