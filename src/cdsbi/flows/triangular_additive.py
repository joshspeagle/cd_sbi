"""TriangularAdditiveFlow: r_k = a_k(θ_k; ctx_k) - b_k(X_k; ctx_k), ctx_k = (θ_<k, X_<k).

Autoregressive additive form per manuscript §6.1 form 1. Lower-triangular
Jacobian in both θ and X (autoregressive R1, R2). The k-th coordinate uses
two UMNNBlocks (one over θ_k, one over X_k), each conditioned on the
preceding 2(k-1) values (θ_<k, X_<k). For k=0 the context is empty
(reducing to a standalone scalar UMNN on each of θ_0, X_0), and the
coordinate-wise scaling α_a, α_b match AdditiveFlow1D's convention.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from cdsbi.flows.base import Flow, Guarantee
from cdsbi.flows.umnn import UMNNBlock


class TriangularAdditiveFlow(nn.Module, Flow):
    monotonicity_guarantees = frozenset({Guarantee.R1, Guarantee.R2})

    def __init__(
        self,
        d: int,
        hidden: int = 32,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        if d < 1:
            raise ValueError(f"d must be ≥ 1, got {d}")
        self.d = d
        # Per-coordinate UMNN pairs. Coord k has context dim 2*k (θ_<k, X_<k).
        self.a_blocks = nn.ModuleList()
        self.b_blocks = nn.ModuleList()
        for k in range(d):
            ctx_dim = 2 * k  # θ_<k concat X_<k
            self.a_blocks.append(
                UMNNBlock(context_dim=ctx_dim, hidden=hidden, bias_trainable=True,
                          dropout=dropout, layer_norm=layer_norm)
            )
            self.b_blocks.append(
                UMNNBlock(context_dim=ctx_dim, hidden=hidden, bias_trainable=False,
                          dropout=dropout, layer_norm=layer_norm)
            )
        # Per-coordinate α scalars (matches AdditiveFlow1D init convention).
        init_log_alpha = math.log(0.5 / math.log(2.0))
        self.log_alpha_a = nn.Parameter(torch.full((d,), init_log_alpha))
        self.log_alpha_b = nn.Parameter(torch.full((d,), init_log_alpha))

    def forward(
        self, theta: torch.Tensor, context: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # context is the conditioner-encoded X (Identity ⇒ context == X)
        x = context
        n = theta.shape[0]
        r_cols = []
        log_det_terms = []
        alpha_a = torch.exp(self.log_alpha_a)  # (d,)
        alpha_b = torch.exp(self.log_alpha_b)  # (d,)
        for k in range(self.d):
            theta_k = theta[:, k : k + 1]
            x_k = x[:, k : k + 1]
            if k == 0:
                ctx_k = None
            else:
                ctx_k = torch.cat([theta[:, :k], x[:, :k]], dim=-1)  # (n, 2k)
            a_k = self.a_blocks[k](theta_k, context=ctx_k)
            b_k = self.b_blocks[k](x_k, context=ctx_k)
            r_k = alpha_a[k] * a_k - alpha_b[k] * b_k  # shape (n, 1)
            r_cols.append(r_k)
            # ∂r_k/∂X_k = -α_b[k] · b_k'(X_k; ctx_k). log|·| = log α_b[k] + log b_k'(X_k).
            b_prime = self.b_blocks[k].jacobian_factor(x_k, context=ctx_k)  # (n, 1)
            log_det_terms.append(
                (self.log_alpha_b[k] + torch.log(b_prime.squeeze(-1)))
            )
        r = torch.cat(r_cols, dim=-1)  # (n, d)
        log_det = torch.stack(log_det_terms, dim=-1).sum(dim=-1)  # (n,)
        return r, log_det

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
