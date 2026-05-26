"""AdditiveFlow1D: r(θ, X) = α_a · UMNN(θ) − α_b · UMNN(X). Monotone in θ and X by construction."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from cdsbi.flows.base import Flow, Guarantee
from cdsbi.flows.umnn import UMNNBlock


class AdditiveFlow1D(nn.Module, Flow):
    monotonicity_guarantees = frozenset({Guarantee.R1, Guarantee.R2})

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.a = UMNNBlock(context_dim=0, hidden=hidden)
        self.b = UMNNBlock(context_dim=0, hidden=hidden)
        # log-parameterize so effective alpha = exp(log_alpha) > 0 always
        # init: effective alpha ≈ 0.72 ⇒ slope 0.5 so optimization moves
        init_log_alpha = math.log(0.5 / math.log(2.0))
        self.log_alpha_a = nn.Parameter(torch.tensor(init_log_alpha))
        self.log_alpha_b = nn.Parameter(torch.tensor(init_log_alpha))

    def forward(
        self, theta: torch.Tensor, context: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # context is the conditioner-encoded X (Identity ⇒ context == X)
        alpha_a = torch.exp(self.log_alpha_a)
        alpha_b = torch.exp(self.log_alpha_b)
        a = self.a(theta, context=None)
        b = self.b(context, context=None)
        r = alpha_a * a - alpha_b * b
        # ∂r/∂input where "input" = X = context: derivative is −α_b · b'(X)
        # log |∂r/∂X| = log(α_b) + log(b'(X)) = log_alpha_b + log(b'(X)); positive by construction
        b_prime = self.b.jacobian_factor(context, context=None)
        log_det = (self.log_alpha_b + torch.log(b_prime)).sum(dim=-1)
        return r, log_det

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
