"""SufficientStatConditioner: oracle reduction X → (s², X̄) for the (μ, σ²) target.

Features are ordered (s², X̄) to pair with θ = (log σ, μ) in the autoregressive
flow. Zero-parameter (the oracle), θ-independent constant log-det (sufficiency).
"""
from __future__ import annotations

from typing import Tuple

import torch

from cdsbi.simulators.normal_unknown_mean_var import _SUFFICIENT_STAT_LOG_DET_CONST


class SufficientStatConditioner:
    def __init__(self, n_iid: int):
        self.n_iid = n_iid

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xbar = x.mean(dim=-1, keepdim=True)                       # (n, 1)
        s2 = x.var(dim=-1, unbiased=True, keepdim=True)           # (n, 1)
        feats = torch.cat([s2, xbar], dim=-1)                     # (n, 2): [s², X̄]
        log_det = torch.full(
            (x.shape[0],), _SUFFICIENT_STAT_LOG_DET_CONST, dtype=x.dtype, device=x.device,
        )
        return feats, log_det

    def n_params(self) -> int:
        return 0
