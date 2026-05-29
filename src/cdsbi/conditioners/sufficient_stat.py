"""SufficientStatConditioner: oracle reduction X → (log s², X̄) for the (μ, σ²) target.

Features are ordered (log s², X̄) to pair with θ = (log σ, μ) in the autoregressive
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
        assert x.shape[-1] == self.n_iid, (
            f"SufficientStatConditioner expected {self.n_iid} iid obs per row, "
            f"got width {x.shape[-1]} — n_iid config is out of sync with the target."
        )
        xbar = x.mean(dim=-1, keepdim=True)
        s2 = x.var(dim=-1, unbiased=True, keepdim=True)
        feats = torch.cat([torch.log(s2.clamp_min(1e-12)), xbar], dim=-1)   # (n,2): [log s², X̄]
        log_det = torch.full(
            (x.shape[0],), _SUFFICIENT_STAT_LOG_DET_CONST, dtype=x.dtype, device=x.device,
        )
        return feats, log_det

    def n_params(self) -> int:
        return 0
