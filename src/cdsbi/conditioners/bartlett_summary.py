"""BartlettSummaryConditioner: oracle reduction X → (log D₁₁, log D₂₂, D₂₁, X̄₁, X̄₂)
for the bivariate-normal (μ, Σ) target. D = lower-Cholesky of the sample scatter.
Zero-parameter; θ-independent constant log-det (sufficiency). Pairs with the
log-Cholesky θ-order (ℓ₁₁, ℓ₂₂, L₂₁, μ₁, μ₂) in the autoregressive flow.
"""
from __future__ import annotations

from typing import Tuple

import torch

_BARTLETT_LOG_DET_CONST = 0.0


class BartlettSummaryConditioner:
    def __init__(self, n_iid: int, p: int = 2):
        self.n_iid = n_iid
        self.p = p

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[-1] == self.n_iid * self.p, (
            f"BartlettSummaryConditioner expected width {self.n_iid * self.p}, got {x.shape[-1]}"
        )
        n = x.shape[0]
        obs = x.reshape(n, self.n_iid, self.p)
        xbar = obs.mean(dim=1)
        Xc = obs - xbar[:, None, :]
        A = Xc.transpose(1, 2) @ Xc
        D = torch.linalg.cholesky(A)
        D11 = D[:, 0, 0].clamp_min(1e-12); D22 = D[:, 1, 1].clamp_min(1e-12)
        feats = torch.stack([torch.log(D11), torch.log(D22), D[:, 1, 0],
                             xbar[:, 0], xbar[:, 1]], dim=-1)
        log_det = torch.full((n,), _BARTLETT_LOG_DET_CONST, dtype=x.dtype, device=x.device)
        return feats, log_det

    def n_params(self) -> int:
        return 0
