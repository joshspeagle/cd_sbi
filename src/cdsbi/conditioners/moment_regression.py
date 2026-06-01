"""MomentRegressionConditioner: a learned d_θ-dim summary h_φ(X) trained (Stage 1,
by the TwoStageCDSBIRunner) via L² regression to a d_θ-dim posterior-moment target
m(θ). Permutation-invariant DeepSets backbone (mean-pool of a per-element MLP). The
summary is a feature map, not part of the NF-MLE change-of-variables, so
log_det_contrib = 0 (mirrors DeepSetsConditioner)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _tanh_mlp(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Sequential:
    assert depth >= 1
    layers: list = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class MomentRegressionConditioner(nn.Module):
    def __init__(self, n_iid: int, d_theta: int, p: int = 1, hidden: int = 64,
                 depth: int = 2, target: str = "theta"):
        super().__init__()
        if target not in ("theta", "theta_sq"):
            raise ValueError(f"target must be 'theta' or 'theta_sq', got {target}")
        self.n_iid = n_iid
        self.d_theta = d_theta
        self.p = p                                              # obs dim (1 scalar; 2 bivariate)
        self.target = target
        self.phi = _tanh_mlp(p, hidden, hidden, depth)          # per-observation ℝ^p→ℝ^h
        self.rho = _tanh_mlp(hidden, hidden, d_theta, depth)    # pooled ℝ^h→ℝ^{d_θ}

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # X is n_iid observations each in ℝ^p, flattened to width n_iid·p (p=1 ⟹ scalar
        # iid, the μσ/sign case; p=2 ⟹ bivariate, the d=5 mu_cov case — pooling over
        # observations, NOT over the p coords, so cross-moment structure is preserved).
        assert x.shape[-1] == self.n_iid * self.p, (
            f"MomentRegressionConditioner expected n_iid·p={self.n_iid * self.p} "
            f"features, got {x.shape[-1]}"
        )
        n = x.shape[0]
        obs = x.reshape(n, self.n_iid, self.p)                            # (n, n_iid, p)
        h = self.phi(obs.reshape(n * self.n_iid, self.p)).reshape(
            n, self.n_iid, -1).mean(dim=1)                                # (n, hidden)
        feats = self.rho(h)                                              # (n, d_θ)
        log_det = torch.zeros(n, dtype=x.dtype, device=x.device)
        return feats, log_det

    def regression_target(self, theta: torch.Tensor) -> torch.Tensor:
        if self.target == "theta":
            return theta
        return theta ** 2          # coordinatewise (d_θ-dim, respects the §16 cap)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
