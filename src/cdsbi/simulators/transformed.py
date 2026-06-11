"""AsinhTransformedSimulator: a θ-independent data bijection for heavy tails.

The pilot measured the gate directly (hardening findings, 2026-06-10): a MAF
trained on RAW Cauchy draws produces a garbage score (coverage_error_max ≈ 0.50
— 50% sets covering 99.9%), while the original probe's asinh pre-transform
reached the floor. asinh(x) = log(x + √(x²+1)) compresses tails to log scale,
making the flow trainable; because the transform is θ-INDEPENDENT, the score
∇_θ log q(asinh(X)|θ) is a perfectly valid test statistic of (θ, X) — data
conditioning changes the model's coordinates, never the inferential target.

Wrapper semantics (so every diagnostic stays exact):
- `sample` / `sample_x_given_theta` emit Y = asinh(X).
- `r_star(θ, y)` inverts the transform (X = sinh(Y)) before the base oracle
  pivot — the measured floor row remains EXACTLY calibrated.
- `log_prob(y, θ)` carries the change-of-variables Jacobian:
  log p_Y(y|θ) = log p_X(sinh y|θ) + Σ log cosh(y).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class AsinhTransformedSimulator:
    def __init__(self, base):
        self.base = base

    # --- protocol delegation -------------------------------------------------
    @property
    def d_theta(self) -> int:
        return self.base.d_theta

    @property
    def d_x(self) -> int:
        return self.base.d_x

    @property
    def theta_range(self) -> Tuple[float, float]:
        return self.base.theta_range

    @property
    def theta_lower(self):
        return self.base.theta_lower

    @property
    def theta_upper(self):
        return self.base.theta_upper

    # --- sampling (transformed coordinates) ----------------------------------
    def sample(self, n: int, rng: np.random.Generator):
        theta, x = self.base.sample(n, rng)
        return theta, torch.asinh(x)

    def sample_x_given_theta(self, theta_0, n: int, rng: np.random.Generator) -> torch.Tensor:
        return torch.asinh(self.base.sample_x_given_theta(theta_0, n, rng))

    # --- closed forms (transform inverted / Jacobian carried) ----------------
    def r_star(self, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.base.r_star(theta, torch.sinh(y.to(torch.float64)).to(y.dtype))

    def log_prob(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        x = torch.sinh(y)
        # log|dx/dy| = log cosh(y), summed over the data axis. Stable form
        # (log cosh y = |y| + log1p(e^{-2|y|}) - log 2) avoids cosh overflow.
        a = y.abs()
        jac = (a + torch.log1p(torch.exp(-2.0 * a)) - 0.6931471805599453).sum(dim=-1)
        return self.base.log_prob(x, theta) + jac
