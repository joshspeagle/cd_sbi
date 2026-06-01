"""SignNormal1D: X = (X_1,…,X_{n_iid}) iid N(θ², 1), θ ∈ ℝ. The map θ↦data depends
only on θ², so the posterior is bimodal at ±√·: the canonical multimodal-with-a-
1-dim-sufficient-reduction target (X̄ is sufficient). Used to show CD-SBI yields a
disconnected, exactly-covering C_α with a non-monotone pivot (theory §15)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class SignNormal1D:
    n_iid: int = 10
    theta_abs_max: float = 3.0          # prior: θ ~ U(−3, 3) (symmetric ⟹ bimodal)
    d_theta: int = 1

    @property
    def d_x(self) -> int:
        return self.n_iid

    @property
    def theta_range(self) -> Tuple[float, float]:
        return (-self.theta_abs_max, self.theta_abs_max)

    @property
    def theta_lower(self) -> Tuple[float]:
        return (-self.theta_abs_max,)

    @property
    def feat_signs(self) -> Tuple[float]:
        """The summary (predicts θ²≈X̄) enters the pivot increasingly: r ∝ (T−θ²)."""
        return (1.0,)

    def _draw_theta(self, n, rng):
        return rng.uniform(-self.theta_abs_max, self.theta_abs_max, size=(n, 1))

    def sample(self, n, rng):
        theta = self._draw_theta(n, rng)
        x = rng.normal(loc=theta ** 2, scale=1.0, size=(n, self.n_iid))
        return torch.from_numpy(theta).float(), torch.from_numpy(x).float()

    def sample_x_given_theta(self, theta_0, n, rng):
        tv = np.atleast_1d(np.asarray(theta_0, dtype=np.float64))
        assert tv.shape == (1,), f"theta_0 shape {tv.shape}, expected (1,)"
        x = rng.normal(loc=float(tv[0]) ** 2, scale=1.0, size=(n, self.n_iid))
        return torch.from_numpy(x).float()

    def log_prob(self, x, theta):
        mean = theta[:, 0:1] ** 2                       # (n,1), θ²
        per_obs = -0.5 * (x - mean) ** 2 - 0.5 * math.log(2 * math.pi)
        return per_obs.sum(dim=-1)

    def r_star(self, theta, x):
        """Analytic calibrated pivot r*=√n(X̄−θ²) (non-monotone in θ). Also lets
        the harness PivotRMSE diagnostic run (it checks `simulator.r_star is None`)."""
        xbar = x.mean(dim=-1, keepdim=True)             # (n,1)
        return math.sqrt(self.n_iid) * (xbar - theta[:, 0:1] ** 2)
