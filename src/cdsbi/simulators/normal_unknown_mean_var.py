"""NormalUnknownMeanVar: X = (X_1, …, X_{n_iid}) iid N(μ, σ²).

θ = (log σ, μ) — the forced KR order (scale first). The first target with a
scale/nuisance parameter. Exact closed-form joint pivot (Basu independence of
X̄ and s²):
    r*_σ(θ, X) = Φ⁻¹(F_{χ²_{n-1}}((n-1) s² / σ²))     # uses (σ², s²)
    r*_μ(θ, X) = √n (X̄ − μ) / σ                        # uses (μ, σ; X̄)
with σ = exp(log σ); (r*_σ, r*_μ) ~ N(0, I₂) at the true θ₀.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.stats import chi2, norm


@dataclass
class NormalUnknownMeanVar:
    n_iid: int = 10
    mu_range: Tuple[float, float] = (-5.0, 5.0)
    log_sigma_range: Tuple[float, float] = (math.log(0.3), math.log(3.0))
    d_theta: int = 2

    @property
    def d_x(self) -> int:
        return self.n_iid

    def _draw_theta(self, n: int, rng: np.random.Generator) -> np.ndarray:
        log_sigma = rng.uniform(*self.log_sigma_range, size=(n, 1))
        mu = rng.uniform(*self.mu_range, size=(n, 1))
        return np.concatenate([log_sigma, mu], axis=1)  # (n, 2): [log σ, μ]

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_np = self._draw_theta(n, rng)
        sigma = np.exp(theta_np[:, 0:1])
        mu = theta_np[:, 1:2]
        x_np = rng.normal(loc=mu, scale=sigma, size=(n, self.n_iid))
        return torch.from_numpy(theta_np).float(), torch.from_numpy(x_np).float()

    def sample_x_given_theta(self, theta_0, n: int, rng: np.random.Generator) -> torch.Tensor:
        theta_vec = np.atleast_1d(np.asarray(theta_0, dtype=np.float64))
        assert theta_vec.shape == (self.d_theta,), (
            f"theta_0 has shape {theta_vec.shape}, expected ({self.d_theta},)"
        )
        sigma = float(np.exp(theta_vec[0]))
        mu = float(theta_vec[1])
        x_np = rng.normal(loc=mu, scale=sigma, size=(n, self.n_iid))
        return torch.from_numpy(x_np).float()
