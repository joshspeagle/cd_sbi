"""LocationGaussian2D_iid: X | θ ~ N(θ, I_2) with θ ~ U[a, b]^2."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class LocationGaussian2D_iid:
    theta_range: Tuple[float, float] = (-7.0, 7.0)
    d_theta: int = 2
    d_x: int = 2

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = self.theta_range
        theta_np = rng.uniform(a, b, size=(n, self.d_theta))
        eps_np = rng.standard_normal(size=(n, self.d_x))
        x_np = theta_np + eps_np
        return (
            torch.from_numpy(theta_np).float(),
            torch.from_numpy(x_np).float(),
        )

    def sample_x_given_theta(self, theta_0, n: int, rng: np.random.Generator) -> torch.Tensor:
        """Draw n samples of X conditional on θ = θ_0. Used by coverage / size diagnostics."""
        theta_vec = np.atleast_1d(np.asarray(theta_0, dtype=np.float64))
        assert theta_vec.shape == (self.d_theta,), (
            f"theta_0 has shape {theta_vec.shape}, expected ({self.d_theta},)"
        )
        eps_np = rng.standard_normal(size=(n, self.d_x))
        x_np = theta_vec[None, :] + eps_np  # broadcast (1, d_x) + (n, d_x) → (n, d_x)
        return torch.from_numpy(x_np).float()

    def r_star(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return theta - x

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # -d/2 log(2π) - ½ ||x - θ||²
        return -0.5 * self.d_x * math.log(2 * math.pi) - 0.5 * (x - theta).pow(2).sum(dim=-1)

    def entropy_lower_bound(self) -> float:
        # H(N(θ, I_d)) = d * ½ log(2πe)
        return self.d_x * 0.5 * math.log(2 * math.pi * math.e)
