"""LocationNormal1D simulator: X ~ N(θ, 1) with θ ~ U[a, b]."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class LocationNormal1D:
    theta_range: Tuple[float, float] = (-7.0, 7.0)
    d_theta: int = 1
    d_x: int = 1

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = self.theta_range
        theta_np = rng.uniform(a, b, size=(n, 1))
        eps_np = rng.standard_normal(size=(n, 1))
        x_np = theta_np + eps_np
        return (
            torch.from_numpy(theta_np).float(),
            torch.from_numpy(x_np).float(),
        )

    def sample_x_given_theta(self, theta_0, n: int, rng: "np.random.Generator") -> torch.Tensor:
        """Draw n samples of X conditional on θ = θ_0. Used by coverage / size diagnostics."""
        # theta_0 may be a scalar float or a length-1 sequence — both map to a (1,) vector.
        import numpy as np
        theta_vec = np.atleast_1d(np.asarray(theta_0, dtype=np.float64))
        assert theta_vec.shape == (self.d_theta,), (
            f"theta_0 has shape {theta_vec.shape}, expected ({self.d_theta},)"
        )
        eps = rng.standard_normal(size=(n, self.d_x))
        x_np = theta_vec[None, :] + eps  # broadcast (1, d_x) + (n, d_x) → (n, d_x)
        return torch.from_numpy(x_np).float()

    def r_star(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return theta - x

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return -0.5 * math.log(2 * math.pi) - 0.5 * (x - theta).pow(2).sum(dim=-1)

    def entropy_lower_bound(self) -> float:
        return 0.5 * math.log(2 * math.pi * math.e)
