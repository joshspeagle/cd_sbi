"""LocationGaussian2D_corr: X | θ ~ N(θ, Σ) with Σ = [[1, 0.5], [0.5, 1]]
and θ ~ U[-7, 7]^2. r*(θ, X) = L⁻¹(θ - X) where L L^T = Σ.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch


_SIGMA_DEFAULT = ((1.0, 0.5), (0.5, 1.0))


@dataclass
class LocationGaussian2D_corr:
    theta_range: Tuple[float, float] = (-7.0, 7.0)
    d_theta: int = 2
    d_x: int = 2
    # Σ as a nested tuple so the @dataclass remains hashable + YAML-friendly.
    sigma: Tuple[Tuple[float, float], Tuple[float, float]] = field(
        default_factory=lambda: _SIGMA_DEFAULT,
    )

    def __post_init__(self) -> None:
        assert self.d_theta == self.d_x, (
            "correlated 2D Gaussian assumes d_theta == d_x"
        )
        sigma_np = np.asarray(self.sigma, dtype=np.float64)
        assert sigma_np.shape == (self.d_theta, self.d_theta), (
            f"sigma has shape {sigma_np.shape}, expected ({self.d_theta}, {self.d_theta})"
        )
        # Cholesky factor; lower-triangular by convention.
        self._L = np.linalg.cholesky(sigma_np)
        self._L_inv = np.linalg.inv(self._L)
        self._log_det_sigma = float(np.log(np.linalg.det(sigma_np)))

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = self.theta_range
        theta_np = rng.uniform(a, b, size=(n, self.d_theta))
        z_np = rng.standard_normal(size=(n, self.d_x))
        # X = θ + L @ z  ⇒  X - θ ~ N(0, Σ)
        eps_np = z_np @ self._L.T
        x_np = theta_np + eps_np
        return (
            torch.from_numpy(theta_np).float(),
            torch.from_numpy(x_np).float(),
        )

    def sample_x_given_theta(self, theta_0, n: int, rng: np.random.Generator) -> torch.Tensor:
        """Draw n samples of X conditional on θ = θ_0."""
        theta_vec = np.atleast_1d(np.asarray(theta_0, dtype=np.float64))
        assert theta_vec.shape == (self.d_theta,), (
            f"theta_0 has shape {theta_vec.shape}, expected ({self.d_theta},)"
        )
        z_np = rng.standard_normal(size=(n, self.d_x))
        eps_np = z_np @ self._L.T
        x_np = theta_vec[None, :] + eps_np
        return torch.from_numpy(x_np).float()

    def r_star(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Truth pivot: r* = L⁻¹(θ - X). Returns shape (n, d)."""
        L_inv = torch.from_numpy(self._L_inv).to(dtype=theta.dtype, device=theta.device)
        # (theta - x) is (n, d); (L_inv @ (theta-x)^T)^T == (theta-x) @ L_inv^T
        return (theta - x) @ L_inv.T

    def r_star_jacobian(self) -> torch.Tensor:
        """Jacobian ∂r*/∂θ = L⁻¹ (constant across (θ, X) by the location-family structure)."""
        return torch.from_numpy(self._L_inv).float()

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # log p(X | θ) = -d/2 log(2π) - ½ log|Σ| - ½ (X-θ)^T Σ⁻¹ (X-θ)
        L_inv = torch.from_numpy(self._L_inv).to(dtype=x.dtype, device=x.device)
        z = (x - theta) @ L_inv.T  # whitened residuals; ||z||² == (X-θ)^T Σ⁻¹ (X-θ)
        return (
            -0.5 * self.d_x * math.log(2 * math.pi)
            - 0.5 * self._log_det_sigma
            - 0.5 * z.pow(2).sum(dim=-1)
        )

    def entropy_lower_bound(self) -> float:
        # H(N(θ, Σ)) = ½ log((2πe)^d |Σ|)
        return 0.5 * (self.d_x * math.log(2 * math.pi * math.e) + self._log_det_sigma)
