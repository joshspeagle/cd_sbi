"""NormalBivariateUnknownCov: bivariate normal with unknown mean AND covariance.

θ = (ℓ₁₁, ℓ₂₂, L₂₁, μ₁, μ₂) log-Cholesky (Σ = L Lᵀ, L lower-triangular with
exp-diagonal). d_theta=5; data X (n_iid bivariate observations) flattened to ℝ²⁰.
The closed-form joint pivot r* (Task 2) is built from the Wishart Bartlett
decomposition. Autoregressive order: covariance Cholesky (diagonals, then
off-diagonal) then mean.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from scipy.stats import chi2, norm


class NormalBivariateUnknownCov:
    def __init__(self, n_iid: int = 10,
                 mu_range: Tuple[float, float] = (-3.0, 3.0),
                 log_chol_range: Tuple[float, float] = (math.log(0.4), math.log(2.5)),
                 l21_range: Tuple[float, float] = (-1.5, 1.5)):
        self.n_iid = n_iid
        self.mu_range = mu_range
        self.log_chol_range = log_chol_range
        self.l21_range = l21_range
        self.d_theta = 5
        self.p = 2

    @property
    def d_x(self) -> int:
        return self.n_iid * self.p

    @property
    def theta_signs(self) -> Tuple[float, ...]:
        return (1.0, 1.0, -1.0, 1.0, 1.0)

    @property
    def feat_signs(self) -> Tuple[float, ...]:
        return (-1.0, -1.0, 1.0, -1.0, -1.0)

    def _draw_theta(self, n: int, rng: np.random.Generator) -> np.ndarray:
        l11 = rng.uniform(*self.log_chol_range, size=(n, 1))
        l22 = rng.uniform(*self.log_chol_range, size=(n, 1))
        L21 = rng.uniform(*self.l21_range, size=(n, 1))
        mu = rng.uniform(*self.mu_range, size=(n, 2))
        return np.concatenate([l11, l22, L21, mu], axis=1)

    def _chol(self, theta_np: np.ndarray) -> np.ndarray:
        n = theta_np.shape[0]
        L = np.zeros((n, 2, 2))
        L[:, 0, 0] = np.exp(theta_np[:, 0])
        L[:, 1, 1] = np.exp(theta_np[:, 1])
        L[:, 1, 0] = theta_np[:, 2]
        return L

    def _sample_x(self, theta_np: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = theta_np.shape[0]
        L = self._chol(theta_np)
        mu = theta_np[:, 3:5]
        z = rng.standard_normal(size=(n, self.n_iid, 2))
        x = mu[:, None, :] + np.einsum('nij,nkj->nki', L, z)
        return x

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_np = self._draw_theta(n, rng)
        x = self._sample_x(theta_np, rng).reshape(n, self.d_x)
        return torch.from_numpy(theta_np).float(), torch.from_numpy(x).float()

    def sample_x_given_theta(self, theta_0, n: int, rng: np.random.Generator) -> torch.Tensor:
        theta_vec = np.atleast_1d(np.asarray(theta_0, dtype=np.float64))
        assert theta_vec.shape == (self.d_theta,), (
            f"theta_0 has shape {theta_vec.shape}, expected ({self.d_theta},)"
        )
        theta_rep = np.tile(theta_vec[None, :], (n, 1))
        x = self._sample_x(theta_rep, rng).reshape(n, self.d_x)
        return torch.from_numpy(x).float()

    def _bartlett(self, x: torch.Tensor):
        n = x.shape[0]
        obs = x.reshape(n, self.n_iid, self.p)
        xbar = obs.mean(dim=1)
        Xc = obs - xbar[:, None, :]
        A = Xc.transpose(1, 2) @ Xc
        D = torch.linalg.cholesky(A)
        return xbar, D

    def r_star(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        n_obs = self.n_iid
        xbar, D = self._bartlett(x)
        D11 = D[:, 0, 0]; D21 = D[:, 1, 0]; D22 = D[:, 1, 1]
        l11 = theta[:, 0]; l22 = theta[:, 1]; L21 = theta[:, 2]
        mu1 = theta[:, 3]; mu2 = theta[:, 4]
        C11 = torch.exp(l11); C22 = torch.exp(l22)
        T11 = D11 / C11; T22 = D22 / C22
        T21 = (D21 - (L21 / C11) * D11) / C22
        w1 = (T11 ** 2).detach().cpu().numpy(); w2 = (T22 ** 2).detach().cpu().numpy()
        u1 = np.clip(1.0 - chi2.cdf(w1, df=n_obs - 1), 1e-12, 1 - 1e-12)
        u2 = np.clip(1.0 - chi2.cdf(w2, df=n_obs - 2), 1e-12, 1 - 1e-12)
        r1 = torch.from_numpy(norm.ppf(u1)).float().to(theta.device)
        r2 = torch.from_numpy(norm.ppf(u2)).float().to(theta.device)
        r3 = T21
        sq = math.sqrt(n_obs)
        r4 = sq * (mu1 - xbar[:, 0]) / C11
        r5 = sq * (-(L21 / (C11 * C22)) * (mu1 - xbar[:, 0]) + (mu2 - xbar[:, 1]) / C22)
        return torch.stack([r1, r2, r3, r4, r5], dim=-1)

    def oracle_summary(self, x: torch.Tensor) -> torch.Tensor:
        xbar, D = self._bartlett(x)
        D11 = D[:, 0, 0].clamp_min(1e-12); D22 = D[:, 1, 1].clamp_min(1e-12)
        return torch.stack([torch.log(D11), torch.log(D22), D[:, 1, 0],
                            xbar[:, 0], xbar[:, 1]], dim=-1)

    def data_entropy_lower_bound(self) -> float:
        mid = 0.5 * (self.log_chol_range[0] + self.log_chol_range[1])
        return self.n_iid * (0.5 * self.p * (1 + math.log(2 * math.pi)) + 2 * mid)
