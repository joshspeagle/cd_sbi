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
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.stats import chi2, norm


# θ-independent volume constant for the (X̄, s²) sufficient-statistic reduction.
# By sufficiency p(X | X̄, s²) is θ-free, so this only shifts the loss scale,
# not the argmin. Set to 0.0 (the reduction's constant is folded into the
# entropy-floor reference, not the optimization). Kept as a named constant so
# the floor calc (here) and the conditioner (Task 3) stay in lockstep.
_SUFFICIENT_STAT_LOG_DET_CONST = 0.0


@dataclass
class NormalUnknownMeanVar:
    n_iid: int = 10
    mu_range: Tuple[float, float] = (-5.0, 5.0)
    log_sigma_range: Tuple[float, float] = (math.log(0.3), math.log(3.0))
    d_theta: int = 2

    @property
    def d_x(self) -> int:
        return self.n_iid

    @property
    def theta_lower(self) -> Tuple[float, float]:
        """Per-coordinate prior lower bounds (log σ_min, μ_min) — the autoregressive
        flow's theta_ref, so R2 holds by construction across the support."""
        return (self.log_sigma_range[0], self.mu_range[0])

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

    def _suff_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(X̄, s²) with s² the unbiased sample variance (ddof=1) over the n_iid axis."""
        xbar = x.mean(dim=-1, keepdim=True)                       # (n, 1)
        s2 = x.var(dim=-1, unbiased=True, keepdim=True)           # (n, 1), ddof=1
        return xbar, s2

    def r_star(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Closed-form joint pivot (r_σ, r_μ), shape (n, 2). θ = (log σ, μ)."""
        xbar, s2 = self._suff_stats(x)
        log_sigma = theta[:, 0:1]
        mu = theta[:, 1:2]
        sigma = torch.exp(log_sigma)
        n = self.n_iid
        w = ((n - 1) * s2 / (sigma ** 2)).detach().cpu().numpy()
        u = chi2.cdf(w, df=n - 1)
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        r_sigma = torch.from_numpy(norm.ppf(u)).float().to(theta.device)   # (n, 1)
        r_mu = math.sqrt(n) * (xbar - mu) / sigma                          # (n, 1)
        return torch.cat([r_sigma, r_mu], dim=-1)                          # (n, 2)

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Σ_i log N(X_i; μ, σ²)."""
        log_sigma = theta[:, 0:1]
        mu = theta[:, 1:2]
        sigma = torch.exp(log_sigma)
        z = (x - mu) / sigma
        per_obs = -0.5 * z ** 2 - log_sigma - 0.5 * math.log(2 * math.pi)
        return per_obs.sum(dim=-1)

    def entropy_lower_bound(self, n_mc: int = 50000, seed: int = 42) -> float:
        """MC estimate of E[NF-MLE loss at r*] on the (θ, (X̄,s²)) scale."""
        rng = np.random.default_rng(seed)
        theta_np = self._draw_theta(n_mc, rng)
        sigma = np.exp(theta_np[:, 0:1]); mu = theta_np[:, 1:2]
        x = rng.normal(loc=mu, scale=sigma, size=(n_mc, self.n_iid))
        xbar = x.mean(axis=-1, keepdims=True)
        s2 = x.var(axis=-1, ddof=1, keepdims=True)
        n = self.n_iid
        w = (n - 1) * s2 / sigma ** 2
        u = np.clip(chi2.cdf(w, df=n - 1), 1e-12, 1 - 1e-12)
        r_sigma = norm.ppf(u)
        r_mu = math.sqrt(n) * (xbar - mu) / sigma
        dr_sigma_ds2 = (n - 1) / sigma ** 2 * chi2.pdf(w, df=n - 1) / np.clip(norm.pdf(r_sigma), 1e-30, None)
        dr_mu_dxbar = math.sqrt(n) / sigma
        log_det_feat = np.log(np.clip(dr_sigma_ds2, 1e-30, None)) + np.log(dr_mu_dxbar)
        log_det_contrib = _SUFFICIENT_STAT_LOG_DET_CONST
        loss = (0.5 * (r_sigma ** 2 + r_mu ** 2) + math.log(2 * math.pi)
                - log_det_feat - log_det_contrib)
        return float(loss.mean())
