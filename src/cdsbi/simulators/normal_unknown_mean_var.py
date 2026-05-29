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
from scipy.stats import chi2, norm, t


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

    @property
    def theta_signs(self) -> Tuple[float, float]:
        """∂r_k/∂θ_k sign per coord: both increasing-in-θ (= +1)."""
        return (1.0, 1.0)

    @property
    def feat_signs(self) -> Tuple[float, float]:
        """∂r_k/∂feat_k sign per coord: both decreasing-in-feature (= -1).
        Features are (log s², X̄); r_σ ↓ in log s², r_μ ↓ in X̄."""
        return (-1.0, -1.0)

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
        r_sigma = torch.from_numpy(norm.ppf(1.0 - u)).float().to(theta.device)   # increasing in log σ
        r_mu = math.sqrt(n) * (mu - xbar) / sigma                                # increasing in μ
        return torch.cat([r_sigma, r_mu], dim=-1)                          # (n, 2)

    @property
    def marginal_cd_spec(self) -> dict:
        """Which coord is the scale nuisance (marginalize over) vs the location
        target, for MarginalCDRecovery. θ = (log σ, μ)."""
        return {"scale_coord": 0, "location_coord": 1}

    def analytic_marginal_cd_pit(self, theta_0, x: torch.Tensor) -> dict:
        """Closed-form marginal-CD PIT values at the true θ₀, the oracle the
        trained marginalization is checked against. Returns {sigma_pit, mu_pit},
        each (n,). σ²-CD = 1 − F_{χ²_{n−1}}((n−1)s²/σ₀²); μ-CD = F_{t_{n−1}}(√n(μ₀−X̄)/s)."""
        log_sigma0, mu0 = float(theta_0[0]), float(theta_0[1])
        sigma0 = math.exp(log_sigma0)
        xbar, s2 = self._suff_stats(x)
        xbar = xbar.squeeze(-1).detach().cpu().numpy()
        s2 = s2.squeeze(-1).detach().cpu().numpy()
        n = self.n_iid
        w0 = (n - 1) * s2 / sigma0 ** 2
        sigma_pit = 1.0 - chi2.cdf(w0, df=n - 1)
        mu_pit = t.cdf(math.sqrt(n) * (mu0 - xbar) / np.sqrt(s2), df=n - 1)
        return {
            "sigma_pit": torch.from_numpy(sigma_pit).float(),
            "mu_pit": torch.from_numpy(mu_pit).float(),
        }

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
        r_sigma = norm.ppf(1.0 - u)                          # increasing convention
        r_mu = math.sqrt(n) * (mu - xbar) / sigma
        # feature = (log s², X̄): |∂r_σ/∂ log s²| = w·f_χ²(w)/φ(r_σ);  |∂r_μ/∂X̄| = √n/σ
        dr_sigma_dlogs2 = w * chi2.pdf(w, df=n - 1) / np.clip(norm.pdf(r_sigma), 1e-30, None)
        dr_mu_dxbar = math.sqrt(n) / sigma
        log_det_feat = np.log(np.clip(dr_sigma_dlogs2, 1e-30, None)) + np.log(dr_mu_dxbar)
        log_det_contrib = _SUFFICIENT_STAT_LOG_DET_CONST
        loss = (0.5 * (r_sigma ** 2 + r_mu ** 2) + math.log(2 * math.pi)
                - log_det_feat - log_det_contrib)
        return float(loss.mean())
