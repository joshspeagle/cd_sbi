"""ExponentialRate: X_i ~ Exp(θ) iid (n=5), θ ~ U[0.3, 3.0].

Sufficient statistic T = Σ X_i ~ Gamma(n, 1/θ); equivalently 2θT ~ χ²_{2n}.
Truth pivot per manuscript §8.4: r*(θ, T) = Φ⁻¹(F_{χ²_{2n}}(2θT)).
This is the v3 non-additive target — the multiplicative θT interaction
takes the model outside the additive class of §8.1–§8.3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from scipy.stats import chi2, norm


@dataclass
class ExponentialRate:
    theta_range: Tuple[float, float] = (0.3, 3.0)
    n_iid: int = 5
    d_theta: int = 1

    @property
    def d_x(self) -> int:
        return self.n_iid

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = self.theta_range
        theta_np = rng.uniform(a, b, size=(n, self.d_theta))
        # X_i ~ Exp(θ) — scale parameter 1/θ. numpy's Exponential takes scale.
        x_np = rng.exponential(scale=1.0 / theta_np, size=(n, self.n_iid))
        return (
            torch.from_numpy(theta_np).float(),
            torch.from_numpy(x_np).float(),
        )

    def sample_x_given_theta(self, theta_0, n: int, rng: np.random.Generator) -> torch.Tensor:
        """Draw n samples of X = (X_1, …, X_n_iid) conditional on θ = θ_0."""
        theta_vec = np.atleast_1d(np.asarray(theta_0, dtype=np.float64))
        assert theta_vec.shape == (self.d_theta,), (
            f"theta_0 has shape {theta_vec.shape}, expected ({self.d_theta},)"
        )
        x_np = rng.exponential(scale=1.0 / float(theta_vec[0]), size=(n, self.n_iid))
        return torch.from_numpy(x_np).float()

    def r_star(self, theta: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Truth pivot on the (θ, T) sufficient-statistic space.

        r*(θ, T) = Φ⁻¹(F_{χ²_{2n}}(2θT)).
        Inputs:
          theta: (n, 1) — parameter samples
          T:     (n, 1) — sufficient statistic Σ X_i
        Returns: (n, 1)
        """
        df = 2 * self.n_iid
        # Compute via scipy on CPU then move to theta's device. The truth pivot
        # is only used by PivotRMSE and diagnostics, never inside autograd loops.
        twothT = (2.0 * theta * T).detach().cpu().numpy()
        u = chi2.cdf(twothT, df=df)
        # Clamp away from {0, 1} for the inverse-Φ; the diagnostic eval support
        # is well inside (0, 1) for the simulator's θ range, but be defensive.
        u = np.clip(u, 1e-10, 1.0 - 1e-10)
        r_np = norm.ppf(u)
        return torch.from_numpy(r_np).float().to(theta.device)

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """log p(X_1, …, X_n_iid | θ) = Σ_i log p(X_i | θ); X_i ~ Exp(θ)."""
        # log p(X_i | θ) = log θ − θ X_i for X_i > 0
        # x: (n, n_iid); theta: (n, 1)
        return (torch.log(theta) - theta * x).sum(dim=-1)

    def entropy_lower_bound(self, n_mc: int = 50000, seed: int = 42) -> float:
        """Monte-Carlo estimate of E[NF-MLE loss at truth r*] on the (θ, T)
        space with the conditioner's log|∂T/∂X| accounted for.

        The NF-MLE loss the trainer sees with X→T conditioner is
            L = ½ r² + ½ log(2π) − log|∂r/∂T| − log|∂T/∂X|.
        At r = r*(θ, T), the expected value of L is the conditional-entropy
        lower bound that any valid normalized surrogate must satisfy
        (Theorem 3.2 / §3.2). Per manuscript §8.4, this evaluates to ≈ 0.88
        for the exponential-rate model with n=5 and θ ~ U[0.3, 3.0].

        Implemented via MC rather than a closed form because the analytic
        derivation requires expectations over (θ, T) involving log f_{χ²}
        and log φ that don't simplify cleanly; MC is a faithful evaluation
        of the same loss function the trainer optimizes.
        """
        rng = np.random.default_rng(seed)
        theta_np = rng.uniform(*self.theta_range, size=(n_mc, 1))
        z_np = rng.standard_normal(size=(n_mc, self.n_iid))
        # X_i ~ Exp(θ): scale = 1/θ
        x_np = rng.exponential(scale=1.0 / theta_np, size=(n_mc, self.n_iid))
        T_np = x_np.sum(axis=-1, keepdims=True)  # (n_mc, 1)
        df = 2 * self.n_iid
        twothT_np = 2.0 * theta_np * T_np
        u_np = chi2.cdf(twothT_np, df=df)
        u_np = np.clip(u_np, 1e-10, 1.0 - 1e-10)
        r_star_np = norm.ppf(u_np)  # (n_mc, 1)
        # ∂r*/∂T = 2θ · f_{χ²_{2n}}(2θT) / φ(r*) by chain rule + inverse-CDF
        f_chi2_np = chi2.pdf(twothT_np, df=df)
        phi_r_np = norm.pdf(r_star_np)
        dr_dT_np = (2.0 * theta_np * f_chi2_np) / np.clip(phi_r_np, 1e-30, None)
        log_dr_dT_np = np.log(np.clip(dr_dT_np, 1e-30, None))
        log_dT_dX = 0.5 * math.log(self.n_iid)
        loss_per_sample = (
            0.5 * (r_star_np ** 2)
            + 0.5 * math.log(2 * math.pi)
            - log_dr_dT_np
            - log_dT_dX
        )
        return float(loss_per_sample.mean())
