"""CauchyLocScale: X = (X_1, …, X_{n_iid}) iid Cauchy(x0, γ); θ = (log γ, x0).

The LF2I-Score stress target for the no-sufficient-statistic regime (hardening
item 5, 2026-06-10): by Pitman–Koopman–Darmois (smooth densities, θ-independent
support), the Cauchy location-scale family admits NO sufficient statistic of
dimension bounded below the sample size — its minimal sufficient statistic is
the full order statistics. Every fixed-dimensional summary pipeline necessarily
discards information here; a score readout needs no summary at all.

Oracle pivot (`r_star`) — a closed-form, EXACTLY calibrated pivot used as the
measured coverage-floor reference:

    z_i = (X_i − x0)/γ  ~ iid standard Cauchy            (at the true θ)
    u_i = 1/2 + arctan(z_i)/π ~ iid U(0,1)               (Cauchy PIT)
    g_i = Φ⁻¹(u_i) ~ iid N(0,1)                          (normal scores)
    r_1 = Σ_i g_i / √n,   r_2 = Σ_i a_i g_i              (a = alternating ±1/√n)

The two projection vectors are orthonormal, so r(θ0; X) | θ0 ~ N(0, I_2)
EXACTLY for every θ0. NOTE: this is *a* calibrated pivot — the floor reference
for the evaluation engine — NOT an efficient one (none of fixed dimension can
be sufficient here, which is precisely the point of this target). Coverage of
{‖r‖² ≤ χ²_{2,α}} is exact up to Monte-Carlo error; set SHAPE/width from this
pivot carries no optimality claim.

Conventions follow `NormalUnknownMeanVar`: θ = (log-scale, location) on a
uniform box; `theta_range` exposes a scalar (lo, hi) cover for procedures that
take per-dim-agnostic bisection bounds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class CauchyLocScale:
    n_iid: int = 10
    log_gamma_range: Tuple[float, float] = (-1.0, 1.0)
    loc_range: Tuple[float, float] = (-3.0, 3.0)
    d_theta: int = 2

    @property
    def d_x(self) -> int:
        return self.n_iid

    @property
    def theta_lower(self) -> Tuple[float, float]:
        return (self.log_gamma_range[0], self.loc_range[0])

    @property
    def theta_upper(self) -> Tuple[float, float]:
        return (self.log_gamma_range[1], self.loc_range[1])

    @property
    def theta_range(self) -> Tuple[float, float]:
        """Scalar (lo, hi) covering both coordinates — the loosest per-dim
        bound, used by procedures that take a single bisection range."""
        return (
            min(self.log_gamma_range[0], self.loc_range[0]),
            max(self.log_gamma_range[1], self.loc_range[1]),
        )

    # --- sampling -----------------------------------------------------------

    def _draw_theta(self, n: int, rng: np.random.Generator) -> np.ndarray:
        log_gamma = rng.uniform(*self.log_gamma_range, size=(n, 1))
        loc = rng.uniform(*self.loc_range, size=(n, 1))
        return np.concatenate([log_gamma, loc], axis=1)

    def _draw_x(self, theta_np: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        gamma = np.exp(theta_np[:, 0:1])
        loc = theta_np[:, 1:2]
        z = rng.standard_cauchy(size=(theta_np.shape[0], self.n_iid))
        return loc + gamma * z

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_np = self._draw_theta(n, rng)
        x_np = self._draw_x(theta_np, rng)
        return torch.from_numpy(theta_np).float(), torch.from_numpy(x_np).float()

    def sample_x_given_theta(self, theta_0, n: int, rng: np.random.Generator) -> torch.Tensor:
        theta_vec = np.atleast_1d(np.asarray(theta_0, dtype=np.float64))
        assert theta_vec.shape == (self.d_theta,), (
            f"theta_0 has shape {theta_vec.shape}, expected ({self.d_theta},)"
        )
        theta_np = np.repeat(theta_vec[None, :], n, axis=0)
        return torch.from_numpy(self._draw_x(theta_np, rng)).float()

    # --- closed forms -------------------------------------------------------

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Σ_i log Cauchy(x_i; x0, γ) over the n_iid axis."""
        gamma = theta[..., 0:1].exp()
        loc = theta[..., 1:2]
        z = (x - loc) / gamma
        return (-math.log(math.pi) - theta[..., 0:1] - torch.log1p(z.pow(2))).sum(dim=-1)

    def r_star(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """The exactly-calibrated oracle pivot (see module docstring).

        Computed in float64 (the PIT→Φ⁻¹ composition is tail-sensitive); u is
        clamped to (1e-12, 1−1e-12) so extreme replicates map to finite normal
        scores (|Φ⁻¹| ≲ 7) rather than ±inf — a < 1e-11 calibration distortion,
        far below any Monte-Carlo floor in use.
        """
        n = self.n_iid
        th = theta.to(torch.float64)
        xx = x.to(torch.float64)
        gamma = th[..., 0:1].exp()
        loc = th[..., 1:2]
        z = (xx - loc) / gamma
        u = (0.5 + torch.atan(z) / math.pi).clamp(1e-12, 1.0 - 1e-12)
        g = torch.special.ndtri(u)                      # (.., n) iid N(0,1) at truth
        # Second projection: alternating ±1, Gram-Schmidt-orthogonalized against
        # the mean vector and renormalized — exactly orthonormal for ANY n_iid
        # (plain alternation is only ⊥ the mean for even n).
        ones = torch.ones(n, dtype=torch.float64, device=g.device) / math.sqrt(n)
        alt = torch.tensor(
            [1.0 if i % 2 == 0 else -1.0 for i in range(n)], dtype=torch.float64,
            device=g.device,
        )
        alt = alt - (alt @ ones) * ones
        alt = alt / alt.norm()
        r1 = g @ ones
        r2 = g @ alt
        return torch.stack([r1, r2], dim=-1).to(theta.dtype)
