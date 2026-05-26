"""ConfidenceProcedure subtypes — what every method's TrainedModel.procedure is.

Each subtype produces a ConfidenceSet via its own mechanism:
- PivotBased: chi-square inversion of ‖r‖² ≤ χ²_{d, α}
- CriticalValue: {θ : T(θ, X) ≤ c_α(θ)} (LF2I)
- PosteriorBased: highest-posterior-density region on sampled posterior (NPE)
- LikelihoodBased: Wilks-style likelihood-ratio inversion (NLE)
- RatioBased: ratio thresholding (NRE)
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable

import torch
from scipy.stats import chi2

from cdsbi.confidence_set.datatypes import ConfidenceSet
from cdsbi.confidence_set.root_find import bisect_1d


@runtime_checkable
class ConfidenceProcedure(Protocol):
    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet: ...


class PivotBasedProcedure:
    """CDSBI: 1D pivot inverted via chi-square."""

    def __init__(self, pivot_fn: Callable, d_theta: int, theta_range: tuple = (-20.0, 20.0)):
        self.pivot_fn = pivot_fn
        self.d_theta = d_theta
        self.theta_range = theta_range

    def pivot(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.pivot_fn(theta, x)

    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        assert self.d_theta == 1, "Multivariate confidence_set lands in v1+."
        thresh = float(chi2.ppf(alpha, df=1))

        def f(theta_val: float) -> float:
            theta = torch.tensor([[theta_val]], dtype=x_obs.dtype, device=x_obs.device)
            r = self.pivot_fn(theta, x_obs)
            return (r.pow(2).sum().item() - thresh)

        # Pivot is monotone in θ at fixed X_obs; r² is U-shaped with min where r=0.
        # Locate the minimum (where r ≈ 0) by bisecting r itself.
        def r_only(theta_val: float) -> float:
            theta = torch.tensor([[theta_val]], dtype=x_obs.dtype, device=x_obs.device)
            return self.pivot_fn(theta, x_obs).item()

        lo, hi = self.theta_range
        center = bisect_1d(r_only, lo, hi, tol=1e-4)
        # Now find left and right zeros of f(θ) = r(θ)² − thresh
        # On each side of the center, r is monotone, so f has a single zero
        left = bisect_1d(f, lo, center, tol=1e-4)
        right = bisect_1d(f, center, hi, tol=1e-4)

        def contains(theta_val: float) -> bool:
            return left <= theta_val <= right

        return ConfidenceSet(
            contains=contains,
            boundary_repr=torch.tensor([left, right]),
            alpha=alpha,
        )


class CriticalValueProcedure:
    """LF2I: {θ : T(θ, X) ≤ c_α(θ)}."""

    def __init__(
        self,
        test_stat_fn: Callable,
        critical_value_fn: Callable,
        d_theta: int,
        theta_range: tuple = (-20.0, 20.0),
    ):
        self.test_stat_fn = test_stat_fn
        self.critical_value_fn = critical_value_fn
        self.d_theta = d_theta
        self.theta_range = theta_range

    def test_statistic(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.test_stat_fn(theta, x)

    def critical_value(self, theta: torch.Tensor, alpha: float) -> torch.Tensor:
        return self.critical_value_fn(theta, alpha)

    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        assert self.d_theta == 1

        def f(theta_val: float) -> float:
            theta = torch.tensor([[theta_val]], dtype=x_obs.dtype, device=x_obs.device)
            t = self.test_stat_fn(theta, x_obs).item()
            c = self.critical_value_fn(theta, alpha).item()
            return t - c  # negative ⇒ in set; positive ⇒ outside

        lo, hi = self.theta_range
        grid = torch.linspace(lo, hi, 200).tolist()
        signs = [f(t) <= 0 for t in grid]
        try:
            i_first = signs.index(True)
            i_last = len(signs) - 1 - signs[::-1].index(True)
        except ValueError:
            return ConfidenceSet(
                contains=lambda th: False, boundary_repr=torch.tensor([0.0, 0.0]), alpha=alpha
            )
        left_lo, left_hi = grid[max(i_first - 1, 0)], grid[i_first]
        right_lo, right_hi = grid[i_last], grid[min(i_last + 1, len(grid) - 1)]
        left = bisect_1d(f, left_lo, left_hi, tol=1e-4) if i_first > 0 else grid[0]
        right = bisect_1d(f, right_lo, right_hi, tol=1e-4) if i_last < len(grid) - 1 else grid[-1]

        def contains(theta_val: float) -> bool:
            return left <= theta_val <= right

        return ConfidenceSet(
            contains=contains, boundary_repr=torch.tensor([left, right]), alpha=alpha
        )


class PosteriorBasedProcedure:
    """NPE: highest-posterior-density region via posterior samples."""

    def __init__(self, sample_fn: Callable, d_theta: int):
        self.sample_fn = sample_fn
        self.d_theta = d_theta

    def posterior_samples(self, x_obs: torch.Tensor, n: int = 10_000) -> torch.Tensor:
        return self.sample_fn(x_obs, n)

    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        from cdsbi.confidence_set.hpd import hpd_1d
        samples = self.sample_fn(x_obs, 10_000).flatten()
        return hpd_1d(samples, alpha=alpha)


class LikelihoodBasedProcedure:
    """NLE: Wilks-style likelihood-ratio confidence set."""

    def __init__(self, log_likelihood_fn: Callable, d_theta: int, theta_range: tuple = (-20.0, 20.0)):
        self.log_likelihood_fn = log_likelihood_fn
        self.d_theta = d_theta
        self.theta_range = theta_range

    def log_likelihood(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.log_likelihood_fn(theta, x)

    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        assert self.d_theta == 1
        thresh = float(chi2.ppf(alpha, df=1))

        lo, hi = self.theta_range
        grid = torch.linspace(lo, hi, 200).view(-1, 1).to(x_obs.device)
        lls = torch.stack([self.log_likelihood_fn(g.view(1, 1), x_obs).squeeze() for g in grid])
        ll_max = lls.max().item()

        def f(theta_val: float) -> float:
            theta = torch.tensor([[theta_val]], dtype=x_obs.dtype, device=x_obs.device)
            ll = self.log_likelihood_fn(theta, x_obs).item()
            return 2 * (ll_max - ll) - thresh

        grid_f = [2 * (ll_max - ll.item()) - thresh for ll in lls]
        signs = [g <= 0 for g in grid_f]
        try:
            i_first = signs.index(True)
            i_last = len(signs) - 1 - signs[::-1].index(True)
        except ValueError:
            return ConfidenceSet(
                contains=lambda th: False, boundary_repr=torch.tensor([0.0, 0.0]), alpha=alpha
            )
        gs = grid.flatten().tolist()
        left_lo, left_hi = gs[max(i_first - 1, 0)], gs[i_first]
        right_lo, right_hi = gs[i_last], gs[min(i_last + 1, len(gs) - 1)]
        left = bisect_1d(f, left_lo, left_hi, tol=1e-4) if i_first > 0 else gs[0]
        right = bisect_1d(f, right_lo, right_hi, tol=1e-4) if i_last < len(gs) - 1 else gs[-1]

        def contains(theta_val: float) -> bool:
            return left <= theta_val <= right

        return ConfidenceSet(
            contains=contains, boundary_repr=torch.tensor([left, right]), alpha=alpha
        )


class RatioBasedProcedure:
    """NRE: ratio thresholding — same shape as Likelihood but using log-ratio."""

    def __init__(self, log_ratio_fn: Callable, d_theta: int, theta_range: tuple = (-20.0, 20.0)):
        self.log_ratio_fn = log_ratio_fn
        self.d_theta = d_theta
        self.theta_range = theta_range

    def log_ratio(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.log_ratio_fn(theta, x)

    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        wrapper = LikelihoodBasedProcedure(self.log_ratio_fn, self.d_theta, self.theta_range)
        return wrapper.confidence_set(x_obs, alpha)
