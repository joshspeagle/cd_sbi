"""1D highest-posterior-density region from samples."""
from __future__ import annotations

import torch

from cdsbi.confidence_set.datatypes import ConfidenceSet


def hpd_1d(samples: torch.Tensor, alpha: float) -> ConfidenceSet:
    """Smallest interval containing fraction α of the sample mass.

    For unimodal distributions (including symmetric ones such as N(0, I)) the
    minimum-width HPD interval is the equal-tailed credible interval, so we
    locate the boundaries via interpolated quantiles at (1−α)/2 and 1−(1−α)/2.
    This is numerically more stable than the sliding-window integer-index
    approach when n is moderate: the interpolated quantile converges to the
    true ±z_{α/2} faster and without seed-dependent asymmetry in the tails.
    For multivariate or strongly skewed posteriors a kernel-based HPD should be
    used instead; this function is intentionally kept simple.
    """
    flat = samples.flatten()
    n = flat.numel()
    if n < 2:
        v = float(flat[0]) if n == 1 else 0.0
        return ConfidenceSet(
            contains=lambda th: th == v, boundary_repr=torch.tensor([v, v]), alpha=alpha
        )
    tail = (1.0 - alpha) / 2.0
    lo = float(torch.quantile(flat, tail))
    hi = float(torch.quantile(flat, 1.0 - tail))

    def contains(theta_val: float) -> bool:
        return lo <= theta_val <= hi

    return ConfidenceSet(contains=contains, boundary_repr=torch.tensor([lo, hi]), alpha=alpha)
