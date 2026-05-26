"""1D equal-tailed credible interval from samples."""
from __future__ import annotations

import torch

from cdsbi.confidence_set.datatypes import ConfidenceSet


def equal_tailed_1d(samples: torch.Tensor, alpha: float) -> ConfidenceSet:
    """Equal-tailed credible interval at level α from 1D samples.

    Returns the interval [q_{(1−α)/2}, q_{1−(1−α)/2}] using interpolated
    sample quantiles. This is *not* the highest-posterior-density (HPD)
    interval in general — for skewed posteriors the two differ. A true
    kernel-based HPD function can be added as a sibling in v1+ when skewed
    posteriors enter the picture.
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
