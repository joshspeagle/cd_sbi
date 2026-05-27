"""ConfidenceSet dataclass."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class ConfidenceSet:
    """Confidence set at a single X_obs and confidence level α.

    `contains(theta_value)` → bool; `boundary_repr` is shape (2,) in 1D
    (lower, upper) and shape (K, d) in d > 1 — K boundary points sampled
    via ray-bisection from the set's center.
    """
    contains: Callable
    boundary_repr: torch.Tensor
    alpha: float
