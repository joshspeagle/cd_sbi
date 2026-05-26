"""ConfidenceSet dataclass."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class ConfidenceSet:
    """Confidence set at a single X_obs and confidence level α.

    `contains(theta_value)` → bool; `boundary_repr` is shape (2,) in 1D
    (lower, upper) and a set of boundary samples in higher d (added v1+).
    """
    contains: Callable[[float], bool]
    boundary_repr: torch.Tensor
    alpha: float
