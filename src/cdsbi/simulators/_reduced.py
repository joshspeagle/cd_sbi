"""ReducedSimulator: wraps a base simulator with a known sufficient-statistic reduction.

For methods that do NOT do NF-MLE-style change-of-variables on X
(NPE, NLE, NRE, LF2I-BFF), the conditioner's log|∂T/∂X| Jacobian term
is irrelevant for training — they just need the data in the reduced
space. This wrapper pre-applies the reduction so those methods receive
`(θ, T)` from `sample()` / `sample_x_given_theta()` and the underlying
flow / classifier is built with the right (smaller) input dim.

CDSBI continues to use the BASE simulator + its own MLPConditioner —
NFMLELoss needs the Jacobian term in the loss, and that comes from
the conditioner.encode() call inside CDSBIRunner.fit().

This wrapper exists so the v3 §8.4 cross-method comparison is
apples-to-apples: all methods see the sufficient statistic T, and any
remaining gap reflects the architectural prescription (the R1+R2
pivot vs. Bayesian objects / likelihood ratios / critical-value
calibration), not the data preprocessing.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from cdsbi.conditioners.mlp import MLPConditioner


class ReducedSimulator:
    """Wraps a base simulator and pre-applies a frozen-sum X → T reduction.

    Exposes the same Simulator protocol but with `d_x = 1` (T is scalar) and
    `sample()` / `sample_x_given_theta()` returning the reduced statistic.
    `r_star()` delegates to the base — ExponentialRate's r_star accepts both
    raw X and pre-reduced T via its dual-shape interface, so callers see the
    same answers either way.
    """

    def __init__(self, base, conditioner: MLPConditioner):
        self._base = base
        self._conditioner = conditioner
        # Required Simulator attributes
        self.d_theta = base.d_theta
        self.theta_range = base.theta_range

    @property
    def d_x(self) -> int:
        # After reduction, X-dim is the conditioner's output_dim.
        return int(self._conditioner.output_dim)

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        theta, x = self._base.sample(n, rng)
        T, _ = self._conditioner.encode(x)
        return theta, T

    def sample_x_given_theta(self, theta_0, n: int, rng: np.random.Generator) -> torch.Tensor:
        x = self._base.sample_x_given_theta(theta_0, n, rng)
        T, _ = self._conditioner.encode(x)
        return T

    def r_star(self, theta: torch.Tensor, x_or_T: torch.Tensor) -> torch.Tensor:
        # Base simulator's r_star already accepts either shape (dual-interface).
        return self._base.r_star(theta, x_or_T)

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self._base.log_prob(x, theta)

    def entropy_lower_bound(self) -> float:
        return self._base.entropy_lower_bound()

    def __getattr__(self, name: str):
        return getattr(self._base, name)
