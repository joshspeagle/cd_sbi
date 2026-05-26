"""Simulator protocol."""
from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class Simulator(Protocol):
    """Generates (θ, X | θ) pairs. Optionally exposes analytical r*, log p, entropy floor."""

    d_theta: int
    d_x: int

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (theta, x) with shapes (n, d_theta) and (n, d_x)."""
        ...

    def r_star(self, theta: torch.Tensor, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Analytical canonical pivot if known; None otherwise."""
        ...

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> Optional[torch.Tensor]:
        """Analytical conditional log p(x | theta) if known; None otherwise."""
        ...

    def entropy_lower_bound(self) -> Optional[float]:
        """E_ρ[H(X | θ)] when known in closed form; None otherwise."""
        ...
