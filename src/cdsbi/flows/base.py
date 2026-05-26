"""Flow protocol + monotonicity-guarantee enum."""
from __future__ import annotations

from enum import Enum
from typing import Optional, Protocol, Tuple, runtime_checkable

import torch


class Guarantee(str, Enum):
    R1 = "R1"  # monotone in θ
    R2 = "R2"  # monotone in X (more precisely: invertible in X)


@runtime_checkable
class Flow(Protocol):
    monotonicity_guarantees: frozenset

    def forward(
        self, theta: torch.Tensor, context: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (r, log_det_jac_input). r has shape (n, d); log_det_jac_input has shape (n,)."""
        ...

    def n_params(self) -> int: ...
