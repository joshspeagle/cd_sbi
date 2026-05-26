"""NF-MLE loss: ½ ‖r‖² − log |∂r/∂X|, with monotonicity-guarantee check."""
from __future__ import annotations

import math
from typing import Optional

import torch

from cdsbi.flows.base import Guarantee
from cdsbi.losses.base import Loss, MonotonicityMismatchError


class NFMLELoss(Loss):
    required_guarantees = frozenset({Guarantee.R1, Guarantee.R2})

    def __call__(self, r: torch.Tensor, log_det_jac_input: torch.Tensor) -> torch.Tensor:
        """Per-sample loss: ½ ‖r‖² + ½ d log(2π) − log |∂r/∂X|, then averaged.

        The ½ d log(2π) term is the standard-normal normalization for r;
        we include it so loss values are directly comparable to the
        conditional-entropy lower bound (Theorem 3.2).
        """
        d = r.shape[-1]
        per_sample = 0.5 * r.pow(2).sum(dim=-1) + 0.5 * d * math.log(2 * math.pi) - log_det_jac_input
        return per_sample.mean()

    def check_guarantees(self, flow) -> None:
        guarantees = getattr(flow, "monotonicity_guarantees", frozenset())
        if not self.required_guarantees.issubset(guarantees):
            missing = self.required_guarantees - guarantees
            raise MonotonicityMismatchError(
                f"NFMLELoss requires {sorted(g.value for g in self.required_guarantees)}; "
                f"flow {type(flow).__name__} provides {sorted(g.value for g in guarantees)}; "
                f"missing {sorted(g.value for g in missing)}."
            )

    def population_lower_bound(self, simulator) -> Optional[float]:
        return simulator.entropy_lower_bound()
