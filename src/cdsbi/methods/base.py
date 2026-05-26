"""Method.Runner base + TrainedModel + n_params kind conventions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from cdsbi.confidence_set.procedures import ConfidenceProcedure


class BudgetUnreachableError(RuntimeError):
    """Raised when no candidate width lands within ±15% of target_params."""


@dataclass
class TrainedModel:
    procedure: ConfidenceProcedure
    state_dict: dict
    final_loss: float
    n_steps: int
    wall_clock_sec: float
    arch_metadata: dict = field(default_factory=dict)


@runtime_checkable
class Runner(Protocol):
    def fit(self, simulator, config, seed: int) -> TrainedModel: ...
    def n_params(self) -> dict:
        """{'backbone': int, 'head': int, 'calibration_stage': int, 'total': int, 'kind': str}.

        Every field always populated (zero when not applicable):
        - kind='flow' (CDSBI, NPE, NLE): backbone = flow weights; head = scalar α / scale params
        - kind='classifier' (NRE): head = MLP; backbone = 0
        - kind='two_stage' (LF2I): backbone = test-stat flow; calibration_stage = quantile head;
                                   head = 0
        """
        ...
