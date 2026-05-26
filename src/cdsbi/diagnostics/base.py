"""Diagnostic protocol + DiagnosticResult."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable, Union

import pandas as pd


@dataclass
class DiagnosticResult:
    name: str
    value: Union[float, pd.Series, pd.DataFrame]
    passed: bool
    noise_floor: float
    n_samples: int
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Diagnostic(Protocol):
    name: str

    def __call__(self, trained, simulator, eval_data) -> DiagnosticResult: ...
