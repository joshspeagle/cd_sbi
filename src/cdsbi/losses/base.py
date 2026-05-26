"""Loss protocol + MonotonicityMismatchError."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


class MonotonicityMismatchError(RuntimeError):
    """Raised when a Loss requires guarantees the Flow doesn't advertise."""


@runtime_checkable
class Loss(Protocol):
    required_guarantees: frozenset

    def check_guarantees(self, flow) -> None:
        """Raise MonotonicityMismatchError if flow.monotonicity_guarantees < self.required_guarantees."""
        ...

    def population_lower_bound(self, simulator) -> Optional[float]: ...
