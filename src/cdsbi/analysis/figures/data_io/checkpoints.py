"""Load model.pt checkpoints for figures that need training internals.

Verified shape of model.pt as written by cdsbi.experiments.run:
    {"arch_metadata": {... "loss_history_tail": [floats] ...}, "final_loss": float}

There is NO state_dict / weights saved. Figures needing trained weights
(C1, C3) cannot be driven from these checkpoints — see the F0 plan's
"Known gap for later milestones" note.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CheckpointData:
    """What a model.pt actually contains, plus a convenience accessor."""
    arch_metadata: dict
    final_loss: float

    @property
    def loss_history_tail(self) -> list[float] | None:
        """Last ~100 training-loss values, or None if not recorded."""
        return self.arch_metadata.get("loss_history_tail")


def load_checkpoint(run_dir: str) -> CheckpointData:
    """Load `<run_dir>/model.pt`. Raises FileNotFoundError if absent."""
    path = Path(run_dir) / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"No model.pt at {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return CheckpointData(
        arch_metadata=raw.get("arch_metadata", {}),
        final_loss=float(raw["final_loss"]),
    )
