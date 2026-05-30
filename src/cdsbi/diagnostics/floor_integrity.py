"""FloorIntegrity: did the trained objective respect its lower bound, or cheat?

Arm-aware. For an NF-MLE loss the conditional-entropy floor H = E[loss at r*]
applies: cheating = final_loss < H − tol (the §3.5 folding / Stage-B collapse
pathology). For a non-likelihood loss (e.g. the energy calibration loss) there is
no H to undercut, so the floor comparison is N/A (cheats=False, pass).
"""
from __future__ import annotations

import math

import pandas as pd

from cdsbi.diagnostics.base import DiagnosticResult

# loss_class -> the simulator method giving its conditional-entropy floor
_FLOOR_METHOD_BY_LOSS = {
    "NFMLELoss": "entropy_lower_bound",
    "ExactDensityLoss": "data_entropy_lower_bound",
}


class FloorIntegrity:
    name = "floor_integrity"

    def __init__(self, tol: float = 0.10):
        self.tol = tol

    def __call__(self, trained, simulator, eval_data=None, x_per_theta=None) -> DiagnosticResult:
        loss_class = getattr(trained, "arch_metadata", {}).get("loss_class", "")
        final_loss = float(getattr(trained, "final_loss", float("nan")))
        floor_method = _FLOOR_METHOD_BY_LOSS.get(loss_class)
        applicable = floor_method is not None and hasattr(simulator, floor_method)
        if applicable:
            H = float(getattr(simulator, floor_method)())
            margin = final_loss - H
            cheats = bool(margin < -self.tol)
            meta = {"loss_class": loss_class, "tol": self.tol, "floor_method": floor_method}
        else:
            H = float("nan")
            margin = float("nan")
            cheats = False
            meta = {"reason": f"floor N/A for loss {loss_class!r}"}
        df = pd.DataFrame([{
            "final_loss": final_loss, "entropy_floor": H,
            "floor_margin": margin, "cheats": cheats,
        }])
        return DiagnosticResult(self.name, value=df, passed=(not cheats),
                                noise_floor=0.0, n_samples=0, meta=meta)
