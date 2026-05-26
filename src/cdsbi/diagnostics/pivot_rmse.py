"""PivotRMSE: RMSE of trained pivot vs analytical r* on eval data."""
from __future__ import annotations

import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.base import Diagnostic, DiagnosticResult


class PivotRMSE(Diagnostic):
    name = "pivot_rmse"

    def __call__(self, trained, simulator, eval_data) -> DiagnosticResult:
        if not isinstance(trained.procedure, PivotBasedProcedure):
            return DiagnosticResult(
                name=self.name,
                value=float("nan"),
                passed=True,
                noise_floor=0.0,
                n_samples=0,
                meta={"reason": "not a pivot-based procedure"},
            )
        if simulator.r_star is None:
            return DiagnosticResult(
                name=self.name,
                value=float("nan"),
                passed=True,
                noise_floor=0.0,
                n_samples=0,
                meta={"reason": "no analytical r*"},
            )
        theta, x = eval_data
        with torch.no_grad():
            r_hat = trained.procedure.pivot(theta, x)
            r_star = simulator.r_star(theta, x).to(r_hat.device)
        rmse = (r_hat - r_star).pow(2).mean().sqrt().item()
        return DiagnosticResult(
            name=self.name,
            value=rmse,
            passed=rmse < 0.05,
            noise_floor=0.05,
            n_samples=theta.shape[0],
        )
