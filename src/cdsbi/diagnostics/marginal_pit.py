"""MarginalPIT: KS test of Φ(r(θ, X)) vs U(0,1) on training-distribution samples."""
from __future__ import annotations

import torch
from scipy.stats import kstest, norm

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.base import Diagnostic, DiagnosticResult
from cdsbi.diagnostics.ks_floor import ks_noise_floor


class MarginalPIT(Diagnostic):
    name = "marginal_pit"

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
        theta, x = eval_data
        with torch.no_grad():
            r = trained.procedure.pivot(theta, x).flatten()
        u = norm.cdf(r.cpu().numpy())
        ks_stat, _ = kstest(u, "uniform")
        floor = ks_noise_floor(N=u.size, n_bins=1)
        return DiagnosticResult(
            name=self.name,
            value=float(ks_stat),
            passed=ks_stat <= floor,
            noise_floor=floor,
            n_samples=u.size,
        )
