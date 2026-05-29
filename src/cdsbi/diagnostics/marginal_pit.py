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
                name=self.name, value=float("nan"), passed=True, noise_floor=0.0,
                n_samples=0, meta={"reason": "not a pivot-based procedure"},
            )
        # F6: accept either 2-tuple (theta, x) or 3-tuple (theta, x, r).
        if len(eval_data) == 3 and eval_data[2] is not None:
            theta, x, r = eval_data
        else:
            theta, x = eval_data[:2]
            with torch.no_grad():
                r = trained.procedure.pivot(theta, x)  # (N, d)
        r_np = r.cpu().numpy()
        d = r_np.shape[-1] if r_np.ndim > 1 else 1
        floor = ks_noise_floor(N=r_np.shape[0], n_bins=1)
        if d == 1:
            u = norm.cdf(r_np.flatten())
            ks_stat, _ = kstest(u, "uniform")
            return DiagnosticResult(
                name=self.name, value=float(ks_stat),
                passed=ks_stat <= floor, noise_floor=floor, n_samples=u.size,
                meta={"pit_u": u, "d": 1},
            )
        rows = []
        for k in range(d):
            u_k = norm.cdf(r_np[:, k])
            ks_stat, _ = kstest(u_k, "uniform")
            rows.append({
                "coord": int(k),
                "ks": float(ks_stat),
                "noise_floor": floor,
                "passed": ks_stat <= floor,
                "n_samples": int(u_k.size),
            })
        import numpy as _np
        u_all = _np.column_stack([norm.cdf(r_np[:, k]) for k in range(d)])
        import pandas as pd
        df = pd.DataFrame(rows)
        return DiagnosticResult(
            name=self.name, value=df,
            passed=bool(df["passed"].all()), noise_floor=floor,
            n_samples=r_np.shape[0], meta={"d": int(d), "pit_u": u_all},
        )
