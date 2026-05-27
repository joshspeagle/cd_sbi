"""ConditionalPIT: per-θ-bin KS test of Φ(r(θ, X)) | θ vs U(0,1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import kstest, norm

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.base import Diagnostic, DiagnosticResult
from cdsbi.diagnostics.ks_floor import ks_noise_floor


class ConditionalPIT(Diagnostic):
    name = "conditional_pit"

    def __init__(self, n_bins: int):
        self.n_bins = n_bins

    def _per_bin_floor(self, N: int) -> float:
        return ks_noise_floor(N=N, n_bins=self.n_bins)

    def __call__(self, trained, simulator, eval_data) -> DiagnosticResult:
        if not isinstance(trained.procedure, PivotBasedProcedure):
            return DiagnosticResult(
                name=self.name, value=float("nan"), passed=True, noise_floor=0.0,
                n_samples=0, meta={"reason": "not a pivot-based procedure"},
            )
        # F6: accept either 2-tuple (theta, x) or 3-tuple (theta, x, r).
        if len(eval_data) == 3 and eval_data[2] is not None:
            theta, x, r_t = eval_data
            r = r_t.detach().cpu().numpy()
        else:
            theta, x = eval_data[:2]
            with torch.no_grad():
                r = trained.procedure.pivot(theta, x).cpu().numpy()  # (N, d)
        N = theta.shape[0]
        floor = self._per_bin_floor(N)
        theta_np = theta.cpu().numpy()
        d = theta_np.shape[-1] if theta_np.ndim > 1 else 1
        rows = []
        for c in range(d):
            theta_c = theta_np[:, c] if d > 1 else theta_np.flatten()
            r_c = r[:, c] if r.ndim > 1 else r.flatten()
            edges = np.quantile(theta_c, np.linspace(0, 1, self.n_bins + 1))
            for k in range(self.n_bins):
                lo, hi = edges[k], edges[k + 1]
                mask = (theta_c >= lo) & (theta_c <= hi)
                r_bin = r_c[mask]
                if r_bin.size < 10:
                    continue
                u_bin = norm.cdf(r_bin)
                ks_stat, _ = kstest(u_bin, "uniform")
                rows.append({
                    "coord": int(c),
                    "theta_0_bin": k,
                    "theta_0_center_0": 0.5 * (lo + hi),
                    "ks": ks_stat,
                    "per_bin_noise_floor": floor,
                    "n_per_bin": int(r_bin.size),
                    "passed": ks_stat <= floor,
                })
        df = pd.DataFrame(rows)
        passed = bool(df["passed"].all())
        return DiagnosticResult(
            name=self.name, value=df, passed=passed, noise_floor=floor,
            n_samples=N, meta={"n_bins": self.n_bins, "d": int(d)},
        )
