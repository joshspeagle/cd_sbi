"""JointMahalanobis: at each θ_0, the empirical distribution of
||r(θ_0; X_i)||² for X_i ~ p(X | θ_0) should match χ²_d.

Diagnostic 4 from §7.3 of the manuscript. Degenerate in d=1 (reduces
to the squared marginal-PIT residual against a χ²_1 distribution),
so we no-op in 1D.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2, kstest

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.base import Diagnostic, DiagnosticResult
from cdsbi.diagnostics.ks_floor import ks_noise_floor


class JointMahalanobis(Diagnostic):
    name = "joint_mahalanobis"

    def __init__(self, theta_0_grid: Sequence, n_per_theta: int = 2000):
        self.theta_0_grid = theta_0_grid
        self.n_per_theta = n_per_theta

    def __call__(self, trained, simulator, eval_data=None) -> DiagnosticResult:
        # Diagnostic only applies to pivot-based procedures.
        if not isinstance(trained.procedure, PivotBasedProcedure):
            return DiagnosticResult(
                name=self.name, value=float("nan"), passed=True, noise_floor=0.0,
                n_samples=0, meta={"reason": "not a pivot-based procedure"},
            )
        d = simulator.d_theta
        if d < 2:
            return DiagnosticResult(
                name=self.name, value=float("nan"), passed=True, noise_floor=0.0,
                n_samples=0, meta={"reason": "degenerate in d=1; use marginal PIT"},
            )
        floor = ks_noise_floor(N=self.n_per_theta, n_bins=1)
        rng = np.random.default_rng(0)
        rows = []
        chi2_cdf = chi2(df=d).cdf
        for theta_0 in self.theta_0_grid:
            x = simulator.sample_x_given_theta(theta_0, self.n_per_theta, rng)
            theta_vec = torch.tensor(list(theta_0), dtype=x.dtype).view(1, -1)
            theta_t = theta_vec.expand_as(x)
            with torch.no_grad():
                r = trained.procedure.pivot(theta_t, x)
            r_sq = r.pow(2).sum(dim=-1).cpu().numpy()
            ks_stat, _ = kstest(r_sq, chi2_cdf)
            rows.append({
                "theta_0_repr": str(list(map(float, list(theta_0)))),
                "ks": float(ks_stat),
                "noise_floor": floor,
                "n_per_theta": int(self.n_per_theta),
                "passed": ks_stat <= floor,
            })
        df = pd.DataFrame(rows)
        passed = bool(df["passed"].all())
        return DiagnosticResult(
            name=self.name, value=df, passed=passed, noise_floor=floor,
            n_samples=len(self.theta_0_grid) * self.n_per_theta,
            meta={"d": int(d)},
        )
