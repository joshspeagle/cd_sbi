"""SetSize: distribution of confidence-set widths at multiple (θ_0, α).

Complements Coverage: Coverage measures *inclusion* of the true θ_0 by the
random set C_α(X_obs); SetSize measures the *size* of that set. Together
they discriminate "right coverage with bad power" (wide sets) from "right
coverage with good power" (narrow sets) — the failure mode that Coverage
alone cannot see.

For 1D, "size" is the width `right - left` of the confidence interval.
Multivariate sizing (volume / area / boundary-sample dispersion) lands
with v1+ flows.

Fast path: PivotBasedProcedure exposes `confidence_set_batch` which
returns (left, right) tensors across an X_obs batch in a single vectorized
bisection. Other procedures fall back to a per-X_obs `confidence_set` loop
until Phase B adds equivalent batched paths.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.base import Diagnostic, DiagnosticResult


class SetSize(Diagnostic):
    name = "set_size"

    def __init__(self, theta_0_grid: List[float], alpha_grid: List[float], n_per_theta: int = 200):
        self.theta_0_grid = theta_0_grid
        self.alpha_grid = alpha_grid
        self.n_per_theta = n_per_theta

    def __call__(self, trained, simulator, eval_data=None) -> DiagnosticResult:
        rng = np.random.default_rng(0)
        rows = []
        for theta_0 in self.theta_0_grid:
            # Same X | θ_0 sampling Coverage uses (hardcoded LocationNormal1D form;
            # v1+ simulators will need a sample_x_given_theta hook).
            theta_t = torch.full((self.n_per_theta, 1), float(theta_0), dtype=torch.float32)
            eps = rng.standard_normal(size=(self.n_per_theta, 1))
            x = theta_t + torch.from_numpy(eps).float()
            for alpha in self.alpha_grid:
                if isinstance(trained.procedure, PivotBasedProcedure):
                    # Fast path: single vectorized 3-stage bisection across batch.
                    with torch.no_grad():
                        left, right = trained.procedure.confidence_set_batch(x, alpha)
                    widths = (right - left).detach().cpu().numpy()
                else:
                    widths = np.empty(self.n_per_theta, dtype=np.float64)
                    for i in range(self.n_per_theta):
                        cs = trained.procedure.confidence_set(x[i : i + 1], alpha=alpha)
                        widths[i] = float(cs.boundary_repr[1] - cs.boundary_repr[0])
                rows.append({
                    "theta_0_0": float(theta_0),
                    "alpha": float(alpha),
                    "mean_width": float(widths.mean()),
                    "median_width": float(np.median(widths)),
                    "p90_width": float(np.quantile(widths, 0.9)),
                    "min_width": float(widths.min()),
                    "max_width": float(widths.max()),
                    "n_eval": int(self.n_per_theta),
                })
        df = pd.DataFrame(rows)
        return DiagnosticResult(
            name=self.name,
            value=df,
            passed=True,
            noise_floor=0.0,
            n_samples=len(self.theta_0_grid) * self.n_per_theta,
            meta={"n_thetas": len(self.theta_0_grid), "n_alphas": len(self.alpha_grid)},
        )
