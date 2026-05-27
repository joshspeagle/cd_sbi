"""SetSize: distribution of confidence-set widths at multiple (θ_0, α).

Complements Coverage: Coverage measures *inclusion* of the true θ_0 by the
random set C_α(X_obs); SetSize measures the *size* of that set. Together
they discriminate "right coverage with bad power" (wide sets) from "right
coverage with good power" (narrow sets) — the failure mode that Coverage
alone cannot see.

For 1D, "size" is the width `right - left` of the confidence interval.
Multivariate sizing (volume / area / boundary-sample dispersion) lands
with v1+ flows.

Fast path: any procedure that exposes `confidence_set_batch` (returning
(left, right) tensors across an X_obs batch) takes the vectorized path;
otherwise SetSize falls back to a per-X_obs `confidence_set` loop. As of
v1 Task B6, all five procedure types implement `confidence_set_batch` for
d_theta=1; multivariate fast paths are future work, and d>1 falls through
to the slow loop with the diameter convention.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch

from cdsbi.diagnostics.base import Diagnostic, DiagnosticResult


class SetSize(Diagnostic):
    name = "set_size"

    def __init__(self, theta_0_grid: List[float], alpha_grid: List[float], n_per_theta: int = 200):
        self.theta_0_grid = theta_0_grid
        self.alpha_grid = alpha_grid
        self.n_per_theta = n_per_theta

    def __call__(self, trained, simulator, eval_data=None, x_per_theta=None) -> DiagnosticResult:
        # F7: shared X|θ_0 dict (see Coverage docstring) — slice to n_per_theta.
        rng = np.random.default_rng(0)
        rows = []
        for theta_0 in self.theta_0_grid:
            theta_repr = str(list(map(
                float,
                list(theta_0) if hasattr(theta_0, "__iter__") else [theta_0],
            )))
            if x_per_theta is not None and theta_repr in x_per_theta:
                x = x_per_theta[theta_repr][: self.n_per_theta]
            else:
                x = simulator.sample_x_given_theta(theta_0, self.n_per_theta, rng)
            for alpha in self.alpha_grid:
                has_fast = (
                    hasattr(trained.procedure, "confidence_set_batch")
                    and getattr(trained.procedure, "d_theta", 1) == 1
                )
                if has_fast:
                    # Fast path: single vectorized bisection across batch.
                    # confidence_set_batch is d_theta=1 only in v1; d>1 falls
                    # through to the slow loop with the diameter convention.
                    with torch.no_grad():
                        left, right = trained.procedure.confidence_set_batch(x, alpha)
                    widths = (right - left).detach().cpu().numpy()
                else:
                    widths = np.empty(self.n_per_theta, dtype=np.float64)
                    for i in range(self.n_per_theta):
                        cs = trained.procedure.confidence_set(x[i : i + 1], alpha=alpha)
                        br = cs.boundary_repr
                        if br.ndim == 1:
                            # 1D: (lower, upper)
                            widths[i] = float(br[1] - br[0])
                        elif br.numel() == 0:
                            widths[i] = 0.0  # empty set
                        else:
                            # d > 1: 2 × max distance from boundary centroid —
                            # diameter proxy that matches the d=1 (right - left)
                            # convention. For a perfect ellipsoid, this equals
                            # 2 × semi-major axis (so the column is dimensionally
                            # consistent across d).
                            center = br.mean(dim=0)
                            dist = (br - center).pow(2).sum(dim=-1).sqrt()
                            widths[i] = float(2.0 * dist.max().item())
                theta_vec = np.atleast_1d(np.asarray(theta_0, dtype=np.float64)).reshape(-1)
                row = {f"theta_0_{k}": float(theta_vec[k]) for k in range(theta_vec.shape[0])}
                row.update({
                    "alpha": float(alpha),
                    "mean_width": float(widths.mean()),
                    "median_width": float(np.median(widths)),
                    "p90_width": float(np.quantile(widths, 0.9)),
                    "min_width": float(widths.min()),
                    "max_width": float(widths.max()),
                    "n_eval": int(self.n_per_theta),
                })
                rows.append(row)
        df = pd.DataFrame(rows)
        return DiagnosticResult(
            name=self.name,
            value=df,
            passed=True,
            noise_floor=0.0,
            n_samples=len(self.theta_0_grid) * self.n_per_theta,
            meta={"n_thetas": len(self.theta_0_grid), "n_alphas": len(self.alpha_grid)},
        )
