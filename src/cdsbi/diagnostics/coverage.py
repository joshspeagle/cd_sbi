"""Coverage: empirical coverage of C_α(X_obs) at multiple (θ_0, α) — cross-method axis."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch

from cdsbi.diagnostics.base import Diagnostic, DiagnosticResult
from cdsbi.reproducibility.seeding import seed_everything


class Coverage(Diagnostic):
    name = "coverage"

    def __init__(self, theta_0_grid: List[float], alpha_grid: List[float], n_per_theta: int = 1000):
        self.theta_0_grid = theta_0_grid
        self.alpha_grid = alpha_grid
        self.n_per_theta = n_per_theta

    def __call__(self, trained, simulator, eval_data=None) -> DiagnosticResult:
        rng = np.random.default_rng(0)
        rows = []
        for theta_0 in self.theta_0_grid:
            # Draw n_per_theta X | θ_0 (fix θ; vary X) for LocationNormal1D-style models.
            # NOTE: this hardcodes X = θ + N(0, 1). For v1+ simulators, the Simulator
            # protocol should add a sample_x_given_theta hook.
            theta_t = torch.full((self.n_per_theta, 1), theta_0, dtype=torch.float32)
            eps = rng.standard_normal(size=(self.n_per_theta, 1))
            x = theta_t + torch.from_numpy(eps).float()
            for alpha in self.alpha_grid:
                inside = 0
                for i in range(self.n_per_theta):
                    cs = trained.procedure.confidence_set(x[i : i + 1], alpha=alpha)
                    if cs.contains(theta_0):
                        inside += 1
                empirical = inside / self.n_per_theta
                rows.append({
                    "theta_0_0": float(theta_0),
                    "alpha": float(alpha),
                    "nominal": float(alpha),
                    "empirical": float(empirical),
                    "n_eval": int(self.n_per_theta),
                    "passed": abs(empirical - alpha) <= 0.02,
                    "tolerance": 0.02,
                })
        df = pd.DataFrame(rows)
        passed = bool(df["passed"].all())
        return DiagnosticResult(
            name=self.name, value=df, passed=passed, noise_floor=0.02,
            n_samples=len(self.theta_0_grid) * self.n_per_theta,
        )
