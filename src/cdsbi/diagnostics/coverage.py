"""Coverage: empirical coverage of C_α(X_obs) at multiple (θ_0, α) — cross-method axis."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch

from cdsbi.diagnostics.base import Diagnostic, DiagnosticResult


class Coverage(Diagnostic):
    name = "coverage"

    def __init__(self, theta_0_grid: List[float], alpha_grid: List[float], n_per_theta: int = 1000):
        self.theta_0_grid = theta_0_grid
        self.alpha_grid = alpha_grid
        self.n_per_theta = n_per_theta

    def __call__(self, trained, simulator, eval_data=None) -> DiagnosticResult:
        rng = np.random.default_rng(0)
        rows = []
        has_fast_path = hasattr(trained.procedure, "contains_batch")
        for theta_0 in self.theta_0_grid:
            x = simulator.sample_x_given_theta(theta_0, self.n_per_theta, rng)
            for alpha in self.alpha_grid:
                if has_fast_path:
                    # Single batched forward — any procedure that exposes
                    # contains_batch picks up this fast path automatically.
                    with torch.no_grad():
                        inside_t = trained.procedure.contains_batch(theta_0, x, alpha)
                    empirical = float(inside_t.float().mean().item())
                else:
                    # Fallback: per-sample confidence_set construction.
                    inside = 0
                    for i in range(self.n_per_theta):
                        cs = trained.procedure.confidence_set(x[i : i + 1], alpha=alpha)
                        if cs.contains(theta_0):
                            inside += 1
                    empirical = inside / self.n_per_theta
                theta_vec = np.atleast_1d(np.asarray(theta_0, dtype=np.float64)).reshape(-1)
                row = {f"theta_0_{k}": float(theta_vec[k]) for k in range(theta_vec.shape[0])}
                row.update({
                    "alpha": float(alpha),
                    "nominal": float(alpha),
                    "empirical": float(empirical),
                    "n_eval": int(self.n_per_theta),
                    "passed": abs(empirical - alpha) <= 0.02,
                    "tolerance": 0.02,
                })
                rows.append(row)
        df = pd.DataFrame(rows)
        passed = bool(df["passed"].all())
        return DiagnosticResult(
            name=self.name, value=df, passed=passed, noise_floor=0.02,
            n_samples=len(self.theta_0_grid) * self.n_per_theta,
        )
