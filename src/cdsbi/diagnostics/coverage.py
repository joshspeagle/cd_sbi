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

    def __call__(self, trained, simulator, eval_data=None, x_per_theta=None) -> DiagnosticResult:
        # F7: x_per_theta is an optional dict {theta_0_repr: x_tensor} pre-drawn
        # at the dispatcher level so Coverage / SetSize / JointMahalanobis share
        # a single X|θ_0 draw per θ_0 (slicing to each diagnostic's n_per_theta).
        proc = trained.procedure
        # Scalable fast path: delegate to the chunked, statistic-once engine when the
        # procedure exposes pivot / test_statistic / contains_batch (everything except
        # confidence_set-only procedures). The engine is verified bit-identical to the
        # loop below (max|Δ|=0.0000) but memory-bounded (no n×d precompute → no OOM)
        # and statistic-once (no per-α / per-diagnostic recompute) — what makes high-d /
        # autograd (Score-CD) evaluation tractable.
        if any(hasattr(proc, m) for m in ("pivot", "test_statistic", "contains_batch")):
            from cdsbi.diagnostics.engine import evaluate_coverage
            out = evaluate_coverage(proc, simulator, self.theta_0_grid, self.alpha_grid,
                                    n_per_theta=self.n_per_theta, x_per_theta=x_per_theta)
            cov = out["coverage"].copy()
            cov["n_eval"] = int(self.n_per_theta)
            cov["passed"] = (cov["empirical"] - cov["nominal"]).abs() <= 0.02
            cov["tolerance"] = 0.02
            return DiagnosticResult(
                name=self.name, value=cov, passed=bool(cov["passed"].all()),
                noise_floor=0.02, n_samples=len(self.theta_0_grid) * self.n_per_theta,
            )

        # Fallback: per-sample confidence_set construction (set-only procedures).
        rng = np.random.default_rng(0)
        rows = []
        has_fast_path = False
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
