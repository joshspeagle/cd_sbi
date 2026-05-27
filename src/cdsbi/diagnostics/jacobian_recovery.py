"""JacobianRecovery: at K random (θ, X) joint samples, compare ∂r/∂θ
(via autograd of the trained pivot) to the simulator's r_star_jacobian.

Validates the §8.3 Knothe–Rosenblatt uniqueness claim: even when pointwise
pivot RMSE is large (because L⁻¹ scales the (θ - X) residual nontrivially),
the average Jacobian should match the unique KR rearrangement to within a
few percent. Only meaningful for PivotBasedProcedure + simulators that
expose r_star_jacobian (currently LocationGaussian2D_corr only; falls
through to a no-op skip otherwise).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.base import Diagnostic, DiagnosticResult


def _skip(name: str, reason: str) -> DiagnosticResult:
    df = pd.DataFrame([{
        "max_residual": float("nan"),
        "norm_residual": float("nan"),
        "passed": True,
        "n_points": 0,
    }])
    return DiagnosticResult(
        name=name, value=df, passed=True, noise_floor=0.0,
        n_samples=0, meta={"reason": reason},
    )


class JacobianRecovery(Diagnostic):
    name = "jacobian_recovery"

    def __init__(self, n_points: int = 200, max_residual_tol: float = 0.05) -> None:
        self.n_points = n_points
        self.max_residual_tol = max_residual_tol

    def __call__(self, trained, simulator, eval_data=None, x_per_theta=None) -> DiagnosticResult:
        # Structural check: any procedure exposing a callable `pivot(θ, X)` is
        # eligible (matches the duck-typed `_LinearPivotProcedure` test fixture
        # and stays forward-compatible if a non-`PivotBasedProcedure` pivot
        # class is added later).
        procedure = trained.procedure
        if not (isinstance(procedure, PivotBasedProcedure) or callable(getattr(procedure, "pivot", None))):
            return _skip(self.name, "not a pivot-based procedure")
        if not hasattr(simulator, "r_star_jacobian"):
            return _skip(self.name, "simulator has no r_star_jacobian (no truth target)")

        J_true = simulator.r_star_jacobian()
        d = int(simulator.d_theta)
        assert J_true.shape == (d, d), (
            f"r_star_jacobian returned shape {J_true.shape}, expected ({d}, {d})"
        )

        # Sample n_points joint (θ, X) from the eval split if it covers enough
        # points; otherwise draw fresh ones from the simulator.
        if eval_data is not None:
            unpacked = eval_data if not isinstance(eval_data, tuple) or len(eval_data) == 2 else eval_data[:2]
            theta_eval, x_eval = unpacked
            if theta_eval.shape[0] >= self.n_points:
                idx = torch.randperm(theta_eval.shape[0])[: self.n_points]
                theta_pts = theta_eval[idx].clone()
                x_pts = x_eval[idx].clone()
            else:
                rng = np.random.default_rng(0)
                theta_pts, x_pts = simulator.sample(self.n_points, rng)
        else:
            rng = np.random.default_rng(0)
            theta_pts, x_pts = simulator.sample(self.n_points, rng)

        # Per-point Jacobian via autograd. We avoid jacfwd/jacrev to stay
        # compatible with the existing flow forward (which doesn't promise to
        # be vmap-traceable); a Python loop is fine at n_points=200.
        # x_k is detached: the Jacobian is ∂r/∂θ holding (X, flow weights)
        # fixed, so we explicitly cut grad propagation through X to avoid
        # tracing through the flow parameters during the inner grad call.
        J_emp = torch.empty(self.n_points, d, d)
        for k in range(self.n_points):
            theta_k = theta_pts[k : k + 1].clone().detach().requires_grad_(True)
            x_k = x_pts[k : k + 1].detach()
            r_k = trained.procedure.pivot(theta_k, x_k)  # shape (1, d)
            for i in range(d):
                grad_i = torch.autograd.grad(
                    r_k[0, i], theta_k, retain_graph=(i < d - 1),
                )[0]  # shape (1, d)
                J_emp[k, i, :] = grad_i[0].detach()

        J_emp_mean = J_emp.mean(dim=0)  # (d, d)
        diff = (J_emp_mean - J_true)
        max_residual = float(diff.abs().max())
        norm_residual = float(diff.norm() / J_true.norm())
        passed = max_residual <= self.max_residual_tol

        df = pd.DataFrame([{
            "max_residual": max_residual,
            "norm_residual": norm_residual,
            "passed": passed,
            "n_points": int(self.n_points),
        }])
        return DiagnosticResult(
            name=self.name, value=df, passed=passed, noise_floor=self.max_residual_tol,
            n_samples=int(self.n_points),
            meta={"d": d, "tol": self.max_residual_tol},
        )
