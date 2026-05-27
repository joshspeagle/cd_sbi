"""Jacobian-recovery diagnostic regression tests."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import pandas as pd


class _LinearPivotProcedure:
    """Pivot procedure that implements r(θ, X) = J_const (θ - X) for a fixed J_const."""
    def __init__(self, J_const: torch.Tensor):
        self._J = J_const

    def pivot(self, theta, x):
        return (theta - x) @ self._J.T

    @property
    def d_theta(self):
        return int(self._J.shape[1])


def test_jacobian_recovery_recovers_truth_when_pivot_is_truth():
    """If procedure.pivot is literally r* = L⁻¹(θ - X), the diagnostic must
    report max_residual ≈ 0 (any non-zero is numerical noise from the K-point
    Monte-Carlo aggregation)."""
    from cdsbi.diagnostics.jacobian_recovery import JacobianRecovery
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    sim = LocationGaussian2D_corr()
    L_inv = sim.r_star_jacobian()
    trained = SimpleNamespace(procedure=_LinearPivotProcedure(L_inv))
    diag = JacobianRecovery(n_points=200)
    rng = np.random.default_rng(0)
    eval_data = sim.sample(500, rng)
    result = diag(trained, sim, eval_data=eval_data)
    # Constant-Jacobian pivot ⇒ empirical mean Jacobian should equal truth exactly
    # (modulo float32 autograd noise; tolerance well below the diagnostic floor).
    assert result.value["max_residual"].iloc[0] < 1e-4, result.value


def test_jacobian_recovery_flags_wrong_pivot():
    """If procedure.pivot uses the wrong constant Jacobian, max_residual must
    be the elementwise distance between the two matrices."""
    from cdsbi.diagnostics.jacobian_recovery import JacobianRecovery
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    sim = LocationGaussian2D_corr()
    # Use the iid identity as the (wrong) trained pivot.
    wrong = torch.eye(2)
    trained = SimpleNamespace(procedure=_LinearPivotProcedure(wrong))
    diag = JacobianRecovery(n_points=200)
    rng = np.random.default_rng(0)
    eval_data = sim.sample(500, rng)
    result = diag(trained, sim, eval_data=eval_data)
    L_inv = sim.r_star_jacobian()
    expected_max = float((L_inv - wrong).abs().max())
    assert abs(result.value["max_residual"].iloc[0] - expected_max) < 1e-4


def test_jacobian_recovery_skips_non_pivot_procedure():
    """Non-pivot procedures get NaN + skip reason in meta, mirroring JointMahalanobis."""
    from cdsbi.diagnostics.jacobian_recovery import JacobianRecovery
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    sim = LocationGaussian2D_corr()
    trained = SimpleNamespace(procedure=object())  # not PivotBasedProcedure
    diag = JacobianRecovery(n_points=200)
    rng = np.random.default_rng(0)
    eval_data = sim.sample(100, rng)
    result = diag(trained, sim, eval_data=eval_data)
    assert np.isnan(result.value["max_residual"].iloc[0])
    assert "not a pivot-based procedure" in result.meta.get("reason", "")


def test_jacobian_recovery_skips_simulator_without_truth_jacobian():
    """If the simulator does not expose r_star_jacobian(), the diagnostic
    no-ops the same way (the truth Jacobian is needed to score the empirical one).
    Forward-compatible: lets the diagnostic be wired by default in the run.py
    battery without breaking §8.1 / §8.2 runs."""
    from cdsbi.diagnostics.jacobian_recovery import JacobianRecovery
    from cdsbi.simulators.location_normal_1d import LocationNormal1D
    sim = LocationNormal1D()
    trained = SimpleNamespace(procedure=_LinearPivotProcedure(torch.tensor([[1.0]])))
    diag = JacobianRecovery(n_points=50)
    rng = np.random.default_rng(0)
    eval_data = sim.sample(100, rng)
    result = diag(trained, sim, eval_data=eval_data)
    assert np.isnan(result.value["max_residual"].iloc[0])
    assert "no r_star_jacobian" in result.meta.get("reason", "")
