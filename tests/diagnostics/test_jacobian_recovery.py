"""Jacobian-recovery diagnostic regression tests."""
from __future__ import annotations

import numpy as np
import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.jacobian_recovery import JacobianRecovery
from cdsbi.methods.base import TrainedModel
from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def _trained_with_linear_pivot(J: torch.Tensor, d_theta: int) -> TrainedModel:
    """Build a TrainedModel whose procedure is a PivotBasedProcedure with
    r(θ, X) = (θ - X) @ J^T (closed-form linear pivot, constant Jacobian J)."""
    proc = PivotBasedProcedure(
        pivot_fn=lambda th, x: (th - x) @ J.T,
        d_theta=d_theta,
        theta_range=(-7.0, 7.0),
    )
    return TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0,
    )


def test_jacobian_recovery_recovers_truth_when_pivot_is_truth():
    """If procedure.pivot is literally r* = L⁻¹(θ - X), the diagnostic must
    report max_residual ≈ 0 (any non-zero is numerical noise from the K-point
    Monte-Carlo aggregation)."""
    sim = LocationGaussian2D_corr()
    L_inv = sim.r_star_jacobian()
    trained = _trained_with_linear_pivot(L_inv, d_theta=2)
    diag = JacobianRecovery(n_points=200)
    rng = np.random.default_rng(0)
    eval_data = sim.sample(500, rng)
    result = diag(trained, sim, eval_data=eval_data)
    assert result.value["max_residual"].iloc[0] < 1e-4, result.value


def test_jacobian_recovery_flags_wrong_pivot():
    """If procedure.pivot uses the wrong constant Jacobian, max_residual must
    be the elementwise distance between the two matrices."""
    sim = LocationGaussian2D_corr()
    wrong = torch.eye(2)
    trained = _trained_with_linear_pivot(wrong, d_theta=2)
    diag = JacobianRecovery(n_points=200)
    rng = np.random.default_rng(0)
    eval_data = sim.sample(500, rng)
    result = diag(trained, sim, eval_data=eval_data)
    L_inv = sim.r_star_jacobian()
    expected_max = float((L_inv - wrong).abs().max())
    assert abs(result.value["max_residual"].iloc[0] - expected_max) < 1e-4


def test_jacobian_recovery_skips_non_pivot_procedure():
    """Non-pivot procedures get NaN + skip reason in meta, mirroring JointMahalanobis."""
    from types import SimpleNamespace
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
    sim = LocationNormal1D()
    trained = _trained_with_linear_pivot(torch.tensor([[1.0]]), d_theta=1)
    diag = JacobianRecovery(n_points=50)
    rng = np.random.default_rng(0)
    eval_data = sim.sample(100, rng)
    result = diag(trained, sim, eval_data=eval_data)
    assert np.isnan(result.value["max_residual"].iloc[0])
    assert "no r_star_jacobian" in result.meta.get("reason", "")
