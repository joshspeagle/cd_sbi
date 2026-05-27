"""Verify the diagnostic dispatcher sharing optimizations:
- F6: PIT diagnostics accept precomputed pivot tensor via 3-tuple eval_data.
- F7: Coverage/SetSize/JointMahalanobis accept shared x_per_theta dict and slice it.
"""
from __future__ import annotations

import numpy as np
import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.conditional_pit import ConditionalPIT
from cdsbi.diagnostics.coverage import Coverage
from cdsbi.diagnostics.marginal_pit import MarginalPIT
from cdsbi.diagnostics.pivot_rmse import PivotRMSE
from cdsbi.diagnostics.set_size import SetSize
from cdsbi.methods.base import TrainedModel
from cdsbi.reproducibility.seeding import seed_everything
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def _oracle_1d():
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: th - x, d_theta=1)
    return TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0,
    )


def test_pivot_rmse_accepts_3tuple_eval_data(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(1000, rng)
    trained = _oracle_1d()
    r = trained.procedure.pivot(theta, x).detach()

    # 2-tuple form (legacy): computes r internally.
    r_legacy = PivotRMSE()(trained, sim, eval_data=(theta, x)).value
    # 3-tuple form (new): uses passed-in r.
    r_new = PivotRMSE()(trained, sim, eval_data=(theta, x, r)).value
    assert abs(r_legacy - r_new) < 1e-8


def test_marginal_pit_accepts_3tuple(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(1000, rng)
    trained = _oracle_1d()
    r = trained.procedure.pivot(theta, x).detach()
    r_legacy = MarginalPIT()(trained, sim, eval_data=(theta, x)).value
    r_new = MarginalPIT()(trained, sim, eval_data=(theta, x, r)).value
    assert abs(float(r_legacy) - float(r_new)) < 1e-8


def test_conditional_pit_accepts_3tuple(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(1000, rng)
    trained = _oracle_1d()
    r = trained.procedure.pivot(theta, x).detach()
    diag = ConditionalPIT(n_bins=5)
    df_legacy = diag(trained, sim, eval_data=(theta, x)).value
    df_new = diag(trained, sim, eval_data=(theta, x, r)).value
    # Same theta, same x, same r → identical per-bin KS stats.
    assert np.allclose(
        df_legacy["ks"].to_numpy(), df_new["ks"].to_numpy(), atol=1e-8,
    )


def test_coverage_accepts_x_per_theta(seed):
    """Coverage should yield identical empirical coverage when given a
    pre-drawn x_per_theta vs drawing internally with the same seed."""
    seed_everything(seed)
    sim = LocationNormal1D()
    trained = _oracle_1d()

    # Draw the shared X-per-θ_0 dict the way the dispatcher will:
    shared_rng = np.random.default_rng(0)
    grid = [-1.0, 0.0, 1.0]
    x_dict = {
        str([float(t)]): sim.sample_x_given_theta(t, 1000, shared_rng) for t in grid
    }

    diag = Coverage(theta_0_grid=grid, alpha_grid=[0.5, 0.9], n_per_theta=500)
    result_shared = diag(trained, sim, eval_data=None, x_per_theta=x_dict).value
    # Note: the legacy path uses its own rng. Values won't match exactly,
    # but should agree to ≈2× MC SE at n=500.
    result_legacy = diag(trained, sim, eval_data=None).value
    deltas = (result_shared["empirical"].to_numpy() - result_legacy["empirical"].to_numpy())
    assert abs(deltas).max() < 0.05, f"deltas {deltas.tolist()} too large"


def test_set_size_accepts_x_per_theta(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    trained = _oracle_1d()
    shared_rng = np.random.default_rng(0)
    grid = [0.0, 1.0]
    x_dict = {
        str([float(t)]): sim.sample_x_given_theta(t, 200, shared_rng) for t in grid
    }
    diag = SetSize(theta_0_grid=grid, alpha_grid=[0.9], n_per_theta=100)
    result = diag(trained, sim, eval_data=None, x_per_theta=x_dict).value
    assert "mean_width" in result.columns
    # mean_width at α=0.9 for oracle r* = θ - X should be 2√χ²_{1, 0.9} ≈ 3.29.
    assert abs(float(result["mean_width"].iloc[0]) - 3.29) < 0.05
