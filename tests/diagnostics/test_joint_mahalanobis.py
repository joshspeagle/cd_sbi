"""JointMahalanobis: ||r(θ_0; X)||² | θ_0 ~ χ²_d test."""
from __future__ import annotations

import numpy as np
import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.joint_mahalanobis import JointMahalanobis
from cdsbi.methods.base import TrainedModel
from cdsbi.reproducibility.seeding import seed_everything
from cdsbi.simulators.location_gauss_2d_iid import LocationGaussian2D_iid


def _oracle_trained_2d():
    proc = PivotBasedProcedure(
        pivot_fn=lambda th, x: th - x, d_theta=2, theta_range=(-7.0, 7.0),
    )
    return TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0,
    )


def test_joint_mahalanobis_oracle_passes(seed):
    seed_everything(seed)
    sim = LocationGaussian2D_iid()
    trained = _oracle_trained_2d()
    diag = JointMahalanobis(
        theta_0_grid=[[-3.0, -3.0], [0.0, 0.0], [3.0, 3.0]],
        n_per_theta=2000,
    )
    result = diag(trained, sim, eval_data=None)
    df = result.value
    assert "ks" in df.columns
    # Oracle pivot should give KS ≤ noise_floor at all θ_0.
    assert (df["ks"] <= df["noise_floor"]).all(), f"KS: {df['ks'].tolist()}, floor: {df['noise_floor'].iloc[0]}"


def test_joint_mahalanobis_catches_correlation(seed):
    """A miscalibrated pivot whose marginals are N(0,1) but with non-zero
    Pearson correlation should fail JointMahalanobis even though marginal
    PIT would pass."""
    seed_everything(seed)

    def miscal_pivot(theta, x):
        # Componentwise N(0,1) marginally, but r_1, r_2 share a common
        # component → corr ≈ 0.5, joint distribution NOT N(0, I_2).
        eps = theta - x  # would be N(0, I) for the oracle
        r_1 = eps[:, 0]
        r_2 = 0.5 * eps[:, 0] + 0.866 * eps[:, 1]  # corr(r_1, r_2) = 0.5, var(r_2) = 1
        return torch.stack([r_1, r_2], dim=-1)

    proc = PivotBasedProcedure(pivot_fn=miscal_pivot, d_theta=2, theta_range=(-7.0, 7.0))
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0,
    )
    sim = LocationGaussian2D_iid()
    diag = JointMahalanobis(theta_0_grid=[[0.0, 0.0]], n_per_theta=5000)
    result = diag(trained, sim, eval_data=None)
    df = result.value
    # ||r||² should be inflated relative to χ²_2 → KS > floor.
    assert (df["ks"] > df["noise_floor"]).all(), (
        f"miscalibrated pivot passes JointMahalanobis: ks={df['ks'].tolist()}, "
        f"floor={df['noise_floor'].iloc[0]}"
    )


def test_joint_mahalanobis_skips_1d():
    """In 1D the diagnostic is degenerate (||r||² = r² has the squared marginal-PIT distribution);
    must no-op gracefully."""
    from cdsbi.simulators.location_normal_1d import LocationNormal1D
    proc = PivotBasedProcedure(
        pivot_fn=lambda th, x: th - x, d_theta=1, theta_range=(-7.0, 7.0),
    )
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0,
    )
    sim = LocationNormal1D()
    diag = JointMahalanobis(theta_0_grid=[0.0], n_per_theta=500)
    result = diag(trained, sim, eval_data=None)
    assert result.passed  # no-op
    assert "reason" in result.meta
