import numpy as np
import torch
from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.coverage import Coverage
from cdsbi.methods.base import TrainedModel
from cdsbi.simulators.location_normal_1d import LocationNormal1D
from cdsbi.reproducibility.seeding import seed_everything


def _oracle_trained():
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: th - x, d_theta=1)
    return TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0
    )


def test_coverage_oracle_within_mc_error(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    trained = _oracle_trained()
    diag = Coverage(theta_0_grid=[-3.0, 0.0, 3.0], alpha_grid=[0.5, 0.68, 0.9, 0.95], n_per_theta=2000)
    result = diag(trained, sim, eval_data=None)
    df = result.value
    df["err"] = (df["empirical"] - df["nominal"]).abs()
    assert (df["err"] < 0.05).all()


def test_coverage_catches_underdispersed(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: 2.0 * (th - x), d_theta=1)
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0
    )
    diag = Coverage(theta_0_grid=[0.0], alpha_grid=[0.9], n_per_theta=2000)
    result = diag(trained, sim, eval_data=None)
    df = result.value
    assert df["empirical"].iloc[0] < 0.85
