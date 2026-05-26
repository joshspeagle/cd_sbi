import numpy as np
import torch
from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.marginal_pit import MarginalPIT
from cdsbi.methods.base import TrainedModel
from cdsbi.simulators.location_normal_1d import LocationNormal1D
from cdsbi.reproducibility.seeding import seed_everything


def test_marginal_pit_oracle_passes(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: th - x, d_theta=1)
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0
    )
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(5000, rng)
    result = MarginalPIT()(trained, sim, eval_data=(theta, x))
    assert result.passed
    assert result.value < result.noise_floor


def test_marginal_pit_catches_underdispersed(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: 0.5 * (th - x), d_theta=1)
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0
    )
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(5000, rng)
    result = MarginalPIT()(trained, sim, eval_data=(theta, x))
    assert not result.passed
