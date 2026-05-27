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


def test_marginal_pit_per_coordinate_d2(seed):
    """In d=2, marginal PIT should report one KS per coordinate."""
    from cdsbi.simulators.location_gauss_2d_iid import LocationGaussian2D_iid
    from cdsbi.confidence_set.procedures import PivotBasedProcedure
    from cdsbi.methods.base import TrainedModel
    from cdsbi.diagnostics.marginal_pit import MarginalPIT
    from cdsbi.reproducibility.seeding import seed_everything
    import numpy as np
    seed_everything(seed)
    sim = LocationGaussian2D_iid()
    proc = PivotBasedProcedure(
        pivot_fn=lambda th, x: th - x, d_theta=2, theta_range=(-7.0, 7.0),
    )
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0,
    )
    rng = np.random.default_rng(seed)
    # N=10000 (vs the 1D oracle's N=5000): two independent KS tests double
    # the per-coord flake probability at the 99% floor, so we need a tighter
    # noise floor (∝ 1/√N) to keep this test deterministic across seeds.
    theta, x = sim.sample(10000, rng)
    diag = MarginalPIT()
    result = diag(trained, sim, eval_data=(theta, x))
    import pandas as pd
    assert isinstance(result.value, pd.DataFrame)
    assert "coord" in result.value.columns
    assert len(result.value) == 2  # d=2 → 2 rows
    assert (result.value["ks"] <= result.noise_floor).all()
