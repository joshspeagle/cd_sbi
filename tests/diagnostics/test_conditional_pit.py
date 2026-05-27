import numpy as np
import torch
from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.conditional_pit import ConditionalPIT
from cdsbi.diagnostics.ks_floor import ks_noise_floor
from cdsbi.methods.base import TrainedModel
from cdsbi.simulators.location_normal_1d import LocationNormal1D
from cdsbi.reproducibility.seeding import seed_everything


def test_conditional_pit_per_bin_floor_calculation():
    diag = ConditionalPIT(n_bins=5)
    assert abs(diag._per_bin_floor(N=5000) - ks_noise_floor(5000, 5)) < 1e-12


def test_conditional_pit_oracle_passes(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: th - x, d_theta=1)
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0
    )
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(5000, rng)
    result = ConditionalPIT(n_bins=5)(trained, sim, eval_data=(theta, x))
    assert result.passed


def test_conditional_pit_per_coordinate_d2(seed):
    from cdsbi.simulators.location_gauss_2d_iid import LocationGaussian2D_iid
    from cdsbi.confidence_set.procedures import PivotBasedProcedure
    from cdsbi.methods.base import TrainedModel
    from cdsbi.diagnostics.conditional_pit import ConditionalPIT
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
    theta, x = sim.sample(5000, rng)
    diag = ConditionalPIT(n_bins=5)
    result = diag(trained, sim, eval_data=(theta, x))
    df = result.value
    assert "coord" in df.columns
    assert df["coord"].nunique() == 2  # one set of bins per θ-coordinate
