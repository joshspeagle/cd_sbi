import torch
from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.pivot_rmse import PivotRMSE
from cdsbi.methods.base import TrainedModel
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def _make_oracle_trained(sim):
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: th - x, d_theta=1)
    return TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0
    )


def test_pivot_rmse_oracle_zero():
    sim = LocationNormal1D()
    trained = _make_oracle_trained(sim)
    theta = torch.linspace(-5, 5, 100).unsqueeze(-1)
    x = theta + 0.1 * torch.randn_like(theta)
    result = PivotRMSE()(trained, sim, eval_data=(theta, x))
    assert result.value < 1e-6


def test_pivot_rmse_nonzero_on_perturbed_pivot():
    sim = LocationNormal1D()
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: 0.5 * (th - x), d_theta=1)
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0
    )
    theta = torch.linspace(-5, 5, 100).unsqueeze(-1)
    x = torch.zeros_like(theta)  # Fixed x; r_hat = 0.5*θ vs r* = θ, so RMSE = 0.5*|θ|
    result = PivotRMSE()(trained, sim, eval_data=(theta, x))
    assert result.value > 0.1
