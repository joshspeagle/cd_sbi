import math
import pytest
import torch
from cdsbi.flows.additive import AdditiveFlow1D
from cdsbi.flows.base import Guarantee
from cdsbi.losses.base import MonotonicityMismatchError
from cdsbi.losses.nfmle import NFMLELoss


def test_required_guarantees():
    loss = NFMLELoss()
    assert loss.required_guarantees == frozenset({Guarantee.R1, Guarantee.R2})


def test_population_lower_bound_when_simulator_has_entropy():
    from cdsbi.simulators.location_normal_1d import LocationNormal1D
    loss = NFMLELoss()
    sim = LocationNormal1D()
    expected = 0.5 * math.log(2 * math.pi * math.e)
    assert abs(loss.population_lower_bound(sim) - expected) < 1e-12


def test_population_lower_bound_none_when_simulator_lacks_entropy():
    class DummySim:
        def entropy_lower_bound(self): return None
    loss = NFMLELoss()
    assert loss.population_lower_bound(DummySim()) is None


def test_check_guarantees_passes_on_r1_r2_flow():
    loss = NFMLELoss()
    flow = AdditiveFlow1D(hidden=8)
    loss.check_guarantees(flow)  # should not raise


def test_check_guarantees_raises_on_missing_r2():
    class FlowMissingR2:
        monotonicity_guarantees = frozenset({Guarantee.R1})
    loss = NFMLELoss()
    with pytest.raises(MonotonicityMismatchError):
        loss.check_guarantees(FlowMissingR2())


def test_nfmle_value_on_oracle_pivot(seed):
    """NF-MLE loss for r* = θ − X on N(θ, 1) data ≈ ½ log(2π) + ½ = entropy of N(0,1)."""
    torch.manual_seed(seed)
    n = 5000
    theta = torch.empty(n, 1).uniform_(-7, 7)
    x = theta + torch.randn(n, 1)
    r = theta - x
    log_det = torch.zeros(n)  # |∂r/∂X| = 1 ⇒ log = 0
    loss_val = NFMLELoss()(r=r, log_det_jac_input=log_det)
    expected = 0.5 + 0.5 * math.log(2 * math.pi)
    assert abs(loss_val.item() - expected) < 0.02
