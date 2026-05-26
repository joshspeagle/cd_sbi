import torch
from cdsbi.flows.additive import AdditiveFlow1D
from cdsbi.flows.base import Guarantee


def test_additive_flow_guarantees_r1_r2():
    flow = AdditiveFlow1D(hidden=16)
    assert flow.monotonicity_guarantees == frozenset({Guarantee.R1, Guarantee.R2})


def test_additive_flow_forward_shapes(seed):
    torch.manual_seed(seed)
    flow = AdditiveFlow1D(hidden=16)
    theta = torch.linspace(-3, 3, 10).unsqueeze(-1)
    context = torch.randn(10, 1)  # this conditioner-encoded X
    r, log_det = flow.forward(theta, context)
    assert r.shape == (10, 1)
    assert log_det.shape == (10,)


def test_additive_flow_monotone_in_theta(seed):
    torch.manual_seed(seed)
    flow = AdditiveFlow1D(hidden=16)
    theta = torch.linspace(-3, 3, 100).unsqueeze(-1)
    context = torch.zeros(100, 1)
    r, _ = flow.forward(theta, context)
    diffs = r[1:] - r[:-1]
    assert (diffs > 0).all()


def test_additive_flow_monotone_decreasing_in_x(seed):
    torch.manual_seed(seed)
    flow = AdditiveFlow1D(hidden=16)
    theta = torch.zeros(100, 1)
    context = torch.linspace(-3, 3, 100).unsqueeze(-1)
    r, _ = flow.forward(theta, context)
    diffs = r[1:] - r[:-1]
    assert (diffs < 0).all()  # subtraction of monotone-in-X term
