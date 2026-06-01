# tests/unit/test_non_monotone_pivot.py
import torch
from cdsbi.flows.base import Guarantee
from cdsbi.flows.non_monotone_pivot import NonMonotonePivotFlow


def test_guarantees_r2_only():
    flow = NonMonotonePivotFlow(d=1, feat_signs=(1.0,))
    assert flow.monotonicity_guarantees == frozenset({Guarantee.R2})


def test_forward_shapes_and_feature_monotone():
    flow = NonMonotonePivotFlow(d=1, feat_signs=(1.0,))
    theta = torch.randn(32, 1)
    feats = torch.randn(32, 1, requires_grad=True)
    r, log_det = flow.forward(theta, context=feats)
    assert r.shape == (32, 1) and log_det.shape == (32,)
    # ∂r/∂feat must be single-signed (C1): check positivity (feat_signs=+1)
    g = torch.autograd.grad(r.sum(), feats)[0]
    assert (g > 0).all(), "feature channel must be monotone increasing (C1)"


def test_can_be_non_monotone_in_theta():
    """Unlike SingleIndexMonotoneFlow, r(θ) can be non-monotone (it must, to give
    disconnected sets). Fit nothing — just check the architecture permits ∂r/∂θ<0
    somewhere after a random init with a curved θ-net."""
    torch.manual_seed(0)
    flow = NonMonotonePivotFlow(d=1, feat_signs=(1.0,), theta_hidden=32)
    feats = torch.zeros(200, 1)
    theta = torch.linspace(-3, 3, 200).unsqueeze(1).requires_grad_(True)
    r, _ = flow.forward(theta, context=feats)
    g = torch.autograd.grad(r.sum(), theta)[0]
    assert (g < 0).any() and (g > 0).any(), "θ-channel should be unconstrained in sign"
