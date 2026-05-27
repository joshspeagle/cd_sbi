"""JointUMNN1DFlow — 1D analog of JointUMNNFlow for the synthetic mechanism test."""
from __future__ import annotations

import torch

from cdsbi.flows.base import Guarantee


def test_joint_umnn_1d_advertises_R1_only():
    from cdsbi.flows.joint_umnn_1d import JointUMNN1DFlow
    flow = JointUMNN1DFlow(hidden=8)
    assert Guarantee.R1 in flow.monotonicity_guarantees
    assert Guarantee.R2 not in flow.monotonicity_guarantees


def test_joint_umnn_1d_forward_shape(seed):
    from cdsbi.flows.joint_umnn_1d import JointUMNN1DFlow
    torch.manual_seed(seed)
    flow = JointUMNN1DFlow(hidden=8)
    theta = torch.rand(32, 1) * 5.0 - 2.5  # uniform on [-2.5, 2.5]
    x = torch.rand(32, 1) * 4.0 - 2.0
    r, log_det = flow(theta, context=x)
    assert r.shape == (32, 1)
    assert log_det.shape == (32,)
    assert torch.isfinite(log_det).all()


def test_joint_umnn_1d_partial_theta_positive(seed):
    """∂r/∂θ is the softplus integrand at θ → strictly positive."""
    from cdsbi.flows.joint_umnn_1d import JointUMNN1DFlow
    torch.manual_seed(seed)
    flow = JointUMNN1DFlow(hidden=8)
    theta = (torch.rand(32, 1) * 5.0 - 2.5).requires_grad_(True)
    x = torch.rand(32, 1) * 4.0 - 2.0
    r, _ = flow(theta, context=x)
    grad_theta = torch.autograd.grad(r.sum(), theta, create_graph=False)[0]
    assert (grad_theta > 0).all(), grad_theta
