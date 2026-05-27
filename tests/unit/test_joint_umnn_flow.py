"""JointUMNNFlow — the R1-only ablation flow for §8.4."""
from __future__ import annotations

import torch

from cdsbi.flows.base import Guarantee


def test_joint_umnn_advertises_R1_only():
    from cdsbi.flows.joint_umnn import JointUMNNFlow
    flow = JointUMNNFlow(hidden=8)
    assert Guarantee.R1 in flow.monotonicity_guarantees
    assert Guarantee.R2 not in flow.monotonicity_guarantees, (
        "ablation flow must NOT advertise R2 — it's the whole point of the ablation"
    )


def test_joint_umnn_forward_shape(seed):
    from cdsbi.flows.joint_umnn import JointUMNNFlow
    torch.manual_seed(seed)
    flow = JointUMNNFlow(hidden=8)
    theta = torch.rand(32, 1) * 2.7 + 0.3
    T = torch.rand(32, 1) * 10 + 0.5
    r, log_det = flow(theta, context=T)
    assert r.shape == (32, 1)
    assert log_det.shape == (32,)
    assert torch.isfinite(log_det).all()


def test_joint_umnn_partial_theta_positive(seed):
    """∂r/∂θ is the softplus integrand at θ → strictly positive."""
    from cdsbi.flows.joint_umnn import JointUMNNFlow
    torch.manual_seed(seed)
    flow = JointUMNNFlow(hidden=8)
    theta = (torch.rand(32, 1) * 2.7 + 0.3).requires_grad_(True)
    T = torch.rand(32, 1) * 10 + 0.5
    r, _ = flow(theta, context=T)
    grad_theta = torch.autograd.grad(r.sum(), theta, create_graph=False)[0]
    assert (grad_theta > 0).all(), grad_theta


def test_joint_umnn_log_det_uses_autograd_through_T(seed):
    """The returned log_det should equal log|∂r/∂T| computed via autograd."""
    from cdsbi.flows.joint_umnn import JointUMNNFlow
    torch.manual_seed(seed)
    flow = JointUMNNFlow(hidden=8)
    theta = torch.rand(8, 1) * 2.7 + 0.3
    T = (torch.rand(8, 1) * 10 + 0.5).requires_grad_(True)
    r, log_det = flow(theta, context=T)
    # Independent autograd of r w.r.t. T
    grad_T = torch.autograd.grad(r.sum(), T, create_graph=False)[0]
    expected_log_det = torch.log(grad_T.abs().clamp_min(1e-12)).squeeze(-1)
    torch.testing.assert_close(log_det, expected_log_det, atol=1e-4, rtol=1e-4)
