"""DoublyMonotoneUMNN — §6.1 form 2 flow for §8.4."""
from __future__ import annotations

import torch

from cdsbi.flows.base import Guarantee


def test_doubly_monotone_advertises_R1_R2():
    from cdsbi.flows.doubly_monotone import DoublyMonotoneUMNN
    flow = DoublyMonotoneUMNN(hidden=8, theta_ref=0.3)
    assert Guarantee.R1 in flow.monotonicity_guarantees
    assert Guarantee.R2 in flow.monotonicity_guarantees


def test_doubly_monotone_forward_shape_and_log_det(seed):
    from cdsbi.flows.doubly_monotone import DoublyMonotoneUMNN
    torch.manual_seed(seed)
    flow = DoublyMonotoneUMNN(hidden=8, theta_ref=0.3)
    theta = torch.rand(64, 1) * 2.7 + 0.3  # uniform on [0.3, 3.0]
    T = torch.rand(64, 1) * 10 + 0.5       # positive scalar (sum-of-exponentials proxy)
    r, log_det = flow(theta, context=T)
    assert r.shape == (64, 1)
    assert log_det.shape == (64,)
    # log_det should be the log of ∂r/∂T evaluated at each (θ, T), a strictly
    # positive quantity ⇒ log_det is finite (no -inf).
    assert torch.isfinite(log_det).all()


def test_doubly_monotone_partial_theta_positive(seed):
    """∂r/∂θ = softplus(α(θ) + β_umnn(T)) > 0 for any (θ, T)."""
    from cdsbi.flows.doubly_monotone import DoublyMonotoneUMNN
    torch.manual_seed(seed)
    flow = DoublyMonotoneUMNN(hidden=8, theta_ref=0.3)
    theta = (torch.rand(64, 1) * 2.7 + 0.3).requires_grad_(True)
    T = torch.rand(64, 1) * 10 + 0.5
    r, _ = flow(theta, context=T)
    grad_theta = torch.autograd.grad(r.sum(), theta, create_graph=False)[0]
    assert (grad_theta > 0).all(), grad_theta


def test_doubly_monotone_partial_T_positive(seed):
    """∂r/∂T = b'_umnn(T) + β'_umnn(T) · ∫_{θ_ref}^θ σ(·) dt > 0 by construction."""
    from cdsbi.flows.doubly_monotone import DoublyMonotoneUMNN
    torch.manual_seed(seed)
    flow = DoublyMonotoneUMNN(hidden=8, theta_ref=0.3)
    theta = torch.rand(64, 1) * 2.7 + 0.3
    T = (torch.rand(64, 1) * 10 + 0.5).requires_grad_(True)
    r, _ = flow(theta, context=T)
    grad_T = torch.autograd.grad(r.sum(), T, create_graph=False)[0]
    assert (grad_T > 0).all(), grad_T


def test_doubly_monotone_at_theta_ref_equals_b(seed):
    """At θ = θ_ref, the integral is zero, so r(θ_ref, T) = b_umnn(T) exactly."""
    from cdsbi.flows.doubly_monotone import DoublyMonotoneUMNN
    torch.manual_seed(seed)
    flow = DoublyMonotoneUMNN(hidden=8, theta_ref=0.3)
    T = torch.tensor([[1.0], [2.0], [3.0]])
    theta_ref_tensor = torch.full((3, 1), 0.3)
    r, _ = flow(theta_ref_tensor, context=T)
    # b_umnn(T) is directly accessible
    b_T = flow._b_umnn(T)
    torch.testing.assert_close(r, b_T, atol=1e-5, rtol=0.0)
