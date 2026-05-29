"""TriangularDoublyMonotoneFlow + its conditioned monotone-scalar building block."""
from __future__ import annotations

import torch


def test_cond_monotone_scalar_increasing_and_derivative_matches_autograd():
    from cdsbi.flows.triangular_doubly_monotone import _CondMonotoneScalarUMNN
    torch.manual_seed(0)
    f = _CondMonotoneScalarUMNN(context_dim=3, hidden=16, depth=2, z_ref=0.0)
    n = 64
    ctx = torch.randn(n, 3)
    z = torch.linspace(-2, 2, n).unsqueeze(-1)
    val = f(z, ctx)
    assert val.shape == (n, 1)
    # monotonicity is only well-defined at a FIXED context, so check there:
    ctx0 = torch.randn(1, 3).expand(n, 3)
    zz = torch.linspace(-3, 3, n).unsqueeze(-1)
    vv = f(zz, ctx0).squeeze(-1)
    assert torch.all(vv[1:] - vv[:-1] > 0)               # increasing in z at fixed ctx
    # analytic derivative matches autograd
    z_req = torch.tensor([[0.7]], requires_grad=True)
    c1 = torch.randn(1, 3)
    out = f(z_req, c1)
    (grad,) = torch.autograd.grad(out.sum(), z_req)
    assert torch.allclose(grad, f.derivative(z_req, c1), atol=1e-4)


def test_cond_monotone_scalar_empty_context():
    from cdsbi.flows.triangular_doubly_monotone import _CondMonotoneScalarUMNN
    f = _CondMonotoneScalarUMNN(context_dim=0, hidden=8, depth=2, z_ref=0.0)
    z = torch.linspace(-2, 2, 32).unsqueeze(-1)
    v = f(z, None).squeeze(-1)
    assert torch.all(v[1:] - v[:-1] > 0)


def test_flow_advertises_R1_R2_and_shapes():
    from cdsbi.flows.triangular_doubly_monotone import TriangularDoublyMonotoneFlow
    from cdsbi.flows.base import Guarantee
    flow = TriangularDoublyMonotoneFlow(d=2, hidden=16)
    assert flow.monotonicity_guarantees == frozenset({Guarantee.R1, Guarantee.R2})
    theta = torch.randn(32, 2)
    feats = torch.randn(32, 2)
    r, log_det = flow.forward(theta, context=feats)
    assert r.shape == (32, 2) and log_det.shape == (32,)


def test_flow_monotone_in_theta_per_coord():
    from cdsbi.flows.triangular_doubly_monotone import TriangularDoublyMonotoneFlow
    torch.manual_seed(1)
    flow = TriangularDoublyMonotoneFlow(d=2, hidden=16)
    feats = torch.randn(1, 2).expand(40, 2).contiguous()
    th = torch.zeros(40, 2); th[:, 1] = torch.linspace(-3, 3, 40)
    r, _ = flow.forward(th, context=feats)
    assert torch.all(r[1:, 1] - r[:-1, 1] > 0)
    th = torch.zeros(40, 2); th[:, 0] = torch.linspace(-3, 3, 40)
    r, _ = flow.forward(th, context=feats)
    assert torch.all(r[1:, 0] - r[:-1, 0] > 0)


def test_flow_logdet_matches_autograd_feature_jacobian():
    from cdsbi.flows.triangular_doubly_monotone import TriangularDoublyMonotoneFlow
    torch.manual_seed(2)
    flow = TriangularDoublyMonotoneFlow(d=2, hidden=16)
    theta = torch.randn(1, 2)
    feats = torch.randn(1, 2, requires_grad=True)
    r, log_det = flow.forward(theta, context=feats)
    J = torch.zeros(2, 2)
    for i in range(2):
        (gi,) = torch.autograd.grad(r[0, i], feats, retain_graph=True)
        J[i] = gi[0]
    logdet_ref = torch.log(torch.abs(torch.det(J)))
    assert torch.allclose(log_det[0], logdet_ref, atol=1e-3)
    assert abs(float(J[0, 1])) < 1e-5     # triangular: ∂r_0/∂feat_1 ≈ 0


def test_flow_monotone_in_features_per_coord():
    from cdsbi.flows.triangular_doubly_monotone import TriangularDoublyMonotoneFlow
    torch.manual_seed(3)
    flow = TriangularDoublyMonotoneFlow(d=2, hidden=16)
    theta = torch.randn(1, 2).expand(40, 2).contiguous()
    fe = torch.zeros(40, 2); fe[:, 0] = torch.linspace(-3, 3, 40)
    r, _ = flow.forward(theta, context=fe)
    assert torch.all(r[1:, 0] - r[:-1, 0] > 0)
