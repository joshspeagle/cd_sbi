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
