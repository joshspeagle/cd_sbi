"""SingleIndexMonotoneFlow: per-coord signed monotonicity + closed-form log_det."""
from __future__ import annotations

import torch
from cdsbi.flows.base import Guarantee


def _flow(d=2):
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    # (μ,σ²) signs: increasing in θ (+1,+1), decreasing in feat (-1,-1)
    return SingleIndexMonotoneFlow(d=d, theta_signs=[1.0, 1.0], feat_signs=[-1.0, -1.0], hidden=16)


def test_advertises_R1_R2_and_shapes():
    flow = _flow()
    assert flow.monotonicity_guarantees == frozenset({Guarantee.R1, Guarantee.R2})
    theta = torch.randn(32, 2); feats = torch.randn(32, 2)
    r, ld = flow.forward(theta, context=feats)
    assert r.shape == (32, 2) and ld.shape == (32,)


def test_increasing_in_theta_decreasing_in_feat():
    torch.manual_seed(1)
    flow = _flow()
    feats = torch.randn(1, 2).expand(40, 2).contiguous()
    # s_theta=+1 → increasing in each θ_k
    for k in (0, 1):
        th = torch.zeros(40, 2); th[:, k] = torch.linspace(-3, 3, 40)
        r, _ = flow.forward(th, context=feats)
        assert torch.all(r[1:, k] - r[:-1, k] > 0), f"not increasing in theta_{k}"
    # s_feat=-1 → DECREASING in each feat_k (at fixed theta)
    theta = torch.randn(1, 2).expand(40, 2).contiguous()
    for k in (0, 1):
        fe = torch.zeros(40, 2); fe[:, k] = torch.linspace(-3, 3, 40)
        r, _ = flow.forward(theta, context=fe)
        assert torch.all(r[1:, k] - r[:-1, k] < 0), f"not decreasing in feat_{k}"


def test_logdet_matches_autograd_and_triangular():
    torch.manual_seed(2)
    flow = _flow()
    theta = torch.randn(1, 2); feats = torch.randn(1, 2, requires_grad=True)
    r, log_det = flow.forward(theta, context=feats)
    J = torch.zeros(2, 2)
    for i in range(2):
        (gi,) = torch.autograd.grad(r[0, i], feats, retain_graph=True)
        J[i] = gi[0]
    assert torch.allclose(log_det[0], torch.log(torch.abs(torch.det(J))), atol=1e-3)
    assert abs(float(J[0, 1])) < 1e-5          # triangular: ∂r_0/∂feat_1 = 0
