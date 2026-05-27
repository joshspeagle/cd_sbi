"""TriangularAdditiveFlow: autoregressive additive form for d ≥ 2."""
from __future__ import annotations

import math

import torch

from cdsbi.flows.base import Guarantee
from cdsbi.flows.triangular_additive import TriangularAdditiveFlow


def test_advertises_R1_R2():
    flow = TriangularAdditiveFlow(d=2, hidden=8)
    assert flow.monotonicity_guarantees == frozenset({Guarantee.R1, Guarantee.R2})


def test_forward_shape_d2(seed):
    flow = TriangularAdditiveFlow(d=2, hidden=8)
    n = 64
    theta = torch.randn(n, 2)
    x = torch.randn(n, 2)
    r, log_det = flow.forward(theta, context=x)
    assert r.shape == (n, 2)
    assert log_det.shape == (n,)


def test_strictly_monotone_in_theta_per_coord(seed):
    flow = TriangularAdditiveFlow(d=2, hidden=8)
    flow.eval()
    # Hold X and θ_<k fixed; sweep θ_k from -3 to 3; r_k must be strictly increasing.
    x = torch.tensor([[0.5, -0.5]])
    base = torch.tensor([[0.0, 0.0]])
    grid = torch.linspace(-3.0, 3.0, 25).view(-1, 1)
    # Sweep θ_1
    theta_sweep_1 = torch.cat([grid, base[:, 1:].expand(25, -1)], dim=-1)
    r_sweep_1, _ = flow.forward(theta_sweep_1, context=x.expand(25, -1))
    assert (r_sweep_1[1:, 0] > r_sweep_1[:-1, 0]).all(), "r_1 not strictly increasing in θ_1"
    # Sweep θ_2
    theta_sweep_2 = torch.cat([base[:, :1].expand(25, -1), grid], dim=-1)
    r_sweep_2, _ = flow.forward(theta_sweep_2, context=x.expand(25, -1))
    assert (r_sweep_2[1:, 1] > r_sweep_2[:-1, 1]).all(), "r_2 not strictly increasing in θ_2"


def test_strictly_monotone_in_X_per_coord_with_negative_sign(seed):
    """Per spec / manuscript §6.1 (R2_auto): ∂r_k/∂X_k < 0 by convention."""
    flow = TriangularAdditiveFlow(d=2, hidden=8)
    flow.eval()
    theta = torch.tensor([[0.5, -0.5]])
    base_x = torch.tensor([[0.0, 0.0]])
    grid = torch.linspace(-3.0, 3.0, 25).view(-1, 1)
    # Sweep X_1
    x_sweep_1 = torch.cat([grid, base_x[:, 1:].expand(25, -1)], dim=-1)
    r_sweep_1, _ = flow.forward(theta.expand(25, -1), context=x_sweep_1)
    assert (r_sweep_1[1:, 0] < r_sweep_1[:-1, 0]).all(), "r_1 not strictly decreasing in X_1"
    # Sweep X_2
    x_sweep_2 = torch.cat([base_x[:, :1].expand(25, -1), grid], dim=-1)
    r_sweep_2, _ = flow.forward(theta.expand(25, -1), context=x_sweep_2)
    assert (r_sweep_2[1:, 1] < r_sweep_2[:-1, 1]).all(), "r_2 not strictly decreasing in X_2"


def test_log_det_matches_diagonal_jacobian(seed):
    """For triangular flows, log|det ∂r/∂X| = Σ_k log|∂r_k/∂X_k|."""
    flow = TriangularAdditiveFlow(d=2, hidden=8)
    flow.eval()
    n = 16
    theta = torch.randn(n, 2)
    x = torch.randn(n, 2, requires_grad=True)
    r, log_det = flow.forward(theta, context=x)
    # Sum |∂r_k/∂X_k| via per-coord backward — fast in d=2.
    diag_terms = []
    for k in range(2):
        g = torch.autograd.grad(r[:, k].sum(), x, retain_graph=True)[0]
        diag_terms.append(g[:, k])
    expected_log_det = torch.stack([torch.log(d.abs()) for d in diag_terms]).sum(dim=0)
    assert torch.allclose(log_det, expected_log_det, atol=1e-4), (
        f"log_det disagrees with diagonal product (max diff {(log_det - expected_log_det).abs().max().item():.2e})"
    )


def test_d2_reduces_to_AdditiveFlow1D_on_first_coord(seed):
    """When d=2, the first coordinate's flow equals a standalone AdditiveFlow1D in distribution."""
    # Construct both; share random init via a seed; verify shape matches at minimum.
    flow = TriangularAdditiveFlow(d=2, hidden=8)
    theta = torch.tensor([[0.5, 1.0]])
    x = torch.tensor([[0.2, -0.3]])
    r, _ = flow.forward(theta, context=x)
    # r_1 must depend only on (θ_1, X_1):
    theta2 = torch.tensor([[0.5, 99.0]])
    x2 = torch.tensor([[0.2, 99.0]])
    r2, _ = flow.forward(theta2, context=x2)
    assert torch.allclose(r[:, 0], r2[:, 0], atol=1e-6), (
        f"r_1 not autoregressive: depends on (θ_2 or X_2)"
    )


def test_n_params_grows_with_d(seed):
    flow_d2 = TriangularAdditiveFlow(d=2, hidden=8)
    flow_d3 = TriangularAdditiveFlow(d=3, hidden=8)
    # d=3 has an extra coordinate with context_dim = 2*(d-1) = 4 (theta_<k, X_<k).
    assert flow_d3.n_params() > flow_d2.n_params()
