"""PivotBasedProcedure in d > 1."""
from __future__ import annotations

import math

import torch
from scipy.stats import chi2

from cdsbi.confidence_set.procedures import PivotBasedProcedure


def _oracle_2d(theta, x):
    # Canonical r* for LocationGaussian2D_iid: r = θ - X (per-coordinate).
    return theta - x


def test_confidence_set_returns_2d_boundary(seed):
    proc = PivotBasedProcedure(pivot_fn=_oracle_2d, d_theta=2, theta_range=(-7.0, 7.0))
    x_obs = torch.tensor([[0.5, -0.2]])
    cs = proc.confidence_set(x_obs, alpha=0.9)
    # boundary_repr: (K, d) where K is the number of sampled rays.
    assert cs.boundary_repr.ndim == 2
    assert cs.boundary_repr.shape[1] == 2
    # Boundary points should satisfy ||r||² ≈ χ²_{2, 0.9}
    thresh = float(chi2.ppf(0.9, df=2))
    r_boundary = _oracle_2d(cs.boundary_repr, x_obs.expand(cs.boundary_repr.shape[0], -1))
    sq = r_boundary.pow(2).sum(dim=-1)
    # Tolerate the 1D-bisection's 1e-4 endpoint precision.
    assert (sq - thresh).abs().max().item() < 1e-2


def test_contains_uses_chisq_condition(seed):
    proc = PivotBasedProcedure(pivot_fn=_oracle_2d, d_theta=2, theta_range=(-7.0, 7.0))
    x_obs = torch.tensor([[0.5, -0.2]])
    cs = proc.confidence_set(x_obs, alpha=0.9)
    # ||θ - X||² ≤ χ²_{2, 0.9} ⇒ contains.
    thresh = float(chi2.ppf(0.9, df=2))
    # θ at center (= X_obs) has r = 0, definitely contained.
    assert cs.contains([0.5, -0.2])
    # Far-away θ definitely outside.
    assert not cs.contains([5.0, 5.0])


def test_contains_batch_unchanged_in_d2(seed):
    """contains_batch already handled d > 1 in v0; verify it still works."""
    proc = PivotBasedProcedure(pivot_fn=_oracle_2d, d_theta=2, theta_range=(-7.0, 7.0))
    x_batch = torch.randn(100, 2)
    inside = proc.contains_batch([0.0, 0.0], x_batch, alpha=0.9)
    assert inside.shape == (100,)
    assert inside.dtype == torch.bool
    # By definition, P(||−X||² ≤ χ²_{2, 0.9} | X ~ N(0, I_2)) ≈ 0.9.
    p = inside.float().mean().item()
    assert abs(p - 0.9) < 0.05


def test_contains_accepts_cuda_tensor_theta_when_available():
    """The contains closure must accept a CUDA tensor θ_0 if one is provided.

    Skips when CUDA is unavailable. Catches the np.asarray(cuda_tensor) bug
    flagged by code review.
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")
    proc = PivotBasedProcedure(pivot_fn=_oracle_2d, d_theta=2, theta_range=(-7.0, 7.0))
    x_obs = torch.tensor([[0.5, -0.2]], device="cuda")
    cs = proc.confidence_set(x_obs, alpha=0.9)
    theta_cuda = torch.tensor([0.5, -0.2], device="cuda")
    assert cs.contains(theta_cuda) is True


def test_confidence_set_empty_when_min_outside_threshold():
    """If even argmin ||r||² > χ²_d, the α-set is provably empty."""
    # A pivot with constant large output ⇒ ||r||² always ≥ 100 ⇒ empty at any α < 1.
    # Multiply by (theta * 0) so the output stays connected to the input graph
    # for _find_center's LBFGS step (which calls .backward()).
    def constant_large(theta, x):
        const = torch.full(
            (theta.shape[0], 2), 10.0, dtype=theta.dtype, device=theta.device,
        )
        return const + 0.0 * theta

    proc = PivotBasedProcedure(pivot_fn=constant_large, d_theta=2, theta_range=(-7.0, 7.0))
    x_obs = torch.tensor([[0.0, 0.0]])
    cs = proc.confidence_set(x_obs, alpha=0.9)
    assert cs.boundary_repr.shape == (0, 2)
    assert cs.contains([0.0, 0.0]) is False
    assert cs.contains([3.0, 3.0]) is False
