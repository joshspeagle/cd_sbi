"""LikelihoodBasedProcedure.contains_batch agreement + timing.

Same agreement contract as PivotBased/CriticalValue: the fast path matches
the per-sample slow path's `confidence_set().contains()` for any interior
query point. Boundary points may differ by ~grid resolution / 1e-4 bisect
tol — tests avoid the boundary.
"""
from __future__ import annotations

import time

import torch
from scipy.stats import chi2

from cdsbi.confidence_set.procedures import LikelihoodBasedProcedure, RatioBasedProcedure


def _toy_likelihood():
    """A toy Gaussian-mean likelihood: ll(θ; X) = -(X - θ)²/2.

    Set = {θ : 2(ll_max - ll(θ)) ≤ χ²} = {θ : (X - θ)² ≤ χ²}
        = [X - √χ², X + √χ²].
    """
    def ll(theta, x_obs):
        # theta (B, 1), x_obs (B, 1) → (B,)
        return -0.5 * ((x_obs - theta) ** 2).squeeze(-1)
    return LikelihoodBasedProcedure(log_likelihood_fn=ll, d_theta=1, theta_range=(-10.0, 10.0))


def test_likelihood_contains_batch_agrees_with_slow_path():
    proc = _toy_likelihood()
    x_batch = torch.linspace(-3.0, 3.0, 25).view(-1, 1)
    # Pick θ_0 = 0.5 so we have a mix of contains / not for various X.
    for alpha in [0.5, 0.9, 0.95]:
        slow = torch.tensor([
            proc.confidence_set(x_batch[i : i + 1], alpha=alpha).contains(0.5)
            for i in range(x_batch.shape[0])
        ])
        fast = proc.contains_batch(0.5, x_batch, alpha=alpha)
        assert torch.equal(slow, fast.cpu()), (
            f"alpha={alpha}: slow={slow.tolist()} fast={fast.cpu().tolist()}"
        )


def test_likelihood_contains_batch_matches_analytic():
    """For the toy Gaussian likelihood, the set is [X ± √χ²_{1,α}].

    contains(θ_0) ⇔ |X - θ_0| ≤ √χ²_{1,α}."""
    proc = _toy_likelihood()
    x_batch = torch.linspace(-3.0, 3.0, 100).view(-1, 1)
    for alpha in [0.5, 0.9, 0.95]:
        radius = float(chi2.ppf(alpha, df=1)) ** 0.5
        theta_0 = 0.5
        analytic = ((x_batch.squeeze(-1) - theta_0).abs() <= radius)
        fast = proc.contains_batch(theta_0, x_batch, alpha=alpha)
        # Allow 1-element disagreement at the boundary (grid resolution).
        diff = (analytic != fast.cpu()).sum().item()
        assert diff <= 1, f"alpha={alpha}: {diff} disagreements vs analytic"


def test_likelihood_contains_batch_faster_than_slow_path():
    proc = _toy_likelihood()
    B = 500
    x_batch = torch.randn(B, 1)
    alpha = 0.9
    # Warm-up
    _ = proc.contains_batch(0.0, x_batch[:10], alpha)
    _ = proc.confidence_set(x_batch[0:1], alpha)

    t0 = time.time()
    fast = proc.contains_batch(0.0, x_batch, alpha)
    t_fast = time.time() - t0
    t0 = time.time()
    slow = torch.tensor([
        proc.confidence_set(x_batch[i : i + 1], alpha=alpha).contains(0.0)
        for i in range(B)
    ])
    t_slow = time.time() - t0
    # Disagreements only at boundary points; allow up to 2.
    assert (slow != fast.cpu()).sum().item() <= 2
    assert t_fast * 10 < t_slow, f"fast={t_fast:.4f}s slow={t_slow:.4f}s"


def test_ratio_based_inherits_contains_batch():
    """RatioBasedProcedure.contains_batch should delegate to a Likelihood wrapper."""
    def lr(theta, x_obs):
        return -0.5 * ((x_obs - theta) ** 2).squeeze(-1)
    proc = RatioBasedProcedure(log_ratio_fn=lr, d_theta=1, theta_range=(-10.0, 10.0))
    x_batch = torch.linspace(-2.0, 2.0, 10).view(-1, 1)
    fast = proc.contains_batch(0.0, x_batch, alpha=0.9)
    assert fast.dtype == torch.bool
    assert fast.shape == (10,)
