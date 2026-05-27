"""confidence_set_batch fast-path agreement and speed tests for non-PivotBased procedures."""
from __future__ import annotations

import time

import torch
from scipy.stats import chi2

from cdsbi.confidence_set.procedures import (
    CriticalValueProcedure,
    LikelihoodBasedProcedure,
    PosteriorBasedProcedure,
    RatioBasedProcedure,
)


def _toy_likelihood():
    """ll(θ; X) = -(X - θ)²/2 → Wilks set is [X ± √χ²_{1, α}]."""
    def ll(theta, x):
        return -0.5 * ((x - theta) ** 2).squeeze(-1)
    return LikelihoodBasedProcedure(log_likelihood_fn=ll, d_theta=1, theta_range=(-10.0, 10.0))


def _toy_critical_value():
    """T(θ; X) = (θ - X)²; c_α = χ²_{1, α} → set = [X ± √χ²]."""
    def t_stat(theta, x):
        return ((theta - x) ** 2).squeeze(-1)
    def c_fn(theta, alpha):
        return torch.full((theta.shape[0],), float(chi2.ppf(alpha, df=1)))
    return CriticalValueProcedure(
        test_stat_fn=t_stat, critical_value_fn=c_fn, d_theta=1, theta_range=(-10.0, 10.0),
    )


def test_likelihood_batch_matches_analytic():
    proc = _toy_likelihood()
    x_batch = torch.linspace(-2.0, 2.0, 25).view(-1, 1)
    for alpha in [0.5, 0.9, 0.95]:
        radius = float(chi2.ppf(alpha, df=1)) ** 0.5
        left, right = proc.confidence_set_batch(x_batch, alpha)
        analytic_left = x_batch.squeeze(-1) - radius
        analytic_right = x_batch.squeeze(-1) + radius
        assert torch.allclose(left, analytic_left, atol=5e-3)
        assert torch.allclose(right, analytic_right, atol=5e-3)


def test_likelihood_batch_agrees_with_slow_path():
    proc = _toy_likelihood()
    x_batch = torch.randn(20, 1)
    left, right = proc.confidence_set_batch(x_batch, alpha=0.9)
    for i in range(20):
        cs = proc.confidence_set(x_batch[i : i + 1], alpha=0.9)
        l_slow = float(cs.boundary_repr[0])
        r_slow = float(cs.boundary_repr[1])
        assert abs(float(left[i]) - l_slow) < 5e-3
        assert abs(float(right[i]) - r_slow) < 5e-3


def test_likelihood_batch_is_faster_than_slow_path():
    proc = _toy_likelihood()
    B = 500
    x_batch = torch.randn(B, 1)
    _ = proc.confidence_set_batch(x_batch[:10], 0.9)  # warm-up

    t0 = time.time()
    left_fast, right_fast = proc.confidence_set_batch(x_batch, 0.9)
    t_fast = time.time() - t0
    t0 = time.time()
    for i in range(B):
        proc.confidence_set(x_batch[i : i + 1], 0.9)
    t_slow = time.time() - t0
    assert t_fast * 10 < t_slow, f"fast={t_fast:.4f}s slow={t_slow:.4f}s"


def test_ratio_inherits_batch():
    def lr(theta, x):
        return -0.5 * ((x - theta) ** 2).squeeze(-1)
    proc = RatioBasedProcedure(log_ratio_fn=lr, d_theta=1, theta_range=(-10.0, 10.0))
    x_batch = torch.linspace(-1.0, 1.0, 10).view(-1, 1)
    left, right = proc.confidence_set_batch(x_batch, 0.9)
    assert left.shape == (10,) and right.shape == (10,)
    assert (right > left).all()


def test_critical_value_batch_matches_analytic():
    proc = _toy_critical_value()
    x_batch = torch.linspace(-2.0, 2.0, 25).view(-1, 1)
    for alpha in [0.5, 0.9, 0.95]:
        radius = float(chi2.ppf(alpha, df=1)) ** 0.5
        left, right = proc.confidence_set_batch(x_batch, alpha)
        analytic_left = x_batch.squeeze(-1) - radius
        analytic_right = x_batch.squeeze(-1) + radius
        assert torch.allclose(left, analytic_left, atol=5e-3)
        assert torch.allclose(right, analytic_right, atol=5e-3)


def test_posterior_batch_with_sample_batched_fn():
    """With a sample_batched_fn (NPE wires this), batched quantiles match
    per-X_obs results."""
    def sample_fn(x_obs, n):
        return float(x_obs.flatten()[0]) + torch.randn(n, 1)

    def sample_batched_fn(x_obs_batch, n):
        center = x_obs_batch.squeeze(-1)
        return center.view(1, -1, 1) + torch.randn(n, x_obs_batch.shape[0], 1)

    proc = PosteriorBasedProcedure(
        sample_fn=sample_fn, d_theta=1, sample_batched_fn=sample_batched_fn,
    )
    torch.manual_seed(0)
    x_batch = torch.linspace(-1.0, 1.0, 5).view(-1, 1)
    left, right = proc.confidence_set_batch(x_batch, alpha=0.9, n_samples=20000)
    # Roughly: interval ≈ [X - 1.645, X + 1.645].
    expected_lo = x_batch.squeeze(-1) - 1.645
    expected_hi = x_batch.squeeze(-1) + 1.645
    # MC tolerance at n=20000:
    assert torch.allclose(left, expected_lo, atol=0.05)
    assert torch.allclose(right, expected_hi, atol=0.05)


def test_posterior_batch_fallback_no_batched_fn():
    """Without sample_batched_fn, falls back to per-X_obs loop. Same output."""
    def sample_fn(x_obs, n):
        return float(x_obs.flatten()[0]) + torch.randn(n, 1)

    proc = PosteriorBasedProcedure(
        sample_fn=sample_fn, d_theta=1, sample_batched_fn=None,
    )
    torch.manual_seed(0)
    x_batch = torch.linspace(-1.0, 1.0, 3).view(-1, 1)
    left, right = proc.confidence_set_batch(x_batch, alpha=0.9, n_samples=2000)
    assert left.shape == (3,) and right.shape == (3,)
    assert (right > left).all()
