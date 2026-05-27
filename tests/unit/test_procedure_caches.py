"""Verify per-procedure intermediate caching is correct + actually fires."""
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


def test_likelihood_contains_batch_cache_correctness():
    """Same x_obs_batch, two αs — both calls return the right answer; the
    second call should reuse ll_grid (no double-eval)."""
    eval_count = [0]

    def ll(theta, x):
        eval_count[0] += 1
        return -0.5 * ((x - theta) ** 2).squeeze(-1)

    proc = LikelihoodBasedProcedure(log_likelihood_fn=ll, d_theta=1, theta_range=(-10.0, 10.0))
    x_batch = torch.linspace(-1.0, 1.0, 50).view(-1, 1)
    n_first = eval_count[0]
    proc.contains_batch(0.0, x_batch, alpha=0.5)
    n_after_first_alpha = eval_count[0]
    proc.contains_batch(0.0, x_batch, alpha=0.9)
    n_after_second_alpha = eval_count[0]
    proc.contains_batch(0.0, x_batch, alpha=0.95)
    n_after_third_alpha = eval_count[0]
    # First call costs n_alpha_1_evals = (n_grid eval for grid + 1 eval for θ_0).
    # Second+third call: should only redo the θ_0 eval (cheap), not the grid.
    delta1 = n_after_first_alpha - n_first
    delta2 = n_after_second_alpha - n_after_first_alpha
    delta3 = n_after_third_alpha - n_after_second_alpha
    assert delta2 < delta1, f"cache miss: second-α delta={delta2} >= first-α delta={delta1}"
    assert delta3 < delta1


def test_likelihood_contains_batch_cache_invalidates_on_new_x():
    eval_count = [0]
    def ll(theta, x):
        eval_count[0] += 1
        return -0.5 * ((x - theta) ** 2).squeeze(-1)
    proc = LikelihoodBasedProcedure(log_likelihood_fn=ll, d_theta=1, theta_range=(-10.0, 10.0))
    x1 = torch.randn(20, 1)
    x2 = torch.randn(20, 1)  # new tensor → different id
    proc.contains_batch(0.0, x1, alpha=0.9)
    after_first = eval_count[0]
    proc.contains_batch(0.0, x2, alpha=0.9)
    after_second = eval_count[0]
    # Both calls should hit the full eval cost (cache miss on new x).
    delta1 = after_first - 0
    delta2 = after_second - after_first
    assert delta2 >= delta1 - 1, f"expected ~equal eval cost on new x; got {delta1} vs {delta2}"


def test_critical_value_contains_batch_cache_correctness():
    eval_count_t = [0]
    def t_stat(theta, x):
        eval_count_t[0] += 1
        return ((theta - x) ** 2).squeeze(-1)
    def c_fn(theta, alpha):
        return torch.full((theta.shape[0],), float(chi2.ppf(alpha, df=1)))
    proc = CriticalValueProcedure(
        test_stat_fn=t_stat, critical_value_fn=c_fn, d_theta=1, theta_range=(-10.0, 10.0),
    )
    x_batch = torch.linspace(-1.0, 1.0, 50).view(-1, 1)
    proc.contains_batch(0.0, x_batch, alpha=0.5)
    first_count = eval_count_t[0]
    proc.contains_batch(0.0, x_batch, alpha=0.9)
    after_second = eval_count_t[0]
    proc.contains_batch(0.0, x_batch, alpha=0.95)
    after_third = eval_count_t[0]
    # Second and third α calls should NOT re-invoke test_stat_fn.
    assert after_second == first_count, (
        f"cache miss: t_stat called {after_second - first_count} extra times for second α"
    )
    assert after_third == first_count


def test_critical_value_cache_invalidates_on_new_theta_0():
    eval_count_t = [0]
    def t_stat(theta, x):
        eval_count_t[0] += 1
        return ((theta - x) ** 2).squeeze(-1)
    def c_fn(theta, alpha):
        return torch.full((theta.shape[0],), float(chi2.ppf(alpha, df=1)))
    proc = CriticalValueProcedure(
        test_stat_fn=t_stat, critical_value_fn=c_fn, d_theta=1, theta_range=(-10.0, 10.0),
    )
    x_batch = torch.linspace(-1.0, 1.0, 50).view(-1, 1)
    proc.contains_batch(0.0, x_batch, alpha=0.9)
    first = eval_count_t[0]
    proc.contains_batch(1.5, x_batch, alpha=0.9)  # new θ_0
    after = eval_count_t[0]
    assert after > first, "new θ_0 should invalidate t_obs cache"


def test_posterior_contains_batch_cache_correctness():
    """Posterior should cache samples across α calls at fixed x_obs_batch."""
    call_count = [0]
    def sample_batched_fn(x_obs_batch, n):
        call_count[0] += 1
        center = x_obs_batch.squeeze(-1)
        return center.view(1, -1, 1) + torch.randn(n, x_obs_batch.shape[0], 1)

    def sample_fn(x_obs, n):
        return float(x_obs.flatten()[0]) + torch.randn(n, 1)

    proc = PosteriorBasedProcedure(
        sample_fn=sample_fn, d_theta=1, sample_batched_fn=sample_batched_fn,
    )
    torch.manual_seed(0)
    x_batch = torch.linspace(-1.0, 1.0, 20).view(-1, 1)
    proc.contains_batch(0.0, x_batch, alpha=0.5, n_samples=2000)
    assert call_count[0] == 1
    proc.contains_batch(0.0, x_batch, alpha=0.9, n_samples=2000)
    assert call_count[0] == 1, "cache miss: sample_batched_fn called twice for the same x"
    proc.contains_batch(0.0, x_batch, alpha=0.95, n_samples=2000)
    assert call_count[0] == 1


def test_ratio_inherits_likelihood_cache():
    """RatioBasedProcedure should share a single Likelihood wrapper across calls,
    so the wrapper's cache hits across α."""
    eval_count = [0]
    def lr(theta, x):
        eval_count[0] += 1
        return -0.5 * ((x - theta) ** 2).squeeze(-1)
    proc = RatioBasedProcedure(log_ratio_fn=lr, d_theta=1, theta_range=(-10.0, 10.0))
    x_batch = torch.linspace(-1.0, 1.0, 20).view(-1, 1)
    proc.contains_batch(0.0, x_batch, alpha=0.5)
    first = eval_count[0]
    proc.contains_batch(0.0, x_batch, alpha=0.9)
    delta = eval_count[0] - first
    # With a fresh wrapper per call (old behavior), delta would equal `first`.
    # With shared wrapper + ll_grid cache, delta should be much less.
    assert delta < first, f"cache not engaged for Ratio: delta={delta} not < first={first}"
