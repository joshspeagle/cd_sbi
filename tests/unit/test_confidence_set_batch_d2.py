"""confidence_set_batch in d=2: agreement with the per-X_obs slow path and
analytic widths for toy 2D Gaussians."""
from __future__ import annotations

import time

import torch
from scipy.stats import chi2

from cdsbi.confidence_set.procedures import (
    CriticalValueProcedure,
    LikelihoodBasedProcedure,
    PivotBasedProcedure,
    PosteriorBasedProcedure,
    RatioBasedProcedure,
)


def test_pivot_d2_batch_matches_analytic(seed):
    """r = θ - X → ellipsoid of radius √χ²_{2,α}."""
    def pivot(th, x):
        return th - x
    proc = PivotBasedProcedure(pivot_fn=pivot, d_theta=2, theta_range=(-7.0, 7.0))
    x_batch = torch.linspace(-2.0, 2.0, 10).view(-1, 1).expand(-1, 2)
    for alpha in [0.5, 0.9]:
        centers, boundaries = proc.confidence_set_batch(x_batch, alpha)
        assert centers.shape == (10, 2)
        assert boundaries.shape[0] == 10 and boundaries.shape[2] == 2
        radius = float(chi2.ppf(alpha, df=2)) ** 0.5
        # Each boundary point should sit ~radius from its center.
        radii = (boundaries - centers.unsqueeze(1)).pow(2).sum(dim=-1).sqrt()
        assert (radii - radius).abs().max().item() < 5e-2


def test_likelihood_d2_batch_matches_analytic(seed):
    """ll = -½‖x-θ‖² → MLE = X, Wilks set = disk of radius √χ²_{2,α}."""
    def ll(th, x):
        return -0.5 * ((x - th) ** 2).sum(dim=-1)
    proc = LikelihoodBasedProcedure(log_likelihood_fn=ll, d_theta=2, theta_range=(-7.0, 7.0))
    x_batch = torch.linspace(-2.0, 2.0, 10).view(-1, 1).expand(-1, 2)
    for alpha in [0.5, 0.9]:
        centers, boundaries = proc.confidence_set_batch(x_batch, alpha)
        radius = float(chi2.ppf(alpha, df=2)) ** 0.5
        radii = (boundaries - centers.unsqueeze(1)).pow(2).sum(dim=-1).sqrt()
        # Loose tolerance (5×) — the helper's grid+L-BFGS approximates ll_max.
        assert (radii - radius).abs().max().item() < 0.15


def test_critical_value_d2_batch_matches_analytic(seed):
    """T = ‖θ - X‖²; c_α = χ²_{2,α} → disk of radius √χ²."""
    def t_stat(th, x):
        return ((th - x) ** 2).sum(dim=-1)
    def c_fn(th, alpha):
        return torch.full((th.shape[0],), float(chi2.ppf(alpha, df=2)))
    proc = CriticalValueProcedure(
        test_stat_fn=t_stat, critical_value_fn=c_fn, d_theta=2, theta_range=(-7.0, 7.0),
    )
    x_batch = torch.linspace(-2.0, 2.0, 10).view(-1, 1).expand(-1, 2)
    centers, boundaries = proc.confidence_set_batch(x_batch, alpha=0.9)
    radius = float(chi2.ppf(0.9, df=2)) ** 0.5
    radii = (boundaries - centers.unsqueeze(1)).pow(2).sum(dim=-1).sqrt()
    assert (radii - radius).abs().max().item() < 5e-2


def test_ratio_d2_batch_delegates_to_likelihood(seed):
    def lr(th, x):
        return -0.5 * ((x - th) ** 2).sum(dim=-1)
    proc = RatioBasedProcedure(log_ratio_fn=lr, d_theta=2, theta_range=(-7.0, 7.0))
    x_batch = torch.linspace(-2.0, 2.0, 10).view(-1, 1).expand(-1, 2)
    centers, boundaries = proc.confidence_set_batch(x_batch, alpha=0.9)
    assert centers.shape == (10, 2)
    assert boundaries.shape == (10, 200, 2)


def test_posterior_d2_batch_matches_analytic(seed):
    """Posterior = N(X, σ² I_2) → Mahalanobis disk of radius σ √χ²_{2,α}."""
    sigma2 = 0.5

    def sample_fn(x_obs, n):
        center = x_obs.flatten()[: 2]
        return center + (sigma2 ** 0.5) * torch.randn(n, 2)

    def sample_batched_fn(x_obs_batch, n):
        return x_obs_batch.unsqueeze(0) + (sigma2 ** 0.5) * torch.randn(
            n, x_obs_batch.shape[0], 2,
        )

    proc = PosteriorBasedProcedure(
        sample_fn=sample_fn, d_theta=2, sample_batched_fn=sample_batched_fn,
    )
    x_batch = torch.tensor([[0.0, 0.0], [1.0, -1.0], [2.0, 2.0]])
    centers, boundaries = proc.confidence_set_batch(x_batch, alpha=0.9, n_samples=10000)
    # Centers should be ≈ X_obs (posterior mean).
    assert torch.allclose(centers, x_batch, atol=0.05)
    raw_radius = (sigma2 * float(chi2.ppf(0.9, df=2))) ** 0.5
    radii = (boundaries - centers.unsqueeze(1)).pow(2).sum(dim=-1).sqrt()
    # MC tolerance at n=10000.
    assert (radii.mean(dim=1) - raw_radius).abs().max().item() < 0.10


def test_posterior_d2_contains_batch_matches_analytic(seed):
    """For posterior = N(X_obs, σ² I), a point exactly at X_obs is always
    inside; a point far away is always outside."""
    sigma2 = 0.5

    def sample_fn(x_obs, n):
        center = x_obs.flatten()[:2]
        return center + (sigma2 ** 0.5) * torch.randn(n, 2)

    def sample_batched_fn(x_obs_batch, n):
        return x_obs_batch.unsqueeze(0) + (sigma2 ** 0.5) * torch.randn(
            n, x_obs_batch.shape[0], 2,
        )

    proc = PosteriorBasedProcedure(
        sample_fn=sample_fn, d_theta=2, sample_batched_fn=sample_batched_fn,
    )
    x_batch = torch.tensor([[0.0, 0.0], [1.0, -1.0], [2.0, 2.0]])
    # Probe at each X_obs's own center — all 3 should be inside.
    for i, theta_at_x in enumerate(x_batch):
        inside = proc.contains_batch(theta_at_x.tolist(), x_batch, alpha=0.9, n_samples=5000)
        assert inside.dtype == torch.bool and inside.shape == (3,)
        assert bool(inside[i]), f"θ at X_obs[{i}] should be inside its own posterior"
    # Far-away θ → outside everywhere.
    inside = proc.contains_batch([10.0, 10.0], x_batch, alpha=0.9, n_samples=5000)
    assert not inside.any(), f"far-away θ should be outside; got {inside.tolist()}"


def test_pivot_d2_batch_faster_than_slow_path():
    """Speedup vs. per-X_obs _confidence_set_d_gt_1 loop."""
    def pivot(th, x):
        return th - x
    proc = PivotBasedProcedure(pivot_fn=pivot, d_theta=2, theta_range=(-7.0, 7.0))
    B = 100
    x_batch = torch.randn(B, 2)
    _ = proc.confidence_set_batch(x_batch[:5], 0.9)  # warm-up

    t0 = time.time()
    centers, boundaries = proc.confidence_set_batch(x_batch, 0.9)
    t_fast = time.time() - t0
    t0 = time.time()
    for i in range(B):
        proc.confidence_set(x_batch[i : i + 1], 0.9)
    t_slow = time.time() - t0
    assert t_fast * 5 < t_slow, f"fast {t_fast:.3f}s slow {t_slow:.3f}s"
