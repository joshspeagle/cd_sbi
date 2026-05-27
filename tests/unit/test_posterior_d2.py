"""PosteriorBasedProcedure.confidence_set in d=2 via empirical Mahalanobis."""
from __future__ import annotations

import torch
from scipy.stats import chi2

from cdsbi.confidence_set.procedures import PosteriorBasedProcedure


def _normal_2d_posterior_fn():
    """Fake NPE: posterior is N([0.5, -0.3], diag(0.5)). Returns samples (n, 2)."""
    def sample_fn(x_obs, n):
        torch.manual_seed(0)
        mu = torch.tensor([0.5, -0.3])
        scale = 0.5 ** 0.5
        return mu + scale * torch.randn(n, 2)
    return PosteriorBasedProcedure(sample_fn=sample_fn, d_theta=2)


def test_posterior_d2_returns_2d_boundary(seed):
    proc = _normal_2d_posterior_fn()
    x_obs = torch.tensor([[0.0, 0.0]])
    cs = proc.confidence_set(x_obs, alpha=0.9)
    # boundary_repr should be (K, 2).
    assert cs.boundary_repr.ndim == 2
    assert cs.boundary_repr.shape[1] == 2
    assert cs.boundary_repr.shape[0] >= 100  # default 200


def test_posterior_d2_contains_mean_excludes_far(seed):
    proc = _normal_2d_posterior_fn()
    x_obs = torch.tensor([[0.0, 0.0]])
    cs = proc.confidence_set(x_obs, alpha=0.9)
    assert cs.contains([0.5, -0.3])  # posterior mean → maximally inside
    assert not cs.contains([10.0, 10.0])


def test_posterior_d2_matches_analytic_radius_for_isotropic_posterior(seed):
    """For posterior = N([μ_1, μ_2], σ² I_2), the Mahalanobis radius squared
    of the α-set is the α-quantile of χ²_2 — match within MC noise."""
    proc = _normal_2d_posterior_fn()  # σ² = 0.5, isotropic
    x_obs = torch.tensor([[0.0, 0.0]])
    for alpha in [0.5, 0.9, 0.95]:
        cs = proc.confidence_set(x_obs, alpha=alpha)
        mu = torch.tensor([0.5, -0.3])
        scale = 0.5 ** 0.5
        # In whitened space the boundary should sit at √χ²_{2, α}.
        # In raw space the boundary is on the ellipsoid (μ, σ² I), so radius
        # in raw space is σ √χ²_{2, α} = √(0.5 · χ²_{2, α}).
        raw_radius = (0.5 * float(chi2.ppf(alpha, df=2))) ** 0.5
        centered = cs.boundary_repr - mu
        empirical_radii = centered.pow(2).sum(dim=-1).sqrt()
        # MC tolerance: σ̂ ≈ σ (1 ± 1/√(2n)); at n=10000, ~1% noise on radius.
        assert (empirical_radii - raw_radius).abs().max().item() < 0.10, (
            f"alpha={alpha}: raw_radius={raw_radius:.3f}, "
            f"observed range [{empirical_radii.min().item():.3f}, "
            f"{empirical_radii.max().item():.3f}]"
        )


def test_posterior_d2_captures_correlation(seed):
    """For a correlated posterior, the Mahalanobis region is NOT axis-aligned —
    contains a point along the correlation axis that an axis-aligned interval
    would exclude.

    Posterior: θ ~ N(0, [[1, 0.8], [0.8, 1]]).
    """
    rho = 0.8
    cov = torch.tensor([[1.0, rho], [rho, 1.0]])
    L = torch.linalg.cholesky(cov)

    def sample_fn(x_obs, n):
        torch.manual_seed(42)
        return torch.randn(n, 2) @ L.T

    proc = PosteriorBasedProcedure(sample_fn=sample_fn, d_theta=2)
    x_obs = torch.tensor([[0.0, 0.0]])
    cs = proc.confidence_set(x_obs, alpha=0.9)
    # A point along the diagonal (correlation axis): θ = (r, r) with
    # ||θ||²_Σ^{-1} = (r, r) Σ^{-1} (r, r)^T = 2r²/(1+ρ) — small for any r at fixed length.
    # Versus anti-correlation axis: (r, -r) has Mahalanobis 2r²/(1-ρ), larger.
    # So (1.0, 1.0) should be INSIDE while (1.0, -1.0) is OUTSIDE for the
    # right α and ρ. With ρ=0.8, ratio is 1.8/0.2 = 9× — comfortable.
    assert cs.contains([1.0, 1.0])
    assert not cs.contains([1.5, -1.5])
