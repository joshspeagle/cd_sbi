"""LikelihoodBasedProcedure.confidence_set in d=2."""
from __future__ import annotations

import torch
from scipy.stats import chi2

from cdsbi.confidence_set.procedures import LikelihoodBasedProcedure


def _toy_2d_gaussian_ll():
    """ll(θ; X) = -½ ‖X - θ‖² → MLE = X, Wilks set = ball X ± √χ²_{2,α}."""
    def ll(theta, x):
        return -0.5 * ((x - theta) ** 2).sum(dim=-1)
    return LikelihoodBasedProcedure(log_likelihood_fn=ll, d_theta=2, theta_range=(-7.0, 7.0))


def test_likelihood_d2_returns_2d_boundary(seed):
    proc = _toy_2d_gaussian_ll()
    x_obs = torch.tensor([[0.3, -0.2]])
    cs = proc.confidence_set(x_obs, alpha=0.9)
    assert cs.boundary_repr.ndim == 2
    assert cs.boundary_repr.shape[1] == 2
    # Analytic radius √χ²_{2, 0.9}; boundary should sit on a circle around X.
    radius = float(chi2.ppf(0.9, df=2)) ** 0.5
    centered = cs.boundary_repr - x_obs
    radii = centered.pow(2).sum(dim=-1).sqrt()
    assert (radii - radius).abs().max().item() < 0.05


def test_likelihood_d2_contains_uses_chisq_condition(seed):
    proc = _toy_2d_gaussian_ll()
    x_obs = torch.tensor([[0.3, -0.2]])
    cs = proc.confidence_set(x_obs, alpha=0.9)
    assert cs.contains([0.3, -0.2])  # at X itself → ll_max, in
    assert not cs.contains([5.0, 5.0])  # far → out
