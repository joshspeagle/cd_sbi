"""CriticalValueProcedure.confidence_set in d=2."""
from __future__ import annotations

import torch
from scipy.stats import chi2

from cdsbi.confidence_set.procedures import CriticalValueProcedure


def _toy_2d_quadratic_T():
    """T(θ; X) = ‖θ - X‖²; c_α = χ²_{2, α} → set = ball X ± √χ²."""
    def t_stat(theta, x):
        return ((theta - x) ** 2).sum(dim=-1)
    def c_fn(theta, alpha):
        return torch.full((theta.shape[0],), float(chi2.ppf(alpha, df=2)))
    return CriticalValueProcedure(
        test_stat_fn=t_stat, critical_value_fn=c_fn, d_theta=2, theta_range=(-7.0, 7.0),
    )


def test_critical_value_d2_returns_2d_boundary(seed):
    proc = _toy_2d_quadratic_T()
    x_obs = torch.tensor([[0.5, -0.3]])
    cs = proc.confidence_set(x_obs, alpha=0.9)
    assert cs.boundary_repr.ndim == 2
    assert cs.boundary_repr.shape[1] == 2
    radius = float(chi2.ppf(0.9, df=2)) ** 0.5
    centered = cs.boundary_repr - x_obs
    radii = centered.pow(2).sum(dim=-1).sqrt()
    assert (radii - radius).abs().max().item() < 0.05


def test_critical_value_d2_contains_uses_threshold(seed):
    proc = _toy_2d_quadratic_T()
    x_obs = torch.tensor([[0.5, -0.3]])
    cs = proc.confidence_set(x_obs, alpha=0.9)
    assert cs.contains([0.5, -0.3])  # at X → T = 0
    assert not cs.contains([5.0, 5.0])
