"""Wilks ll_max refinement (hardening item 2).

At d>1 `LikelihoodBasedProcedure.contains_batch` estimated ll_max(X) from a
fixed 200-point uniform random grid; the nearest-neighbor gap under-estimates
ll_max, which SHRINKS the Wilks statistic 2(ll_max − ll(θ0)) and over-includes
candidate θ0 (biased, X-dependent). The fix refines ll_max by gradient ascent
from the best grid points (log-likelihood is differentiable in θ for NLE flows
and NRE classifiers), with elementwise max so the estimate can only improve.

Closed-form testbed: ll(θ;x) = −½‖θ−x‖² → exact ll_max = 0 at θ = x, and the
Wilks set for θ0 is {‖θ0−x‖² ≤ χ²_{d,α}} exactly.
"""
import math

import torch
from scipy.stats import chi2

from cdsbi.confidence_set.procedures import (
    LikelihoodBasedProcedure,
    RatioBasedProcedure,
)


def _gauss_ll(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return -0.5 * ((theta - x) ** 2).sum(dim=-1)


def _x_at_distance(theta_0: torch.Tensor, dist2: float, n: int, seed: int) -> torch.Tensor:
    """n points x with ‖θ0 − x‖² = dist2 exactly, random directions."""
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, theta_0.shape[-1], generator=g)
    u = u / u.norm(dim=-1, keepdim=True)
    return theta_0 - math.sqrt(dist2) * u


def test_d2_exclusion_just_outside_threshold():
    """θ0 with ‖θ0−x‖² = χ² + 0.1 is OUTSIDE the true Wilks set. The refined
    path must exclude it for every x; the old grid-only path over-includes
    (contained iff 2·gap > 0.1, and the 200-pt grid's gap usually exceeds 0.05)."""
    d, alpha = 2, 0.95
    thresh = float(chi2.ppf(alpha, df=d))
    theta_0 = torch.zeros(d)
    x = _x_at_distance(theta_0, thresh + 0.1, n=64, seed=0)

    refined = LikelihoodBasedProcedure(_gauss_ll, d, theta_range=(-7.0, 7.0))
    got = refined.contains_batch(theta_0, x, alpha)
    assert not bool(got.any()), f"refined path over-included {int(got.sum())}/64"

    legacy = LikelihoodBasedProcedure(
        _gauss_ll, d, theta_range=(-7.0, 7.0), refine_ll_max=False
    )
    got_old = legacy.contains_batch(theta_0, x, alpha)
    assert bool(got_old.any()), (
        "expected the legacy grid-only path to over-include at least one point "
        "(the documented bias); if this fails the testbed no longer exposes it"
    )


def test_d2_inclusion_just_inside_threshold():
    d, alpha = 2, 0.95
    thresh = float(chi2.ppf(alpha, df=d))
    theta_0 = torch.zeros(d)
    x = _x_at_distance(theta_0, thresh - 0.1, n=64, seed=1)
    proc = LikelihoodBasedProcedure(_gauss_ll, d, theta_range=(-7.0, 7.0))
    got = proc.contains_batch(theta_0, x, alpha)
    assert bool(got.all()), f"refined path under-included {int((~got).sum())}/64"


def test_d1_tight_margin_correct():
    """1D linspace is denser but theta_range (−20,20) still leaves a ~0.01 stat
    gap; refinement makes a ±0.05 margin decisively correct."""
    d, alpha = 1, 0.95
    thresh = float(chi2.ppf(alpha, df=d))
    theta_0 = torch.zeros(d)
    proc = LikelihoodBasedProcedure(_gauss_ll, d, theta_range=(-20.0, 20.0))
    x_in = _x_at_distance(theta_0, thresh - 0.05, n=32, seed=2)
    x_out = _x_at_distance(theta_0, thresh + 0.05, n=32, seed=3)
    assert bool(proc.contains_batch(theta_0, x_in, alpha).all())
    assert not bool(proc.contains_batch(theta_0, x_out, alpha).any())


def test_nre_wrapper_inherits_refinement():
    d, alpha = 2, 0.95
    thresh = float(chi2.ppf(alpha, df=d))
    theta_0 = torch.zeros(d)
    x = _x_at_distance(theta_0, thresh + 0.1, n=32, seed=4)
    nre = RatioBasedProcedure(_gauss_ll, d, theta_range=(-7.0, 7.0))
    assert not bool(nre.contains_batch(theta_0, x, alpha).any())


def test_deterministic():
    d, alpha = 2, 0.9
    theta_0 = torch.tensor([1.0, -1.0])
    g = torch.Generator().manual_seed(5)
    x = torch.randn(16, d, generator=g) * 2
    a = LikelihoodBasedProcedure(_gauss_ll, d, theta_range=(-7.0, 7.0))
    b = LikelihoodBasedProcedure(_gauss_ll, d, theta_range=(-7.0, 7.0))
    assert torch.equal(a.contains_batch(theta_0, x, alpha),
                       b.contains_batch(theta_0, x, alpha))


def test_confidence_set_d2_boundary_accuracy():
    """The d>1 confidence_set path (25²-mesh ll_max) must also be refined: the
    set for a single x is the disc ‖θ−x‖² ≤ χ²; check containment at ±5%."""
    d, alpha = 2, 0.95
    thresh = float(chi2.ppf(alpha, df=d))
    x = torch.tensor([[0.7, -0.3]])
    proc = LikelihoodBasedProcedure(_gauss_ll, d, theta_range=(-7.0, 7.0))
    cs = proc.confidence_set(x, alpha)
    r_in = math.sqrt(thresh * 0.95)
    r_out = math.sqrt(thresh * 1.05)
    for ang in (0.0, 1.0, 2.5, 4.0):
        v = torch.tensor([math.cos(ang), math.sin(ang)])
        p_in = (x[0] + r_in * v)
        p_out = (x[0] + r_out * v)
        assert cs.contains(p_in), f"boundary too small at angle {ang}"
        assert not cs.contains(p_out), f"boundary too large at angle {ang}"


def test_nondifferentiable_ll_falls_back_to_local_search():
    """A no_grad-wrapped (graph-killing) log-likelihood must not crash the
    refinement — it falls back to the deterministic local search and still
    beats the grid-only estimate (the d=2 exclusion case)."""
    d, alpha = 2, 0.95
    thresh = float(chi2.ppf(alpha, df=d))

    def ll_nograd(theta, x):
        with torch.no_grad():
            return _gauss_ll(theta, x)

    theta_0 = torch.zeros(d)
    x = _x_at_distance(theta_0, thresh + 0.1, n=64, seed=6)
    proc = LikelihoodBasedProcedure(ll_nograd, d, theta_range=(-7.0, 7.0))
    got = proc.contains_batch(theta_0, x, alpha)
    assert not bool(got.any()), f"fallback path over-included {int(got.sum())}/64"
