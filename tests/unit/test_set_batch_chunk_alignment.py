"""Regression: batched d>1 LR set construction vs _chunked_inside slicing.

Pilot B (2026-06-10) crash: `_confidence_set_batch_d_gt_1`'s inside_fn inferred
the ll_max broadcast layout from theta_flat.shape[0] // B, but _chunked_inside
slices inputs at arbitrary 4096-row boundaries that do not align to multiples
of B — RuntimeError on non-divisible chunks (4050 vs 4096), and a *silent*
ll_max misalignment whenever the sizes happened to divide. The fix appends
ll_max as an extra x-column so row alignment survives every expansion/slice.

B=25 with n_rays=200 gives 5000 ray rows -> chunks of 4096 + 904, neither a
multiple of 25: exactly the geometry that crashed.
"""
import math

import pytest
import torch
from scipy.stats import chi2

from cdsbi.confidence_set.procedures import LikelihoodBasedProcedure


def _gauss_ll(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # X | theta ~ N(theta, I_2): ll = -0.5 ||x - theta||^2 (+ const).
    return -0.5 * ((x - theta) ** 2).sum(dim=-1)


@pytest.fixture
def proc():
    return LikelihoodBasedProcedure(
        _gauss_ll, d_theta=2, theta_range=(-6.0, 6.0), refine_ll_max=False,
    )


def test_batch_set_d2_crosses_chunk_boundary(proc):
    """B*K = 5000 > 4096 with B=25 not dividing 4096 — crashed pre-fix."""
    torch.manual_seed(0)
    x = torch.randn(25, 2)
    centers, boundaries = proc.confidence_set_batch(x, alpha=0.9)
    assert centers.shape == (25, 2)
    assert boundaries.shape == (25, 200, 2)
    assert torch.isfinite(boundaries).all()


def test_batch_set_d2_boundary_radius_matches_analytic(proc):
    """For N(theta, I_2) the LR set is the ball ||theta - x|| <= sqrt(chi2_{2,a})
    around x — every ray-boundary point must sit at that radius, for every row
    including the ones split across the 4096-row chunk boundary."""
    torch.manual_seed(1)
    x = torch.randn(25, 2)
    alpha = 0.9
    r_expect = math.sqrt(chi2.ppf(alpha, df=2))
    centers, boundaries = proc.confidence_set_batch(x, alpha=alpha)
    # Center is the MLE = x itself.
    assert torch.allclose(centers, x, atol=0.05)
    radii = (boundaries - x.unsqueeze(1)).norm(dim=-1)  # (B, K)
    assert (radii - r_expect).abs().max().item() < 0.05


def test_batch_set_d2_refinement_path(proc):
    """The refine_ll_max=True branch (gradient polish of the coarse-mesh
    ll_max) runs and can only tighten: refined ll_max >= mesh ll_max means
    radii <= the unrefined ones (equal here since the mesh already brackets
    the smooth optimum)."""
    refined = LikelihoodBasedProcedure(
        _gauss_ll, d_theta=2, theta_range=(-6.0, 6.0), refine_ll_max=True,
    )
    torch.manual_seed(2)
    x = torch.randn(5, 2)
    centers, boundaries = refined.confidence_set_batch(x, alpha=0.9)
    r_expect = math.sqrt(chi2.ppf(0.9, df=2))
    radii = (boundaries - x.unsqueeze(1)).norm(dim=-1)
    assert (radii - r_expect).abs().max().item() < 0.05
