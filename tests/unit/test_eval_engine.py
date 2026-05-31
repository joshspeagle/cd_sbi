"""Eval-engine correctness + chunking-invariance.

Correctness: for an *exact* analytic pivot/statistic, engine coverage must match the
nominal level (the engine's coverage math is right). Chunking-invariance: the
memory-bounding chunk size must not change any number (chunk is a perf knob only)."""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import chi2

from cdsbi.confidence_set.procedures import PivotBasedProcedure, CriticalValueProcedure
from cdsbi.diagnostics.engine import evaluate_coverage


class _ToyGaussian:
    """X | θ ~ N(θ, I_d), so r=(X−θ) is an exact pivot: ‖r‖²~χ²_d at truth."""
    def __init__(self, d):
        self.d_theta = d
        self.d_x = d
        self.theta_range = (-10.0, 10.0)

    def sample_x_given_theta(self, theta_0, n, rng):
        v = np.atleast_1d(np.asarray(theta_0, dtype=np.float64)).reshape(-1)
        x = rng.normal(v, 1.0, size=(n, self.d_theta))
        return torch.tensor(x, dtype=torch.float32)


def test_engine_pivot_coverage_is_exact():
    sim = _ToyGaussian(2)
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: x - th, d_theta=2)
    out = evaluate_coverage(proc, sim, theta_grid=[(0.0, 0.0), (1.0, -2.0)],
                            alpha_grid=[0.5, 0.9, 0.95], n_per_theta=20000, seed=0)
    cov = out["coverage"]
    # exact pivot ⇒ empirical ≈ nominal within MC noise (~1/sqrt(20000)≈0.007)
    assert (cov["empirical"] - cov["nominal"]).abs().max() < 0.02
    assert out["coverage_error_max"] < 0.02
    # pivot metrics present + sane (‖r‖²~χ²_2 exactly; per-coord r_k~N(0,1))
    assert out["joint_chi2_ks_max"] < 0.03
    assert out["per_coord_pit_ks_max"] < 0.03
    assert out["per_coord_pit_ks"].shape == (2, 2)


def test_engine_critical_value_coverage_is_exact():
    sim = _ToyGaussian(2)
    proc = CriticalValueProcedure(
        test_stat_fn=lambda th, x: ((x - th) ** 2).sum(-1),
        critical_value_fn=lambda th, a: torch.tensor(float(chi2.ppf(a, df=2))),
        d_theta=2, theta_range=(-10.0, 10.0),
    )
    out = evaluate_coverage(proc, sim, theta_grid=[(0.0, 0.0), (-1.0, 1.0)],
                            alpha_grid=[0.5, 0.9, 0.95], n_per_theta=20000, seed=0)
    assert out["coverage_error_max"] < 0.02


def test_engine_chunking_is_invariant():
    """Chunk size is a memory knob, not a result knob — small/large chunks identical."""
    sim = _ToyGaussian(3)
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: x - th, d_theta=3)
    grid = [(0.0, 0.0, 0.0), (1.0, -1.0, 0.5)]
    a = evaluate_coverage(proc, sim, grid, [0.5, 0.9], n_per_theta=4000, chunk_size=64, seed=7)
    b = evaluate_coverage(proc, sim, grid, [0.5, 0.9], n_per_theta=4000, chunk_size=99999, seed=7)
    # identical X (same seed) + chunk-invariant statistic ⇒ identical coverage numbers
    assert np.allclose(a["coverage"]["empirical"].to_numpy(),
                       b["coverage"]["empirical"].to_numpy(), atol=1e-12)
    assert abs(a["coverage_error_max"] - b["coverage_error_max"]) < 1e-12
    assert abs(a["joint_chi2_ks_max"] - b["joint_chi2_ks_max"]) < 1e-12
