"""CauchyLocScale simulator (hardening item 5).

The no-sufficient-statistic stress target, promoted from throwaway probe script
to committed simulator. Acceptance: protocol compliance, closed-form log_prob
against scipy, EXACT calibration of the oracle pivot (per-coord N(0,1), joint
χ²₂, independence), and the oracle hitting the measured coverage floor through
the actual evaluation engine.
"""
import math

import numpy as np
import torch
from scipy import stats

from cdsbi.diagnostics.engine import evaluate_coverage
from cdsbi.simulators.cauchy_loc_scale import CauchyLocScale


def test_shapes_and_protocol():
    sim = CauchyLocScale()
    assert sim.d_theta == 2 and sim.d_x == 10
    rng = np.random.default_rng(0)
    theta, x = sim.sample(32, rng)
    assert theta.shape == (32, 2) and x.shape == (32, 10)
    lo, hi = sim.theta_range
    assert lo <= sim.theta_lower[0] and hi >= sim.theta_upper[1]
    x0 = sim.sample_x_given_theta((0.5, -1.0), 16, np.random.default_rng(1))
    assert x0.shape == (16, 10)


def test_sampling_matches_declared_law():
    """Quantiles of (X − x0)/γ must match the standard Cauchy: median 0,
    quartiles ±1 (F_C(±1) = 1/2 ± 1/4)."""
    sim = CauchyLocScale()
    theta0 = (0.7, 2.0)  # gamma = e^0.7, loc = 2
    x = sim.sample_x_given_theta(theta0, 200_000, np.random.default_rng(2))
    z = (x - 2.0) / math.exp(0.7)
    q25, q50, q75 = np.quantile(z.numpy().ravel(), [0.25, 0.5, 0.75])
    assert abs(q50) < 0.02 and abs(q25 + 1.0) < 0.02 and abs(q75 - 1.0) < 0.02


def test_log_prob_matches_scipy():
    sim = CauchyLocScale()
    rng = np.random.default_rng(3)
    theta, x = sim.sample(64, rng)
    got = sim.log_prob(x, theta).numpy()
    gamma = np.exp(theta[:, 0].numpy())
    loc = theta[:, 1].numpy()
    want = np.array([
        stats.cauchy.logpdf(x[i].numpy(), loc=loc[i], scale=gamma[i]).sum()
        for i in range(64)
    ])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_oracle_pivot_exactly_calibrated_per_theta0():
    """r_star(θ0; X)|θ0 ~ N(0, I₂) EXACTLY, including at the box corners.
    n=20000 ⇒ KS 1% critical ≈ 0.0115; bound at 0.02."""
    sim = CauchyLocScale()
    corners = [(-1.0, -3.0), (0.0, 0.0), (1.0, 3.0)]
    for k, theta0 in enumerate(corners):
        x = sim.sample_x_given_theta(theta0, 20_000, np.random.default_rng(10 + k))
        th = torch.tensor([theta0]).expand(20_000, 2)
        r = sim.r_star(th, x).numpy()
        for j in range(2):
            ks = stats.kstest(r[:, j], "norm").statistic
            assert ks < 0.02, f"coord {j} KS={ks:.4f} at θ0={theta0}"
        ks2 = stats.kstest((r ** 2).sum(axis=1), stats.chi2(df=2).cdf).statistic
        assert ks2 < 0.02, f"‖r‖² χ²₂ KS={ks2:.4f} at θ0={theta0}"
        rho = np.corrcoef(r[:, 0], r[:, 1])[0, 1]
        assert abs(rho) < 0.03, f"pivot coords correlated (ρ={rho:.4f}) at θ0={theta0}"


def test_oracle_pivot_orthonormal_for_odd_n_iid():
    """The Gram-Schmidt contrast keeps the two pivot coords independent for
    ODD n_iid too (plain alternation would not be)."""
    sim = CauchyLocScale(n_iid=7)
    x = sim.sample_x_given_theta((0.0, 0.0), 20_000, np.random.default_rng(4))
    th = torch.zeros(20_000, 2)
    r = sim.r_star(th, x).numpy()
    rho = np.corrcoef(r[:, 0], r[:, 1])[0, 1]
    assert abs(rho) < 0.03
    for j in range(2):
        assert stats.kstest(r[:, j], "norm").statistic < 0.02


def test_oracle_at_floor_through_engine():
    """The acceptance criterion: the oracle pivot, run through the ACTUAL
    evaluation engine over a 3×3 box grid, sits at the Monte-Carlo floor.
    36 cells at n_per_theta=4000 ⇒ expected max|Δ| ≈ 2.8·SE ≈ 0.022; bound 0.03."""
    sim = CauchyLocScale()

    class _OraclePivot:
        d_theta = 2

        def pivot(self, theta_rows, x):
            return sim.r_star(theta_rows, x)

    grid = [(lg, lc) for lg in (-1.0, 0.0, 1.0) for lc in (-3.0, 0.0, 3.0)]
    out = evaluate_coverage(
        _OraclePivot(), sim, grid, [0.5, 0.68, 0.9, 0.95],
        n_per_theta=4000, seed=7,
    )
    assert out["coverage_error_max"] < 0.03, (
        f"oracle floor violated: {out['coverage_error_max']:.4f}"
    )
