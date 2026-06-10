"""Hardening item 6: eval upgrade — fused interior+edge grid, richer coverage
metrics, and the measured oracle floor row."""
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from cdsbi.diagnostics.engine import evaluate_coverage
from cdsbi.experiments.run import _eval_theta_grid, _oracle_floor_metrics
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def _cfg(interior, edge=None):
    exp = {"eval_thetas_interior": interior}
    if edge is not None:
        exp["eval_thetas_edge"] = edge
    return OmegaConf.create({"experiment": exp})


def test_grid_folds_edges_1d():
    grid = _eval_theta_grid(_cfg([-5.0, -3.0, 0.0, 3.0, 5.0], edge=[-7.0, 7.0]))
    assert grid == [-5.0, -3.0, 0.0, 3.0, 5.0, -7.0, 7.0]


def test_grid_dedupes_and_preserves_order():
    grid = _eval_theta_grid(_cfg([0.0, 3.0], edge=[3.0, 7.0]))
    assert grid == [0.0, 3.0, 7.0]


def test_grid_2d_lists():
    grid = _eval_theta_grid(_cfg([[0.0, 0.0], [3.0, 0.0]], edge=[[7.0, 7.0], [0.0, 0.0]]))
    assert grid == [[0.0, 0.0], [3.0, 0.0], [7.0, 7.0]]


def test_grid_missing_or_empty_edge():
    assert _eval_theta_grid(_cfg([1.0, 2.0])) == [1.0, 2.0]
    assert _eval_theta_grid(_cfg([1.0, 2.0], edge=[])) == [1.0, 2.0]


def test_engine_summary_metrics_match_table():
    """mean/p90/max must equal a pandas recomputation from the per-(θ₀,α)
    coverage table — an exact identity, not a statistical check."""
    sim = LocationNormal1D()

    class _Oracle:
        d_theta = 1

        def pivot(self, theta_rows, x):
            return sim.r_star(theta_rows, x)

    out = evaluate_coverage(_Oracle(), sim, [-5.0, 0.0, 5.0], [0.5, 0.9],
                            n_per_theta=1500, seed=0)
    err = (out["coverage"]["empirical"] - out["coverage"]["nominal"]).abs()
    assert out["coverage_error_max"] == float(err.max())
    assert out["coverage_error_mean"] == float(err.mean())
    assert out["coverage_error_p90"] == float(err.quantile(0.9))
    assert out["coverage_error_mean"] <= out["coverage_error_p90"] <= out["coverage_error_max"]


def test_oracle_floor_metrics_at_floor():
    """The measured floor row: LocationNormal1D's exact pivot over an
    interior+edge grid must sit at the MC floor (5 pts × 4 levels at n=4000
    ⇒ expected max ≈ 2.6·SE ≈ 0.02; bound 0.03)."""
    sim = LocationNormal1D()
    df, metrics = _oracle_floor_metrics(
        sim, [-7.0, -3.0, 0.0, 3.0, 7.0], [0.5, 0.68, 0.9, 0.95],
        n_per_theta=4000, seed=11,
    )
    assert isinstance(df, pd.DataFrame) and len(df) == 20
    assert metrics["oracle_coverage_error_max"] < 0.03
    assert metrics["oracle_coverage_error_mean"] <= metrics["oracle_coverage_error_max"]


def test_oracle_floor_absent_without_r_star():
    class _NoOracle:
        d_theta = 1

    df, metrics = _oracle_floor_metrics(_NoOracle(), [0.0], [0.9], 100, seed=0)
    assert df is None and metrics == {}
