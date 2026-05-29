"""figure_data loaders: PIT values, jacobian matrices, coverage curve/tile, loss tail."""
from __future__ import annotations

import numpy as np

from tests.figures_fixtures import (
    make_run_dir, make_sweep, write_marginal_pit_raw, write_jacobian_raw,
    write_coverage, write_coverage_2d, write_joint_mahalanobis_raw,
)


def test_load_joint_mahalanobis_sq(tmp_path):
    from cdsbi.analysis.figures.data_io.figure_data import load_joint_mahalanobis_sq
    rd = tmp_path / "run"; rd.mkdir()
    rng = np.random.default_rng(0)
    r_sq = {"[0.0, 0.0]": rng.chisquare(2, 300), "[2.0, -1.0]": rng.chisquare(2, 300)}
    write_joint_mahalanobis_raw(rd, r_sq)
    out = load_joint_mahalanobis_sq(str(rd))
    assert set(out.keys()) == {"[0.0, 0.0]", "[2.0, -1.0]"}
    assert out["[0.0, 0.0]"].shape == (300,)
    assert (out["[0.0, 0.0]"] >= 0).all()


def test_load_pit_values(tmp_path):
    from cdsbi.analysis.figures.data_io.figure_data import load_pit_values
    rd = tmp_path / "run"; rd.mkdir()
    write_marginal_pit_raw(rd, np.linspace(0, 1, 200))
    u = load_pit_values(str(rd))
    assert u.shape == (200,)


def test_load_jacobian_matrices(tmp_path):
    from cdsbi.analysis.figures.data_io.figure_data import load_jacobian_matrices
    rd = tmp_path / "run"; rd.mkdir()
    j_true = np.array([[1.0, 0.0], [-0.4, 0.9]])
    write_jacobian_raw(rd, j_true + 0.01, j_true)
    j_emp, jt = load_jacobian_matrices(str(rd))
    assert j_emp.shape == (2, 2) and jt.shape == (2, 2)
    np.testing.assert_allclose(jt, j_true)


def test_load_coverage_curve(tmp_path):
    from cdsbi.analysis.figures.data_io.figure_data import load_coverage_curve
    rd = tmp_path / "run"; rd.mkdir()
    write_coverage(rd, theta0_list=[-2.0, 0.0, 2.0], alpha_list=[0.5, 0.68, 0.9, 0.95])
    nominal, empirical = load_coverage_curve(str(rd))
    assert nominal.shape == (4,) and empirical.shape == (4,)
    assert list(nominal) == [0.5, 0.68, 0.9, 0.95]


def test_load_coverage_tile_d1(tmp_path):
    from cdsbi.analysis.figures.data_io.figure_data import load_coverage_tile
    rd = tmp_path / "run"; rd.mkdir()
    write_coverage(rd, theta0_list=[-2.0, 0.0, 2.0], alpha_list=[0.5, 0.68, 0.9, 0.95])
    theta0, alpha, error = load_coverage_tile(str(rd))
    assert theta0.shape == (3,) and alpha.shape == (4,)
    assert error.shape == (3, 4)
    assert (error >= 0).all()
    assert list(theta0) == [-2.0, 0.0, 2.0]


def test_load_coverage_tile_d2_does_not_conflate_shared_first_coord(tmp_path):
    from cdsbi.analysis.figures.data_io.figure_data import load_coverage_tile
    rd = tmp_path / "run2"; rd.mkdir()
    grid = [(-3.0, -3.0), (-3.0, 0.0), (0.0, 0.0), (3.0, 0.0), (3.0, 3.0)]
    write_coverage_2d(rd, grid, alpha_list=[0.5, 0.68, 0.9, 0.95])
    theta0, alpha, error = load_coverage_tile(str(rd))
    assert error.shape == (5, 4)
    assert list(theta0) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_load_loss_tail_mean(tmp_path):
    from cdsbi.analysis.figures.data_io.figure_data import load_loss_tail_mean
    rd = make_run_dir(tmp_path, method="cd_sbi", budget_name="medium", seed=0,
                      coverage_error_max=0.03, final_loss=0.99,
                      loss_history_tail=[1.1, 1.0, 0.99, 0.99])
    assert abs(load_loss_tail_mean(str(rd)) - np.mean([1.1, 1.0, 0.99, 0.99])) < 1e-9


def test_load_sweep_dedups_and_handles_timestamp_layer(tmp_path):
    """load_sweep aggregates across timestamp dirs, dedups (method,budget,seed)
    keeping the latest, and also works on a flat sweep layout."""
    from cdsbi.analysis.figures.data_io.figure_data import load_sweep
    import pandas as pd
    sweep = tmp_path / "8_x_baseline_sweep"
    # two timestamp dirs; the later one re-runs cd_sbi/medium/seed0
    for ts, cov in (("2026-01-01_00-00-00", 0.20), ("2026-01-02_00-00-00", 0.05)):
        rd = sweep / ts / "method=cd_sbi,budget=medium,seed=0"
        rd.mkdir(parents=True)
        (rd / "STATUS").write_text("OK")
        pd.DataFrame([{"method": "cd_sbi", "budget_name": "medium", "seed": 0,
                       "coverage_error_max": cov}]).to_parquet(rd / "index_row.parquet")
    df = load_sweep(str(sweep))
    assert len(df) == 1                       # deduped to one row
    assert float(df["coverage_error_max"].iloc[0]) == 0.05   # kept the later re-run


def test_load_sweep_flat_layout(tmp_path):
    from cdsbi.analysis.figures.data_io.figure_data import load_sweep
    sweep = make_sweep(tmp_path / "flat")     # run-dirs directly under root
    df = load_sweep(str(sweep))
    assert set(df["method"]) == {"cd_sbi", "npe"}
    assert len(df) == 4
