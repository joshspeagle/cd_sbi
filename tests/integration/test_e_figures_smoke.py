"""Smoke tests for the E1–E10 figure builders, driven by synthetic fixtures."""
from __future__ import annotations

import matplotlib
import numpy as np

from cdsbi.analysis.figures.manifest import FigureSpec
from tests.figures_fixtures import (
    make_run_dir, make_sweep, write_marginal_pit_raw, write_coverage,
    write_coverage_2d, write_jacobian_raw, write_joint_mahalanobis_raw,
)


def _spec(**kw):
    base = dict(id="x", description="d", section="8", builder="m:render",
                output_pdf="figures/x.pdf", output_png="figures/x.png",
                source_runs=[], checkpoint_runs=[])
    base.update(kw)
    return FigureSpec(**base)


def test_e1_returns_two_panel_figure(tmp_path):
    from cdsbi.analysis.figures.figures.e1_loc_normal_calibration import render
    rd = tmp_path / "8_1"; rd.mkdir()
    write_marginal_pit_raw(rd, np.random.default_rng(0).uniform(0, 1, 500))
    write_coverage(rd, [-3.0, 0.0, 3.0], [0.5, 0.68, 0.9, 0.95])
    fig = render(_spec(source_runs=[str(rd)]))
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2


def test_e2_returns_single_panel_with_method_lines(tmp_path):
    from cdsbi.analysis.figures.figures.e2_loc_normal_cross_method import render
    import pandas as pd
    root = tmp_path / "sweep"
    for method in ("cd_sbi", "npe", "nle"):
        rd = root / f"method={method},budget=medium,seed=0"
        rd.mkdir(parents=True)
        (rd / "STATUS").write_text("OK")
        write_coverage(rd, [-3.0, 0.0, 3.0], [0.5, 0.68, 0.9, 0.95])
        pd.DataFrame([{"method": method, "budget_name": "medium", "seed": 0}]).to_parquet(rd / "index_row.parquet")
    fig = render(_spec(source_runs=[str(root)], section="8.1"))
    ax = fig.axes[0]
    assert len(ax.lines) == 4   # 3 method curves + 1 diagonal


def test_e3_returns_two_panels(tmp_path):
    from cdsbi.analysis.figures.figures.e3_joint_diagnostics import render
    import numpy as np
    rd = tmp_path / "8_2"; rd.mkdir()
    rng = np.random.default_rng(0)
    write_joint_mahalanobis_raw(rd, {"[0.0, 0.0]": rng.chisquare(2, 400),
                                     "[2.0, -1.0]": rng.chisquare(2, 400)})
    write_coverage_2d(rd, [(-2.0, -2.0), (0.0, 0.0), (2.0, 2.0)], [0.5, 0.68, 0.9, 0.95])
    fig = render(_spec(source_runs=[str(rd)], section="8.2"))
    assert len(fig.axes) >= 2


def test_e4_returns_scatter_panel(tmp_path):
    from cdsbi.analysis.figures.figures.e4_jacobian_recovery import render
    import numpy as np
    rd = tmp_path / "8_3"; rd.mkdir()
    j_true = np.array([[1.0, 0.0], [-0.4, 0.9]])
    write_jacobian_raw(rd, j_true + 0.01, j_true)
    fig = render(_spec(source_runs=[str(rd)], section="8.3"))
    assert len(fig.axes) == 1
    assert len(fig.axes[0].collections) == 1


def test_e5_returns_three_panels(tmp_path):
    from cdsbi.analysis.figures.figures.e5_exp_rate_calibration import render
    import numpy as np
    rd = make_run_dir(tmp_path, method="cd_sbi", budget_name="medium", seed=0,
                      coverage_error_max=0.03, final_loss=0.985,
                      loss_history_tail=[0.99, 0.985, 0.985])
    write_marginal_pit_raw(rd, np.random.default_rng(0).uniform(0, 1, 500))
    write_coverage(rd, [0.5, 1.5, 2.5], [0.5, 0.68, 0.9, 0.95])
    fig = render(_spec(source_runs=[str(rd)], section="8.4"))
    assert len(fig.axes) == 3
