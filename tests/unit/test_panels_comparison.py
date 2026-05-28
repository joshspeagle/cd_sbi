"""Cross-method comparison panels: boxplot, log-y summary, metric-vs-budget."""
from __future__ import annotations

import numpy as np


def _ax():
    from cdsbi.analysis.figures import style
    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    return ax


def test_boxplot_per_method_one_box_per_method_in_canonical_order():
    from cdsbi.analysis.figures.panels.comparison import boxplot_per_method
    ax = _ax()
    data = {
        "npe": np.array([0.06, 0.07, 0.05]),
        "cd_sbi": np.array([0.025, 0.024, 0.026]),
        "nle": np.array([0.10, 0.11, 0.09]),
    }
    out = boxplot_per_method(ax, data)
    assert out is ax
    # One box per method present (3); x tick labels follow CANONICAL_ORDER filtered.
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["CD-SBI", "NLE", "NPE"]


def test_cross_method_summary_log_y_one_line_per_method_plus_floor():
    from cdsbi.analysis.figures.panels.comparison import cross_method_summary_log_y
    ax = _ax()
    data = {
        "cd_sbi": {"8.1": 0.025, "8.2": 0.025, "8.3": 0.025, "8.4": 0.031},
        "nle":    {"8.1": 0.025, "8.2": 0.08, "8.3": 0.11, "8.4": 0.21},
    }
    out = cross_method_summary_log_y(ax, data, floor=0.02)
    assert out is ax
    assert ax.get_yscale() == "log"
    assert len(ax.lines) == 3               # 2 method lines + 1 floor line


def test_metric_vs_budget_log_x_one_line_per_method():
    from cdsbi.analysis.figures.panels.comparison import metric_vs_budget
    ax = _ax()
    series = {
        "cd_sbi": (np.array([1000, 5000, 25000]), np.array([0.03, 0.025, 0.025])),
        "npe":    (np.array([1000, 5000, 25000]), np.array([0.07, 0.06, 0.05])),
    }
    out = metric_vs_budget(ax, series)
    assert out is ax
    assert ax.get_xscale() == "log"
    assert len(ax.lines) == 2
