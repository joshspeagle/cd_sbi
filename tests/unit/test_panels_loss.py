"""Training-loss panels: loss_trajectory_with_floor, loss_bar_with_floor."""
from __future__ import annotations

import numpy as np


def _ax():
    from cdsbi.analysis.figures import style
    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    return ax


def test_loss_trajectory_one_line_per_series_plus_floor():
    from cdsbi.analysis.figures.panels.loss import loss_trajectory_with_floor
    ax = _ax()
    trajectories = {
        "R1+R2": np.linspace(1.4, 0.99, 50),
        "R1 only": np.linspace(1.4, 1.40, 50),
    }
    out = loss_trajectory_with_floor(ax, trajectories, floor=0.99)
    assert out is ax
    assert len(ax.lines) == 3               # 2 trajectories + 1 floor line


def test_loss_trajectory_without_floor_has_no_extra_line():
    from cdsbi.analysis.figures.panels.loss import loss_trajectory_with_floor
    ax = _ax()
    out = loss_trajectory_with_floor(ax, {"a": np.array([1.0, 0.9])})
    assert len(ax.lines) == 1


def test_loss_bar_with_floor_one_bar_per_series_group():
    from cdsbi.analysis.figures.panels.loss import loss_bar_with_floor
    ax = _ax()
    bars = {
        "R1+R2": np.array([0.986, 0.985, 0.983, 0.983]),
        "R1 only": np.array([1.419, 1.406, 1.391, 1.372]),
    }
    out = loss_bar_with_floor(ax, bars, floor=0.99,
                              group_labels=["S", "M", "L", "XL"])
    assert out is ax
    assert len(ax.patches) == 8             # 2 series x 4 budget groups
    assert len(ax.lines) == 1               # floor line
