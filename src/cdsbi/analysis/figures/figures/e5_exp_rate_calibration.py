"""E5 — §8.4 doubly-monotone calibration: PIT + coverage + final-loss-vs-floor."""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures import style, panels
from cdsbi.analysis.figures.manifest import FigureSpec
from cdsbi.analysis.figures.data_io.figure_data import (
    load_pit_values, load_coverage_curve, load_loss_tail_mean,
)
from cdsbi.simulators.exp_rate import ExponentialRate


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    run = spec.source_runs[0]
    u = load_pit_values(run)
    nominal, empirical = load_coverage_curve(run)
    loss = load_loss_tail_mean(run)
    entropy_floor = float(ExponentialRate().entropy_lower_bound())

    fig, (ax_pit, ax_cov, ax_loss) = plt.subplots(1, 3, figsize=(10.5, 3.0))
    panels.pit_histogram(ax_pit, u, bins=20); ax_pit.set_title("Marginal PIT")
    panels.diagonal_reference(ax_cov, lo=float(nominal.min()), hi=1.0)
    panels.coverage_curve(ax_cov, nominal, empirical,
                          color=style.METHOD_STYLE["cd_sbi"]["color"], marker="o")
    ax_cov.set_title("Coverage")
    panels.loss_bar_with_floor(ax_loss, {"CD-SBI": np.array([loss])},
                               floor=entropy_floor, group_labels=["medium"])
    ax_loss.set_title("Final loss vs entropy floor")
    fig.tight_layout()
    return fig
