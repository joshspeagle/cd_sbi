"""E3 — §8.2 joint diagnostics: joint-Mahalanobis PIT + coverage-error tile."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from cdsbi.analysis.figures import style, panels
from cdsbi.analysis.figures.manifest import FigureSpec
from cdsbi.analysis.figures.data_io.figure_data import (
    load_joint_mahalanobis_sq, load_coverage_tile,
)


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    run = spec.source_runs[0]
    r_sq = load_joint_mahalanobis_sq(run)
    pooled = np.concatenate(list(r_sq.values()))
    pit = chi2(df=2).cdf(pooled)
    theta0, alpha, error = load_coverage_tile(run)

    fig, (ax_pit, ax_tile) = plt.subplots(1, 2, figsize=style.SIZES["double_column"])
    panels.pit_histogram(ax_pit, pit, bins=20)
    ax_pit.set_title("Joint Mahalanobis PIT")
    panels.coverage_tile(ax_tile, theta0, alpha, error)
    ax_tile.set_title("Coverage error")
    fig.tight_layout()
    return fig
