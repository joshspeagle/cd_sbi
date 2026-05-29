"""E10 — per-experiment coverage_error_max boxplots (4 panels)."""
from __future__ import annotations

from cdsbi.analysis.figures import style, panels
from cdsbi.analysis.figures.manifest import FigureSpec
from cdsbi.analysis.figures.data_io.aggregates import load_aggregates

LABELS = ["§8.1", "§8.2", "§8.3", "§8.4"]


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
    axes = axes.ravel()
    for ax, label, root in zip(axes, LABELS, spec.source_runs):
        df = load_aggregates([root])
        data = {m: g["coverage_error_max"].to_numpy() for m, g in df.groupby("method")}
        panels.boxplot_per_method(ax, data)
        ax.set_title(label)
    fig.tight_layout()
    return fig
