"""E9 — budget saturation: coverage_error_max vs parameter budget per method (§8.2)."""
from __future__ import annotations

from cdsbi.analysis.figures import style, panels
from cdsbi.analysis.figures.manifest import FigureSpec
from cdsbi.analysis.figures.data_io.aggregates import load_aggregates


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt
    df = load_aggregates(spec.source_runs)
    series = {}
    for m, g in df.groupby("method"):
        gg = (g.groupby("budget_name")
                .agg(p=("actual_params_total", "mean"),
                     c=("coverage_error_max", "mean"))
                .sort_values("p"))
        series[m] = (gg["p"].to_numpy(), gg["c"].to_numpy())
    fig, ax = plt.subplots(figsize=style.SIZES["single_column"])
    panels.metric_vs_budget(ax, series)
    ax.legend(fontsize=7)
    ax.set_title("§8.2 budget saturation")
    fig.tight_layout()
    return fig
