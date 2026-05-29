"""E8 — headline cross-method summary: coverage_error_max (log-y) across experiments."""
from __future__ import annotations

from cdsbi.analysis.figures import style, panels
from cdsbi.analysis.figures.manifest import FigureSpec
from cdsbi.analysis.figures.data_io.figure_data import load_sweep

LABELS = ["8.1", "8.2", "8.3", "8.4"]   # aligned to spec.source_runs order


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt
    labels = LABELS[: len(spec.source_runs)]
    data = {}
    for label, root in zip(labels, spec.source_runs):
        df = load_sweep(root)
        if df.empty:
            continue
        med = df[df["budget_name"] == "medium"]
        for m, v in med.groupby("method")["coverage_error_max"].mean().items():
            data.setdefault(m, {})[label] = float(v)
    # cross_method_summary_log_y needs every method to have every label; keep
    # only methods present in all labels (drop any with a gap).
    data = {m: d for m, d in data.items() if all(l in d for l in labels)}
    fig, ax = plt.subplots(figsize=style.SIZES["double_column"])
    panels.cross_method_summary_log_y(ax, data, floor=0.02)
    ax.legend(fontsize=7)
    ax.set_title("Worst-case coverage error by method")
    fig.tight_layout()
    return fig
