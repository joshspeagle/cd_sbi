"""E2 — §8.1 cross-method coverage curves (all 5 methods, medium budget)."""
from __future__ import annotations

import glob
from pathlib import Path

from cdsbi.analysis.figures import style, panels
from cdsbi.analysis.figures.manifest import FigureSpec
from cdsbi.analysis.figures.data_io.figure_data import load_coverage_curve


def _method_of(run_dir: str) -> str | None:
    name = Path(run_dir).name
    for token in name.split(","):
        if token.startswith("method="):
            return token.split("=", 1)[1]
    return None


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    by_method = {}
    for root in spec.source_runs:
        for rd in glob.glob(str(Path(root) / "*")):
            m = _method_of(rd)
            if m and "budget=medium" in Path(rd).name and "seed=0" in Path(rd).name:
                by_method.setdefault(m, rd)

    fig, ax = plt.subplots(figsize=style.SIZES["single_column"])
    panels.diagonal_reference(ax, lo=0.5, hi=1.0)
    for m in style.CANONICAL_ORDER:
        if m not in by_method:
            continue
        nominal, empirical = load_coverage_curve(by_method[m])
        st = style.METHOD_STYLE[m]
        panels.coverage_curve(ax, nominal, empirical, color=st["color"],
                              marker=st["marker"], label=st["label"])
    ax.legend(fontsize=7)
    ax.set_title("§8.1 coverage by method")
    fig.tight_layout()
    return fig
