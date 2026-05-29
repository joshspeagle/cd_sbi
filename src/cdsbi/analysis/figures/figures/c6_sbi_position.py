"""C6 — position in the SBI literature: target, single-stage, coverage."""
from __future__ import annotations

from cdsbi.analysis.figures import style, panels
from cdsbi.analysis.figures.manifest import FigureSpec

COLUMNS = ["Target", "Single-stage", "Frequentist\ncoverage", r"Finite-$d$ guar."]
ROWS = ["CD-SBI", "NPE", "NLE", "NRE", "LF2I"]
CELLS = [
    ["confidence dist.", "yes", "yes", "yes"],
    ["posterior", "yes", "no", "no"],
    ["likelihood", "yes", "no", "no"],
    ["likelihood ratio", "yes", "no", "no"],
    ["confidence set", "no", "yes", "partial"],
]


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=style.SIZES["double_column"])
    panels.position_table_as_axes(ax, COLUMNS, ROWS, CELLS)
    ax.set_title("CD-SBI vs other SBI methods")
    fig.tight_layout()
    return fig
