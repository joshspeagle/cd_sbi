"""Hello-world synthetic-driver figure. Exercises the F0 render pipeline.

Carries no scientific content; it is the placeholder that proves
style + builder contract + render CLI + gallery work end-to-end before
any real figure is authored in F1-F3.
"""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures.manifest import FigureSpec


def render(spec: FigureSpec):
    """Return a styled single-panel Figure. Pure: no disk IO."""
    style.apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=style.SIZES["single_column"])
    x = np.linspace(0, 2 * np.pi, 200)
    ax.plot(x, np.sin(x), color=style.METHOD_STYLE["cd_sbi"]["color"])
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\sin(\theta)$")
    ax.set_title("CD-SBI figure pipeline — hello world")
    fig.tight_layout()
    return fig
