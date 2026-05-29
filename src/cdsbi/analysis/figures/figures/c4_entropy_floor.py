"""C4 — NF-MLE strict propriety: loss ≥ conditional entropy, equality at truth."""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures import style, panels
from cdsbi.analysis.figures.manifest import FigureSpec

FLOOR = 1.0


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    delta = np.linspace(-2.0, 2.0, 400)
    loss = FLOOR + 0.5 * delta ** 2
    fig, ax = plt.subplots(figsize=style.SIZES["single_column"])
    ax.plot(delta, loss, color=style.METHOD_STYLE["cd_sbi"]["color"], label="NF-MLE loss")
    panels.noise_floor_band(ax, FLOOR,
                            label=r"entropy floor $\mathbb{E}_\rho[H(X\mid\theta)]$")
    ax.plot([0.0], [FLOOR], "o", color="#e8743b", zorder=5)
    ax.annotate("truth: loss = floor", xy=(0.0, FLOOR), xytext=(0.4, FLOOR + 0.6),
                arrowprops=dict(arrowstyle="->", color="#e8743b"), fontsize=7)
    ax.set_xlabel(r"density misfit $\delta$"); ax.set_ylabel("NF-MLE loss")
    ax.set_title(r"Strict propriety: loss $\geq$ floor")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
