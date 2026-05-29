"""C2 — (R1) failure: a non-monotone pivot in θ gives a multi-valued CD inverse."""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures.manifest import FigureSpec


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    theta = np.linspace(-3.0, 3.0, 400)
    r = (theta - 0.7) ** 2 - 2.0
    level = 0.5
    roots = 0.7 + np.array([-1.0, 1.0]) * np.sqrt(level + 2.0)

    fig, ax = plt.subplots(figsize=style.SIZES["single_column"])
    ax.plot(theta, r, color=style.METHOD_STYLE["cd_sbi"]["color"],
            label=r"$r(\theta; X)$ (non-monotone)")
    ax.axhline(level, ls="--", color="0.5", label=r"$r = c$")
    ax.plot(roots, [level, level], "o", color="#e8743b", zorder=5,
            label="two solutions")
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$r(\theta; X)$")
    ax.set_title(r"(R1) violated: two $\theta$ map to one $r$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
