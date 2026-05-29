"""C3 — (R2) failure: a pivot folded in X (∂r/∂X < 0 region) ⇒ Z(θ) > 1."""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures.manifest import FigureSpec


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    x = np.linspace(-3.0, 3.0, 400)
    r = x - 1.6 * np.sin(x)
    folded = (1.0 - 1.6 * np.cos(x)) < 0.0

    fig, ax = plt.subplots(figsize=style.SIZES["single_column"])
    ax.plot(x, r, color=style.METHOD_STYLE["cd_sbi"]["color"], label=r"$r(\theta; X)$")
    ax.fill_between(x, r.min(), r.max(), where=folded, color="#e8743b", alpha=0.2,
                    label=r"$\partial r/\partial X < 0$ (folded)")
    ax.set_xlabel("$X$"); ax.set_ylabel(r"$r(\theta; X)$")
    ax.set_title(r"(R2) violated: folding $\Rightarrow Z(\theta) > 1$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
