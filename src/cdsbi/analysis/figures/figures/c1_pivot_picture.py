"""C1 — the pivot picture: every X | θ₀ collapses to N(0,1) under r* = θ₀ − X."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures.manifest import FigureSpec

THETAS = [-2.0, 0.0, 2.0]
N = 4000


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, (ax_x, ax_r) = plt.subplots(1, 2, figsize=style.SIZES["double_column"])
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(THETAS)))
    for t, c in zip(THETAS, colors):
        x = rng.normal(t, 1.0, N)
        ax_x.hist(x, bins=40, density=True, alpha=0.5, color=c,
                  label=fr"$\theta_0={t:g}$")
        ax_r.hist(t - x, bins=40, density=True, alpha=0.5, color=c)
    ax_x.set_title(r"Data $X \mid \theta_0$")
    ax_x.set_xlabel("$X$"); ax_x.set_ylabel("density"); ax_x.legend(fontsize=7)
    grid = np.linspace(-4, 4, 200)
    ax_r.plot(grid, norm.pdf(grid), color="k", lw=1.5, label=r"$\mathcal{N}(0,1)$")
    ax_r.set_title(r"Pivot $r^*(\theta_0; X) = \theta_0 - X$")
    ax_r.set_xlabel("$r$"); ax_r.set_ylabel("density"); ax_r.legend(fontsize=7)
    fig.tight_layout()
    return fig
