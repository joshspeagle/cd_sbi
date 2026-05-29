"""C5 — KR structure: r_k depends on θ_{1:k} AND X_{1:k} (both lower-triangular)."""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures.manifest import FigureSpec

D = 4


def _tril_panel(ax, var: str):
    mask = np.tril(np.ones((D, D)))
    ax.imshow(mask, cmap="Blues", vmin=0.0, vmax=1.5, aspect="equal")
    for i in range(D):
        for j in range(D):
            if mask[i, j]:
                ax.text(j, i, "✓", ha="center", va="center", color="white", fontsize=10)
    ax.set_xticks(range(D)); ax.set_xticklabels([fr"${var}_{{{j+1}}}$" for j in range(D)])
    ax.set_yticks(range(D)); ax.set_yticklabels([fr"$r_{{{i+1}}}$" for i in range(D)])


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    fig, (ax_t, ax_x) = plt.subplots(1, 2, figsize=style.SIZES["double_column"])
    _tril_panel(ax_t, r"\theta")
    ax_t.set_title(r"$\theta$-dependence: $r_k(\theta_{1:k})$")
    _tril_panel(ax_x, "X")
    ax_x.set_title(r"$X$-dependence: $r_k(X_{1:k})$")
    fig.suptitle(r"Triangular (Knothe--Rosenblatt) structure: "
                 r"$r_k = r_k(\theta_{1:k};\, X_{1:k})$")
    fig.tight_layout()
    return fig
