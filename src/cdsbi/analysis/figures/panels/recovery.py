"""Jacobian recovery panel: trained E[dr/dtheta] vs closed-form L^-1."""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures.panels.primitives import diagonal_reference


def jacobian_recovery_scatter(ax, trained, truth, *, color=None):
    """Scatter trained Jacobian elements against truth (L^-1), with a y=x line.

    `trained` and `truth` are same-shape arrays of Jacobian-matrix elements;
    they are flattened and plotted element-wise. Points on the diagonal mean
    perfect recovery (Theorem A-d).
    """
    trained = np.asarray(trained).ravel()
    truth = np.asarray(truth).ravel()
    color = color or style.METHOD_STYLE["cd_sbi"]["color"]
    ax.scatter(truth, trained, color=color, s=30, zorder=3)
    lo = float(min(truth.min(), trained.min()))
    hi = float(max(truth.max(), trained.max()))
    diagonal_reference(ax, lo=lo, hi=hi, label="$y=x$")
    ax.set_xlabel(r"truth $L^{-1}$ element")
    ax.set_ylabel(r"trained $\mathbb{E}[\partial r/\partial\theta]$ element")
    return ax
