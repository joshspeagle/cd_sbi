"""Jacobian recovery panel: trained-vs-truth element scatter."""
from __future__ import annotations

import numpy as np


def _ax():
    from cdsbi.analysis.figures import style
    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    return ax


def test_jacobian_recovery_scatter_one_collection_plus_diagonal():
    from cdsbi.analysis.figures.panels.recovery import jacobian_recovery_scatter
    ax = _ax()
    truth = np.array([[1.0, 0.0], [-0.5, 0.8]])
    trained = truth + np.array([[0.01, 0.0], [-0.02, 0.015]])
    out = jacobian_recovery_scatter(ax, trained, truth)
    assert out is ax
    assert len(ax.collections) == 1         # the scatter PathCollection
    assert len(ax.lines) == 1               # the y=x reference


def test_jacobian_recovery_scatter_plots_all_elements():
    from cdsbi.analysis.figures.panels.recovery import jacobian_recovery_scatter
    ax = _ax()
    truth = np.zeros((3, 3))
    trained = np.zeros((3, 3))
    jacobian_recovery_scatter(ax, trained, truth)
    offsets = ax.collections[0].get_offsets()
    assert offsets.shape[0] == 9            # all 3x3 elements scattered
