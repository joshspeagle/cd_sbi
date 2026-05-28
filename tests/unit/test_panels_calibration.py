"""Calibration panels: pit_histogram, coverage_curve, coverage_tile."""
from __future__ import annotations

import numpy as np


def _ax():
    from cdsbi.analysis.figures import style
    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    return ax


def test_pit_histogram_draws_bins_patches_plus_uniform_line():
    from cdsbi.analysis.figures.panels.calibration import pit_histogram
    ax = _ax()
    rng = np.random.default_rng(0)
    out = pit_histogram(ax, rng.uniform(0, 1, size=500), bins=20)
    assert out is ax
    assert len(ax.patches) == 20            # one Rectangle per bin
    assert len(ax.lines) == 1               # uniform-density reference line
    assert ax.get_xlim() == (0.0, 1.0)


def test_pit_histogram_uses_cdsbi_navy_by_default():
    from cdsbi.analysis.figures.panels.calibration import pit_histogram
    from cdsbi.analysis.figures import style
    ax = _ax()
    pit_histogram(ax, np.linspace(0, 1, 100), bins=10)
    facecolor = ax.patches[0].get_facecolor()
    import matplotlib.colors as mcolors
    assert mcolors.to_hex(facecolor) == style.METHOD_STYLE["cd_sbi"]["color"]


def test_coverage_curve_draws_single_series():
    from cdsbi.analysis.figures.panels.calibration import coverage_curve
    ax = _ax()
    nominal = np.array([0.5, 0.68, 0.9, 0.95])
    empirical = np.array([0.49, 0.67, 0.91, 0.95])
    out = coverage_curve(ax, nominal, empirical, color="#1f2d5a", marker="o")
    assert out is ax
    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_xdata()) == list(nominal)


def test_coverage_tile_draws_one_quadmesh_and_a_colorbar():
    from cdsbi.analysis.figures.panels.calibration import coverage_tile
    ax = _ax()                              # fig starts with exactly 1 axes
    theta0 = np.array([-2.0, 0.0, 2.0])
    alpha = np.array([0.5, 0.68, 0.9, 0.95])
    error = np.abs(np.random.default_rng(1).normal(0, 0.02, size=(3, 4)))
    out = coverage_tile(ax, theta0, alpha, error)
    assert out is ax
    assert len(ax.collections) == 1         # the QuadMesh
    # The colorbar is required for a readable heatmap; it adds a second axes
    # to the parent figure. Assert it so TDD actually drives that requirement
    # (deleting the ax.figure.colorbar call must break this test).
    assert len(ax.figure.axes) == 2
