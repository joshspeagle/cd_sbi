"""Shared reference primitives: noise_floor_band, diagonal_reference."""
from __future__ import annotations


def _ax():
    from cdsbi.analysis.figures import style
    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    return ax


def test_noise_floor_band_scalar_draws_one_line():
    from cdsbi.analysis.figures.panels.primitives import noise_floor_band
    ax = _ax()
    out = noise_floor_band(ax, 0.99)
    assert out is ax
    assert len(ax.lines) == 1
    assert len(ax.patches) == 0
    # The line is horizontal at y=0.99.
    ydata = ax.lines[0].get_ydata()
    assert ydata[0] == 0.99 and ydata[1] == 0.99


def test_noise_floor_band_halfwidth_draws_one_patch():
    from cdsbi.analysis.figures.panels.primitives import noise_floor_band
    ax = _ax()
    noise_floor_band(ax, 0.5, halfwidth=0.02)
    assert len(ax.patches) == 1
    assert len(ax.lines) == 0


def test_diagonal_reference_draws_unit_slope_line():
    from cdsbi.analysis.figures.panels.primitives import diagonal_reference
    ax = _ax()
    out = diagonal_reference(ax, lo=0.0, hi=1.0)
    assert out is ax
    assert len(ax.lines) == 1
    line = ax.lines[0]
    assert list(line.get_xdata()) == [0.0, 1.0]
    assert list(line.get_ydata()) == [0.0, 1.0]
