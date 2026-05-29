"""Smoke tests for the C1–C6 conceptual figure builders (synthetic drivers)."""
from __future__ import annotations

import matplotlib

from cdsbi.analysis.figures.manifest import FigureSpec


def _spec(**kw):
    base = dict(id="x", description="d", section="theory", builder="m:render",
                output_pdf="figures/x.pdf", output_png="figures/x.png",
                source_runs=[], checkpoint_runs=[])
    base.update(kw)
    return FigureSpec(**base)


def test_c1_two_panels_with_normal_overlay():
    from cdsbi.analysis.figures.figures.c1_pivot_picture import render
    fig = render(_spec())
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2
    assert len(fig.axes[1].lines) >= 1


def test_c2_single_panel_marks_two_roots():
    from cdsbi.analysis.figures.figures.c2_r1_failure import render
    fig = render(_spec())
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) >= 3


def test_c3_single_panel_with_folded_region():
    from cdsbi.analysis.figures.figures.c3_r2_folding import render
    fig = render(_spec())
    assert len(fig.axes) == 1
    assert len(fig.axes[0].collections) >= 1


def test_c4_loss_curve_with_floor():
    from cdsbi.analysis.figures.figures.c4_entropy_floor import render
    fig = render(_spec())
    ax = fig.axes[0]
    assert len(ax.lines) >= 3
    assert float(min(ax.lines[0].get_ydata())) >= 0.99


def test_c5_two_triangular_heatmaps():
    from cdsbi.analysis.figures.figures.c5_kr_structure import render
    fig = render(_spec())
    assert len(fig.axes) == 2
    assert all(len(ax.images) == 1 for ax in fig.axes)


def test_c6_renders_one_table():
    from cdsbi.analysis.figures.figures.c6_sbi_position import render
    fig = render(_spec())
    ax = fig.axes[0]
    assert len(ax.tables) == 1
    assert ax.axison is False
