"""Hello-world figure: pure builder returning a styled Figure, no disk IO."""
from __future__ import annotations

import matplotlib


def _make_spec():
    from cdsbi.analysis.figures.manifest import FigureSpec
    return FigureSpec(
        id="_hello", description="hello", section="infra",
        builder="cdsbi.analysis.figures.figures._hello:render",
        output_pdf="figures/_hello.pdf", output_png="figures/_hello.png",
    )


def test_render_returns_figure_with_one_axes():
    from cdsbi.analysis.figures.figures._hello import render
    fig = render(_make_spec())
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1


def test_render_uses_agg_backend():
    from cdsbi.analysis.figures.figures import _hello
    _hello.render(_make_spec())
    assert matplotlib.get_backend().lower() == "agg"


def test_render_plots_one_line():
    from cdsbi.analysis.figures.figures._hello import render
    fig = render(_make_spec())
    ax = fig.axes[0]
    assert len(ax.lines) == 1
