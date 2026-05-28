"""Table-as-axes panel: position_table_as_axes."""
from __future__ import annotations


def _ax():
    from cdsbi.analysis.figures import style
    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    return ax


def test_position_table_renders_one_table_with_axis_off():
    from cdsbi.analysis.figures.panels.tables import position_table_as_axes
    ax = _ax()
    columns = ["Target", "Single-stage", "Coverage by construction"]
    rows = ["CD-SBI", "NPE", "LF2I"]
    cells = [
        ["CD", "yes", "yes"],
        ["posterior", "yes", "no"],
        ["confidence set", "no", "yes"],
    ]
    out = position_table_as_axes(ax, columns, rows, cells)
    assert out is ax
    assert len(ax.tables) == 1
    assert ax.axison is False               # axis turned off for a clean table
