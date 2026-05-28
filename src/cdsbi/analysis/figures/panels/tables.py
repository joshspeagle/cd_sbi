"""Render a curated comparison table onto an Axes (C6 position figure)."""
from __future__ import annotations


def position_table_as_axes(ax, columns, rows, cells):
    """Draw a table on `ax` with the figure suite's styling.

    `columns`: list of column headers.
    `rows`: list of row labels.
    `cells`: list of rows, each a list of cell strings; shape len(rows) x len(columns).
    """
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        rowLabels=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    return ax
