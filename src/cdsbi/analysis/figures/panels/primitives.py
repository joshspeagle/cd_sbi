"""Shared reference-line primitives reused across panels.

Axes-level helpers: take an Axes, draw a reference, return the Axes.
"""
from __future__ import annotations


def noise_floor_band(ax, floor, *, halfwidth=None, label="noise floor", color="0.5"):
    """Draw a Monte-Carlo noise floor on `ax`.

    Scalar floor -> a dashed horizontal line at y=floor.
    With `halfwidth` -> a shaded horizontal band [floor-halfwidth, floor+halfwidth].
    """
    if halfwidth is None:
        ax.axhline(floor, ls="--", lw=1.0, color=color, label=label)
    else:
        ax.axhspan(floor - halfwidth, floor + halfwidth, color=color, alpha=0.15, label=label)
    return ax


def diagonal_reference(ax, *, lo=0.0, hi=1.0, label=None, color="0.5"):
    """Draw the y=x identity line from (lo, lo) to (hi, hi)."""
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=color, label=label)
    return ax
