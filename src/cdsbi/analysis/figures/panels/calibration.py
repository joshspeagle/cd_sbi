"""Calibration-diagnostic panels: PIT histogram, coverage curve, coverage tile."""
from __future__ import annotations

from cdsbi.analysis.figures import style


def pit_histogram(ax, pit_values, *, bins=20, color=None, label=None):
    """Histogram of PIT values on [0, 1] with a uniform-density reference line.

    A perfectly calibrated pivot has PIT ~ Uniform(0, 1), i.e. a flat histogram
    at density 1. `pit_values` is the raw 1-D array of PIT transforms.
    """
    color = color or style.METHOD_STYLE["cd_sbi"]["color"]
    ax.hist(pit_values, bins=bins, range=(0.0, 1.0), density=True,
            color=color, alpha=0.85, label=label)
    ax.axhline(1.0, ls="--", lw=1.0, color="0.5")
    ax.set_xlabel("PIT")
    ax.set_ylabel("density")
    ax.set_xlim(0.0, 1.0)
    return ax


def coverage_curve(ax, nominal, empirical, *, color=None, marker=None, label=None):
    """Plot one method's empirical coverage against nominal coverage.

    Single-series: F2 calls this once per method and draws the y=x reference
    separately (panels.primitives.diagonal_reference) so the diagonal appears once.
    """
    ax.plot(nominal, empirical, marker=marker, color=color, label=label)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    return ax


def coverage_tile(ax, theta0, alpha, error, *, cmap="magma"):
    """Heatmap of coverage error |empirical - nominal| over (theta_0 x alpha).

    `error` has shape (len(theta0), len(alpha)). Adds a colorbar to the parent
    figure (the one allowed figure-level touch — a heatmap is unreadable without it).
    """
    im = ax.pcolormesh(alpha, theta0, error, cmap=cmap, shading="auto")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\theta_0$")
    ax.figure.colorbar(im, ax=ax, label="coverage error")
    return ax
