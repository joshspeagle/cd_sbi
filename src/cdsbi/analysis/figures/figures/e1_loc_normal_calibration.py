"""E1 — §8.1 CDSBI calibration: marginal PIT histogram + coverage curve."""
from __future__ import annotations

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures.manifest import FigureSpec
from cdsbi.analysis.figures import panels
from cdsbi.analysis.figures.data_io.figure_data import (
    load_pit_values, load_coverage_curve,
)


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    run = spec.source_runs[0]
    u = load_pit_values(run)
    nominal, empirical = load_coverage_curve(run)

    fig, (ax_pit, ax_cov) = plt.subplots(1, 2, figsize=style.SIZES["double_column"])
    panels.pit_histogram(ax_pit, u, bins=20)
    ax_pit.set_title("Marginal PIT")
    panels.diagonal_reference(ax_cov, lo=float(nominal.min()), hi=1.0)
    panels.coverage_curve(ax_cov, nominal, empirical,
                          color=style.METHOD_STYLE["cd_sbi"]["color"], marker="o",
                          label="CD-SBI")
    ax_cov.set_title("Coverage")
    fig.tight_layout()
    return fig
