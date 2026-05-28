"""Reusable Axes-level chart primitives (panels).

Every panel takes an Axes as its first argument, draws onto it, and returns it.
Callers must have applied the shared style (style.apply_style()) first.
"""
from cdsbi.analysis.figures.panels.calibration import (
    coverage_curve,
    coverage_tile,
    pit_histogram,
)
from cdsbi.analysis.figures.panels.comparison import (
    boxplot_per_method,
    cross_method_summary_log_y,
    metric_vs_budget,
)
from cdsbi.analysis.figures.panels.loss import (
    loss_bar_with_floor,
    loss_trajectory_with_floor,
)
from cdsbi.analysis.figures.panels.primitives import (
    diagonal_reference,
    noise_floor_band,
)
from cdsbi.analysis.figures.panels.recovery import jacobian_recovery_scatter
from cdsbi.analysis.figures.panels.tables import position_table_as_axes

__all__ = [
    "noise_floor_band",
    "diagonal_reference",
    "pit_histogram",
    "coverage_curve",
    "coverage_tile",
    "boxplot_per_method",
    "cross_method_summary_log_y",
    "metric_vs_budget",
    "loss_trajectory_with_floor",
    "loss_bar_with_floor",
    "jacobian_recovery_scatter",
    "position_table_as_axes",
]
