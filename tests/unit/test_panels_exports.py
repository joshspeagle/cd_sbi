"""All panels are importable from the panels package root."""
from __future__ import annotations


def test_all_panels_re_exported():
    from cdsbi.analysis.figures import panels
    for name in (
        "noise_floor_band", "diagonal_reference",
        "pit_histogram", "coverage_curve", "coverage_tile",
        "boxplot_per_method", "cross_method_summary_log_y", "metric_vs_budget",
        "loss_trajectory_with_floor", "loss_bar_with_floor",
        "jacobian_recovery_scatter",
        "position_table_as_axes",
    ):
        assert callable(getattr(panels, name)), f"{name} not re-exported"
