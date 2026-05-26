"""Tests for cdsbi.analysis.paper_tables."""
import pandas as pd

from cdsbi.analysis.paper_tables import paper_table_8_1


def test_paper_table_8_1_pivots_method_x_budget():
    """Test that paper_table_8_1 pivots by (method, budget_name) and computes mean/std."""
    df = pd.DataFrame(
        [
            {
                "method": "cd_sbi",
                "budget_name": "medium",
                "seed": 0,
                "coverage_error_max": 0.01,
                "marginal_ks": 0.008,
                "pivot_rmse": 0.03,
            },
            {
                "method": "cd_sbi",
                "budget_name": "medium",
                "seed": 1,
                "coverage_error_max": 0.02,
                "marginal_ks": 0.009,
                "pivot_rmse": 0.04,
            },
            {
                "method": "npe",
                "budget_name": "medium",
                "seed": 0,
                "coverage_error_max": 0.05,
                "marginal_ks": None,
                "pivot_rmse": None,
            },
        ]
    )
    out = paper_table_8_1(df)
    assert ("cd_sbi", "medium") in out.index
    assert "coverage_error_max_mean" in out.columns
