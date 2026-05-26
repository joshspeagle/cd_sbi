"""Paper-table generators for v0 §8.1 comparison."""
from __future__ import annotations

import pandas as pd


def paper_table_8_1(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot (method × budget) → seed-averaged diagnostics.

    Groups by (method, budget_name) and computes mean and std of available metrics:
    - coverage_error_max
    - marginal_ks
    - pivot_rmse
    - actual_params_total

    Returns DataFrame with multi-level columns (metric, statistic).
    """
    metrics = ["coverage_error_max", "marginal_ks", "pivot_rmse", "actual_params_total"]
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby(["method", "budget_name"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg
