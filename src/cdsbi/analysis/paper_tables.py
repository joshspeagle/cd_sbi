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


def paper_table_8_2(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot (method × budget) → seed-averaged §8.2 metrics.

    Same as paper_table_8_1 but adds joint_mahal_ks if present (lifted into
    index_row.parquet by analysis upstream, or computed per-run by callers).
    """
    metrics = [
        "coverage_error_max", "marginal_ks", "pivot_rmse",
        "joint_mahal_ks", "actual_params_total",
    ]
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby(["method", "budget_name"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg
