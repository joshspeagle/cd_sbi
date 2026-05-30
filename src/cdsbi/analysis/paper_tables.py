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


def paper_table_8_3(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot (method × budget) → seed-averaged §8.3 metrics.

    Extends paper_table_8_2 with jacobian_max_residual (the §8.3-specific
    KR-uniqueness empirical metric). Non-pivot methods will have NaN in
    this column; groupby.mean() drops them silently.
    """
    metrics = [
        "coverage_error_max", "marginal_ks", "pivot_rmse",
        "joint_mahal_ks", "jacobian_max_residual", "actual_params_total",
    ]
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby(["method", "budget_name"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg


def paper_table_8_4(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot (method × budget) → seed-averaged §8.4 metrics.

    §8.4 is 1D (target / T scalar) so JointMahalanobis no-ops (NaN) and
    JacobianRecovery skips (non-constant truth Jacobian). The load-bearing
    §8.4 column is final_loss — the doubly-monotone CDSBI flow's loss
    stays ABOVE the entropy lower bound (R1+R2 contract preserves
    Z(θ) ≡ 1), while the R1-only joint_umnn ablation flow's loss falls
    BELOW the floor (the §3.5 mechanism failure).
    """
    metrics = [
        "coverage_error_max", "marginal_ks", "pivot_rmse",
        "final_loss", "actual_params_total",
    ]
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby(["method", "budget_name"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg


def paper_table_mu_cov(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged (μ,Σ) Stage-A table."""
    metrics = ["coverage_error_max", "pivot_rmse", "joint_mahal_ks",
               "mmcd_cov1_ks", "mmcd_cov2_ks", "mmcd_cov3_ks", "mmcd_mu_hotelling_ks",
               "floor_margin", "final_loss", "actual_params_total"]
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby(["method", "budget_name"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg


def paper_table_mu_sigma(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged (μ,σ²) Stage-A table: coverage, pivot RMSE, joint Mahalanobis,
    marginal-CD recovery (σ² χ² + μ t), entropy-floor loss."""
    metrics = [
        "coverage_error_max", "pivot_rmse", "joint_mahal_ks",
        "marginal_cd_sigma_ks", "marginal_cd_mu_ks", "marginal_cd_mu_t_resid",
        "final_loss", "actual_params_total",
    ]
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby(["method", "budget_name"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg
