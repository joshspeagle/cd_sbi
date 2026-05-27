"""paper_table_8_2: aggregates index_row.parquet + joint_mahalanobis.parquet
across (method, budget) into a single (mean, std) table."""
from __future__ import annotations

import pandas as pd

from cdsbi.analysis.paper_tables import paper_table_8_2


def test_paper_table_8_2_aggregates_metrics_with_joint_mahalanobis():
    df = pd.DataFrame([
        {"method": "cd_sbi", "budget_name": "medium", "seed": 0,
         "pivot_rmse": 0.04, "marginal_ks": 0.006, "coverage_error_max": 0.01,
         "joint_mahal_ks": 0.012, "actual_params_total": 4998},
        {"method": "cd_sbi", "budget_name": "medium", "seed": 1,
         "pivot_rmse": 0.05, "marginal_ks": 0.007, "coverage_error_max": 0.012,
         "joint_mahal_ks": 0.014, "actual_params_total": 4998},
        {"method": "npe", "budget_name": "medium", "seed": 0,
         "pivot_rmse": None, "marginal_ks": None, "coverage_error_max": 0.02,
         "joint_mahal_ks": None, "actual_params_total": 4740},
    ])
    out = paper_table_8_2(df)
    cd_row = out.loc[("cd_sbi", "medium")]
    assert abs(cd_row["pivot_rmse_mean"] - 0.045) < 1e-9
    assert abs(cd_row["joint_mahal_ks_mean"] - 0.013) < 1e-9
