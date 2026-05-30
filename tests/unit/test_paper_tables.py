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


def test_paper_table_8_3_includes_jacobian_column():
    import pandas as pd
    from cdsbi.analysis.paper_tables import paper_table_8_3
    df = pd.DataFrame([
        {"method": "cd_sbi", "budget_name": "medium", "coverage_error_max": 0.02,
         "marginal_ks": 0.013, "pivot_rmse": 0.27, "joint_mahal_ks": 0.012,
         "jacobian_max_residual": 0.03, "actual_params_total": 4752},
        {"method": "cd_sbi", "budget_name": "medium", "coverage_error_max": 0.022,
         "marginal_ks": 0.014, "pivot_rmse": 0.28, "joint_mahal_ks": 0.011,
         "jacobian_max_residual": 0.04, "actual_params_total": 4752},
        {"method": "npe", "budget_name": "medium", "coverage_error_max": 0.07,
         "marginal_ks": None, "pivot_rmse": None, "joint_mahal_ks": None,
         "jacobian_max_residual": None, "actual_params_total": 5064},
    ])
    t = paper_table_8_3(df)
    assert "jacobian_max_residual_mean" in t.columns
    assert "jacobian_max_residual_std" in t.columns
    # CDSBI row should have the mean of [0.03, 0.04] = 0.035
    assert abs(t.loc[("cd_sbi", "medium"), "jacobian_max_residual_mean"] - 0.035) < 1e-9


def test_paper_table_mu_sigma_aggregates():
    import pandas as pd
    from cdsbi.analysis.paper_tables import paper_table_mu_sigma
    df = pd.DataFrame([
        {"method": "cd_sbi", "budget_name": "medium", "coverage_error_max": 0.02,
         "pivot_rmse": 0.10, "joint_mahal_ks": 0.03, "marginal_cd_sigma_ks": 0.02,
         "marginal_cd_mu_ks": 0.03, "marginal_cd_mu_t_resid": 0.01,
         "final_loss": 0.99, "actual_params_total": 12000},
        {"method": "cd_sbi", "budget_name": "medium", "coverage_error_max": 0.03,
         "pivot_rmse": 0.11, "joint_mahal_ks": 0.04, "marginal_cd_sigma_ks": 0.03,
         "marginal_cd_mu_ks": 0.04, "marginal_cd_mu_t_resid": 0.012,
         "final_loss": 1.00, "actual_params_total": 12000},
    ])
    tbl = paper_table_mu_sigma(df)
    assert ("cd_sbi", "medium") in tbl.index
    assert "marginal_cd_mu_ks_mean" in tbl.columns
    assert abs(tbl.loc[("cd_sbi", "medium"), "coverage_error_max_mean"] - 0.025) < 1e-9


def test_paper_table_mu_cov_aggregates():
    import pandas as pd
    from cdsbi.analysis.paper_tables import paper_table_mu_cov
    df = pd.DataFrame([
        {"method": "cd_sbi", "budget_name": "medium", "coverage_error_max": 0.03,
         "pivot_rmse": 0.12, "mmcd_mu_hotelling_ks": 0.03, "final_loss": 2.0},
        {"method": "cd_sbi", "budget_name": "medium", "coverage_error_max": 0.04,
         "pivot_rmse": 0.14, "mmcd_mu_hotelling_ks": 0.04, "final_loss": 2.1},
    ])
    tbl = paper_table_mu_cov(df)
    assert ("cd_sbi", "medium") in tbl.index
    assert "mmcd_mu_hotelling_ks_mean" in tbl.columns
