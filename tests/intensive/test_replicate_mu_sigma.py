"""Intensive: (μ,σ²) Stage-A replication — pivot recovers, calibrates, reaches floor."""
import glob
import os
from pathlib import Path
import subprocess

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.intensive
def test_replicate_mu_sigma_stage_a(tmp_path):
    """Run CDSBI on the (μ,σ²) target with SingleIndexMonotoneFlow across 5 seeds;
    seed-averaged metrics within tolerance bands.

    Tolerance bands:
      - pivot_rmse mean ≤ 0.30  (2-D pivot; acceptable for medium budget)
      - coverage_error_max mean ≤ 0.05  (noise floor ~0.02; 2.5× headroom)
      - marginal_cd_sigma_ks mean ≤ 0.06  (χ²-CD for σ²)
      - marginal_cd_mu_ks mean ≤ 0.06  (Student-t CD for μ, marginalized)
      - marginal_cd_mu_t_resid mean ≤ 0.05  (max-element residual vs analytic)
      - final_loss > entropy_lower_bound − 0.10  (R2 floor sanity)
    """
    out_dir = tmp_path / "mu_sigma"
    for seed in range(5):
        cmd = [
            "python", "-m", "cdsbi.experiments.run",
            f"hydra.run.dir={out_dir}/run_seed_{seed}",
            "experiment=mu_sigma_replication",
            "budget=medium",
            f"seed={seed}",
            "training.fresh_batch=false",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        assert result.returncode == 0, f"seed {seed} failed:\n{result.stderr[-2000:]}"

    from cdsbi.analysis.loaders import load_runs

    df = load_runs(str(out_dir / "*"))
    assert len(df) == 5, f"expected 5 runs, got {len(df)}"

    print("\n--- per-seed metrics ---")
    print(df[["pivot_rmse", "coverage_error_max",
              "marginal_cd_sigma_ks", "marginal_cd_mu_ks",
              "marginal_cd_mu_t_resid", "final_loss"]].to_string())

    # (a) pivot recovery + coverage (single-index flow, 2-D)
    assert df["pivot_rmse"].mean() <= 0.30, (
        f"pivot_rmse mean = {df['pivot_rmse'].mean():.4f} > 0.30"
    )
    assert df["coverage_error_max"].mean() <= 0.05, (
        f"coverage_error_max mean = {df['coverage_error_max'].mean():.4f} > 0.05"
    )

    # (b) marginal-CD recovery: σ² (χ²) and μ (Student-t) both calibrate
    assert df["marginal_cd_sigma_ks"].mean() <= 0.06, (
        f"marginal_cd_sigma_ks mean = {df['marginal_cd_sigma_ks'].mean():.4f} > 0.06"
    )
    assert df["marginal_cd_mu_ks"].mean() <= 0.06, (
        f"marginal_cd_mu_ks mean = {df['marginal_cd_mu_ks'].mean():.4f} > 0.06"
    )
    assert df["marginal_cd_mu_t_resid"].mean() <= 0.05, (
        f"marginal_cd_mu_t_resid mean = {df['marginal_cd_mu_t_resid'].mean():.4f} > 0.05"
    )

    # (c) entropy floor — final NF-MLE loss must not sink below the floor (R2 sanity)
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    H = NormalUnknownMeanVar().entropy_lower_bound()
    print(f"\nentropy_lower_bound H = {H:.4f}")
    print(f"final_loss mean = {df['final_loss'].mean():.4f}")
    assert df["final_loss"].mean() > H - 0.10, (
        f"final_loss {df['final_loss'].mean():.3f} below entropy floor {H:.3f} − 0.10"
    )

    # (d) per-run JointMahalanobis at 2× floor on ≥ 4/5 seeds
    run_dirs = sorted(glob.glob(str(out_dir / "*")))
    assert len(run_dirs) == 5, f"expected 5 run dirs, got {len(run_dirs)}"
    n_pass = 0
    for rd in run_dirs:
        jm_path = os.path.join(rd, "diagnostics", "joint_mahalanobis.parquet")
        assert os.path.exists(jm_path), f"JM parquet missing for {rd}"
        jm = pd.read_parquet(jm_path)
        if (jm["ks"] <= 2.0 * jm["noise_floor"]).all():
            n_pass += 1
    assert n_pass >= 4, (
        f"only {n_pass}/5 seeds passed JointMahalanobis at 2× floor; expected ≥ 4"
    )
