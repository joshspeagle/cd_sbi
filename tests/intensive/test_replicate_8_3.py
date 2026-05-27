"""§8.3 replication: CDSBI on LocationGaussian2D_corr matches tolerance bands."""
from pathlib import Path
import subprocess

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.intensive
def test_replicate_8_3_cdsbi_matches_tolerance(tmp_path):
    """Run CDSBI at medium budget across 5 seeds; seed-averaged metrics within tolerance.

    Tolerance bands per manuscript §8.3 (loosened ~1.5–2× from the single-seed
    point estimates to absorb seed scatter, mirroring the v1 approach):
      - pivot_rmse mean ≤ 0.45 (manuscript: 0.27 — high pointwise residual is
        expected because L⁻¹ scales r_2 toward unit marginal variance, not 1/30
        of dynamic range as in §8.2; the Jacobian + joint Mahalanobis are the
        load-bearing diagnostics here, per the §8.3 commentary)
      - coverage_error_max mean ≤ 0.035 (manuscript: < 0.015)
      - joint_mahal_ks ≤ 2 × per-θ_0 floor for ≥ 4/5 seeds
      - jacobian_max_residual mean ≤ 0.05 (manuscript: 1–2%)
    """
    out_dir = tmp_path / "sweep"
    for seed in range(5):
        cmd = [
            "python", "-m", "cdsbi.experiments.run",
            f"hydra.run.dir={out_dir}/run_seed_{seed}",
            "experiment=8_3_replication",
            "method=cd_sbi",
            "budget=medium",
            f"seed={seed}",
            "training.fresh_batch=false",
            # SetSize at d>1 falls through to ray-bisection; cap it the same way
            # v1's §8.2 intensive test does.
            "+experiment.set_size_n_per_theta=25",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        assert result.returncode == 0, f"seed {seed} failed:\n{result.stderr}"

    from cdsbi.analysis.loaders import load_runs
    df = load_runs(str(out_dir / "*"))
    assert len(df) == 5
    assert df["pivot_rmse"].mean() <= 0.45, f"pivot_rmse mean = {df['pivot_rmse'].mean()}"
    assert df["coverage_error_max"].mean() <= 0.035, (
        f"coverage_error mean = {df['coverage_error_max'].mean()}"
    )
    # JacobianRecovery wiring is load-bearing for the §8.3 KR-uniqueness claim
    # — fail loudly if the column is missing rather than silently skipping.
    assert "jacobian_max_residual" in df.columns, (
        "JacobianRecovery not wired through to index_row.parquet"
    )
    assert df["jacobian_max_residual"].mean() <= 0.05, (
        f"jacobian_max_residual mean = {df['jacobian_max_residual'].mean()}"
    )

    # Joint Mahalanobis: load per-run parquet and check KS ≤ 2× floor for ≥ 4/5.
    # Fail loudly if a JM parquet is missing — a silent skip would let a corrupt
    # run mask a real failure by deflating n_pass.
    import glob, os
    n_pass = 0
    run_dirs = sorted(glob.glob(str(out_dir / "*")))
    assert len(run_dirs) == 5, f"expected 5 run dirs, got {len(run_dirs)}"
    for rd in run_dirs:
        jm_path = os.path.join(rd, "diagnostics/joint_mahalanobis.parquet")
        assert os.path.exists(jm_path), f"JM parquet missing for {rd}"
        jm = pd.read_parquet(jm_path)
        if (jm["ks"] <= 2.0 * jm["noise_floor"]).all():
            n_pass += 1
    assert n_pass >= 4, (
        f"only {n_pass}/5 seeds passed JointMahalanobis at 2× floor; expected ≥ 4"
    )
