"""§8.2 replication: CDSBI on LocationGaussian2D_iid matches tolerance bands."""
from pathlib import Path
import subprocess

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.intensive
def test_replicate_8_2_cdsbi_matches_tolerance(tmp_path):
    """Run CDSBI at medium budget across 5 seeds; seed-averaged metrics within tolerance."""
    out_dir = tmp_path / "sweep"
    for seed in range(5):
        cmd = [
            "python", "-m", "cdsbi.experiments.run",
            f"hydra.run.dir={out_dir}/run_seed_{seed}",
            "experiment=8_2_replication",
            "method=cd_sbi",
            "budget=medium",
            f"seed={seed}",
            "training.fresh_batch=false",
            # SetSize at d>1 falls through to a per-X_obs ray-bisection loop
            # (no confidence_set_batch fast path for d>1 PivotBased yet).
            # Each call is ~50 batched pivot evaluations through the trained
            # TriangularAdditiveFlow (~250ms on GPU); n_per_theta=25 keeps
            # the intensive test ≈ 1-2 min/seed wall instead of 8+ min/seed.
            # JointMahalanobis is the inferentially-primary 2D diagnostic
            # here; SetSize widths are sanity-only for the band-check.
            "+experiment.set_size_n_per_theta=25",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        assert result.returncode == 0, f"seed {seed} failed:\n{result.stderr}"

    from cdsbi.analysis.loaders import load_runs
    df = load_runs(str(out_dir / "*"))
    assert len(df) == 5
    # Tolerance bands per manuscript §8.2 (loosened from §8.1 by ~1.5–2×
    # to absorb 2D conditioning-network noise on the r_2 coordinate;
    # paper reports total pivot RMSE 0.044, KS r_1/r_2 0.007/0.006,
    # joint Mahalanobis KS 0.011, coverage error < 0.01).
    assert df["pivot_rmse"].mean() <= 0.08, f"pivot_rmse mean = {df['pivot_rmse'].mean()}"
    assert df["coverage_error_max"].mean() <= 0.03, (
        f"coverage_error mean = {df['coverage_error_max'].mean()}"
    )
    # Joint Mahalanobis: load per-run parquet and check KS ≤ 2× floor for ≥ 4/5.
    import glob, os
    n_pass = 0
    for rd in glob.glob(str(out_dir / "*")):
        jm_path = os.path.join(rd, "diagnostics/joint_mahalanobis.parquet")
        if not os.path.exists(jm_path):
            continue
        jm = pd.read_parquet(jm_path)
        if (jm["ks"] <= 2.0 * jm["noise_floor"]).all():
            n_pass += 1
    assert n_pass >= 4, (
        f"only {n_pass}/5 seeds passed JointMahalanobis at 2× floor; "
        f"expected ≥ 4"
    )
