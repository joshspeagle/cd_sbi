"""Full §8.1 replication: CDSBI on LocationNormal1D matches tolerance bands."""
from pathlib import Path
import subprocess

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.intensive
def test_replicate_8_1_cdsbi_matches_tolerance(tmp_path):
    """Run CDSBI at medium budget across 5 seeds; seed-averaged metrics within tolerance."""
    out_dir = tmp_path / "sweep"
    for seed in range(5):
        # subdir uses 'run_seed_N' (not 'seed=N') so Hydra's key=value override
        # parser doesn't see a stray '=' inside the path argument
        cmd = [
            "python", "-m", "cdsbi.experiments.run",
            f"hydra.run.dir={out_dir}/run_seed_{seed}",
            "method=cd_sbi", "budget=medium",
            f"seed={seed}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        assert result.returncode == 0, f"seed {seed} failed:\n{result.stderr}"

    from cdsbi.analysis.loaders import load_runs
    df = load_runs(str(out_dir / "*"))
    assert len(df) == 5
    # Seed-averaged tolerance bands from spec §11(2)
    assert df["pivot_rmse"].mean() <= 0.05, f"pivot_rmse mean = {df['pivot_rmse'].mean()}"
    assert df["marginal_ks"].mean() <= 0.023, f"marginal_ks mean = {df['marginal_ks'].mean()}"
    assert df["coverage_error_max"].mean() <= 0.02, f"coverage_error mean = {df['coverage_error_max'].mean()}"
