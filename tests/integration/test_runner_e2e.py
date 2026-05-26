import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parents[2]


def test_cli_runs_cdsbi_minimal(tmp_path, repo_root):
    """End-to-end: invoke the CLI on a tiny budget; assert run dir is populated."""
    cmd = [
        "python", "-m", "cdsbi.experiments.run",
        f"hydra.run.dir={tmp_path}/run",
        "method=cd_sbi",
        "budget=small",
        "training.n_steps=20",
        "training.n_train=200",
        "experiment.n_eval_per_theta=100",
        "experiment.eval_thetas_interior=[0.0]",
        "experiment.alpha_grid=[0.9]",
        "seed=0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    run_dir = tmp_path / "run"
    assert (run_dir / "STATUS").read_text().strip() == "OK"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "env.json").exists()
    env = json.loads((run_dir / "env.json").read_text())
    assert "git_sha" in env
    assert (run_dir / "diagnostics" / "coverage.parquet").exists()
    df = pd.read_parquet(run_dir / "diagnostics" / "coverage.parquet")
    assert set(df.columns) >= {"theta_0_0", "alpha", "nominal", "empirical", "n_eval", "passed", "tolerance"}
    assert (run_dir / "index_row.parquet").exists()
