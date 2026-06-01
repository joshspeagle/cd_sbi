"""Intensive: Phase-1 regular consistency on (μ,σ²). Two-stage moment-summary should
be valid + efficient + no-collapse, matching the oracle and beating the M2 collapse."""
from pathlib import Path
import subprocess
import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.intensive
def test_replicate_moment_mu_sigma(tmp_path):
    out = tmp_path / "moment"
    for seed in range(3):
        cmd = ["python", "-m", "cdsbi.experiments.run",
               f"hydra.run.dir={out}/seed_{seed}",
               "experiment=moment_mu_sigma", f"seed={seed}", "training.fresh_batch=true"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, f"seed {seed} failed:\n{r.stderr[-2000:]}"

    from cdsbi.analysis.loaders import load_runs
    df = load_runs(str(out / "*"))
    assert len(df) == 3
    print(df[["coverage_error_max", "marginal_cd_sigma_ks"]].to_string())
    # validity + no-collapse (σ-recovery shows in the σ marginal-CD KS)
    assert df["coverage_error_max"].mean() <= 0.06
    assert df["marginal_cd_sigma_ks"].mean() <= 0.10
