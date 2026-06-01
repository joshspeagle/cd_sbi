"""Intensive: Phase-1 (μ,σ²) replication of the non-oracle moment-summary CD.

VERDICT (theory note §17): this xfails. Central coverage is good (~0.012) and the
summary recovers σ-info (Fisher 0.95, no collapse), but *uniform* (worst-θ₀)
calibration is ~0.18 (vs oracle ~0.02) — the single-index ceiling (§16; a feature-
warp cuts it to ~0.10) plus the learned summary's μ-entanglement. The pivot/CD is
validated for the oracle/regular regime; LF2I is the path for the general regime.
Kept as a recorded finding (not a target) — hence xfail."""
from pathlib import Path
import subprocess
import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.intensive
@pytest.mark.xfail(reason="non-oracle learned-summary uniform-calibration gap — "
                          "recorded finding, see theory note §17", strict=False)
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
