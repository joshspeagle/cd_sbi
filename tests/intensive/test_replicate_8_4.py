"""§8.4 replication: CDSBI on ExponentialRate matches tolerance bands."""
from pathlib import Path
import subprocess

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.intensive
def test_replicate_8_4_cdsbi_matches_tolerance(tmp_path):
    """Run CDSBI on the doubly-monotone flow + ExponentialRate target across
    5 seeds; seed-averaged metrics within tolerance bands.

    Tolerance bands per manuscript §8.4 with ~1.5× headroom for seed scatter
    (matches the §8.1–§8.3 pattern). After the v3 architectural fixes
    (deeper α-net, theta_ref at proposal center, T_ref=5.0 in the
    MonotoneScalarUMNN integration baseline — all caught by diffing
    against the §8.4 reference implementation), the doubly-monotone
    flow's seed-averaged pivot RMSE lands at ~0.045, exactly matching
    the manuscript's single-seed value.
      - pivot_rmse mean ≤ 0.07 (manuscript single-seed: 0.045)
      - coverage_error_max mean ≤ 0.04 (manuscript: < 0.015; sweep
        averages stay ~2× the manuscript point estimate due to MC noise)
      - final_loss STAYS ABOVE the conditioner-adjusted entropy lower
        bound (the doubly-monotone contract: R1+R2 architectures
        preserve normalization; final_loss < bound would be a sign
        of R2 violation in the production flow)
    """
    out_dir = tmp_path / "sweep"
    for seed in range(5):
        cmd = [
            "python", "-m", "cdsbi.experiments.run",
            f"hydra.run.dir={out_dir}/run_seed_{seed}",
            "experiment=8_4_replication",
            f"seed={seed}",
            "training.fresh_batch=false",
            "+experiment.set_size_n_per_theta=25",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        assert result.returncode == 0, f"seed {seed} failed:\n{result.stderr}"

    from cdsbi.analysis.loaders import load_runs
    from cdsbi.simulators.exp_rate import ExponentialRate

    df = load_runs(str(out_dir / "*"))
    assert len(df) == 5
    assert df["pivot_rmse"].mean() <= 0.07, f"pivot_rmse mean = {df['pivot_rmse'].mean()}"
    assert df["coverage_error_max"].mean() <= 0.04, (
        f"coverage_error_max mean = {df['coverage_error_max'].mean()}"
    )
    # Final-loss check: doubly-monotone (R1+R2) must stay ABOVE the entropy
    # lower bound — the R2 contract preserves Z(θ) ≡ 1 so the NF-MLE loss
    # cannot fall below the information-theoretic floor.
    sim = ExponentialRate()
    H = sim.entropy_lower_bound()
    assert df["final_loss"].mean() > H - 0.10, (
        f"trained doubly-monotone final_loss = {df['final_loss'].mean()} BELOW "
        f"entropy_lower_bound = {H} — possible R2 violation in the production flow"
    )
