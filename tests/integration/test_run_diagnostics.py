"""End-to-end integration tests for the diagnostic battery wired into
`cdsbi.experiments.run`. These subprocess-invoke the Hydra entrypoint
to validate that every requested diagnostic produces its parquet and
that the per-run `index_row.parquet` lifts the relevant scalars."""
from __future__ import annotations


def test_run_produces_jacobian_recovery_parquet_on_corr_target(tmp_path, seed):
    """End-to-end: a single CDSBI run on the correlated simulator writes a
    jacobian_recovery.parquet with sane numeric columns and lifts
    jacobian_max_residual into index_row.parquet."""
    import subprocess, pandas as pd
    from pathlib import Path

    out = tmp_path / "run"
    cmd = [
        "python", "-m", "cdsbi.experiments.run",
        f"hydra.run.dir={out}",
        "experiment=8_3_replication",
        "method=cd_sbi",
        "budget=small",
        f"seed={seed}",
        "training.fresh_batch=false",
        # Trim diagnostic costs for the integration smoke. set_size_n_per_theta
        # is NOT in 8_3_replication.yaml, so it needs `+`; joint_mahalanobis_n_per_theta
        # IS in the YAML (= 2000), so use plain assignment to override it.
        "+experiment.set_size_n_per_theta=25",
        "experiment.joint_mahalanobis_n_per_theta=300",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
    assert result.returncode == 0, result.stderr
    diag = pd.read_parquet(out / "diagnostics" / "jacobian_recovery.parquet")
    assert {"max_residual", "norm_residual", "passed", "n_points"}.issubset(diag.columns)
    assert diag["n_points"].iloc[0] > 0
    idx = pd.read_parquet(out / "index_row.parquet")
    assert "jacobian_max_residual" in idx.columns
    assert idx["jacobian_max_residual"].iloc[0] >= 0


def test_run_produces_doubly_monotone_flow_on_exp_rate(tmp_path, seed):
    """End-to-end: a single CDSBI run on the ExponentialRate target with
    DoublyMonotoneUMNN writes STATUS=OK and produces a valid index_row."""
    import subprocess, pandas as pd
    from pathlib import Path
    out = tmp_path / "run"
    cmd = [
        "python", "-m", "cdsbi.experiments.run",
        f"hydra.run.dir={out}",
        "experiment=8_4_replication",
        f"seed={seed}",
        "training.fresh_batch=false",
        "+experiment.set_size_n_per_theta=25",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
    assert result.returncode == 0, result.stderr
    assert (out / "STATUS").read_text().strip() == "OK"
    idx = pd.read_parquet(out / "index_row.parquet")
    assert idx["method"].iloc[0] == "cd_sbi"
    assert idx["flow"].iloc[0] == "doubly_monotone"
    assert idx["target"].iloc[0] == "exp_rate"
    # Final loss should be near the entropy_lower_bound (≈ 0.99 with our
    # MC-based bound) within ~0.2 at small budget — smoke test, not tight.
    assert idx["final_loss"].iloc[0] > 0.4, f"final_loss = {idx['final_loss'].iloc[0]} too low — possible R2 violation"
