"""Tests for cdsbi.analysis.loaders."""
from pathlib import Path

import pandas as pd

from cdsbi.analysis.loaders import load_runs


def test_load_runs_concatenates_index_rows(tmp_path):
    """Test that load_runs concatenates multiple run index_row.parquet files."""
    for i in range(3):
        d = tmp_path / f"run{i}"
        d.mkdir()
        (d / "STATUS").write_text("OK")
        pd.DataFrame(
            [{"config_hash": f"h{i}", "method": "cd_sbi", "seed": i, "coverage_error_max": 0.01 * i}]
        ).to_parquet(d / "index_row.parquet")
    df = load_runs(str(tmp_path / "*"))
    assert len(df) == 3
    assert set(df["method"]) == {"cd_sbi"}


def test_load_runs_skips_non_ok(tmp_path):
    """Test that load_runs skips runs with STATUS != OK."""
    d_ok = tmp_path / "ok"
    d_ok.mkdir()
    (d_ok / "STATUS").write_text("OK")
    pd.DataFrame([{"config_hash": "a", "method": "cd_sbi", "seed": 0}]).to_parquet(d_ok / "index_row.parquet")
    d_fail = tmp_path / "fail"
    d_fail.mkdir()
    (d_fail / "STATUS").write_text("FAILED")
    pd.DataFrame([{"config_hash": "b", "method": "cd_sbi", "seed": 1}]).to_parquet(d_fail / "index_row.parquet")
    df = load_runs(str(tmp_path / "*"))
    assert len(df) == 1
    assert df["config_hash"].iloc[0] == "a"
