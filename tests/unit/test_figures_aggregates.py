"""Aggregates loader: glob sweep roots -> one tidy DataFrame of index rows."""
from __future__ import annotations

from tests.figures_fixtures import make_sweep


def test_load_aggregates_concatenates_all_runs(tmp_path):
    from cdsbi.analysis.figures.data_io.aggregates import load_aggregates
    root = make_sweep(tmp_path / "sweep")
    df = load_aggregates([str(root)])
    assert len(df) == 4
    assert set(df["method"]) == {"cd_sbi", "npe"}
    assert "coverage_error_max" in df.columns


def test_load_aggregates_merges_multiple_roots(tmp_path):
    from cdsbi.analysis.figures.data_io.aggregates import load_aggregates
    root_a = make_sweep(tmp_path / "a")
    root_b = make_sweep(tmp_path / "b")
    df = load_aggregates([str(root_a), str(root_b)])
    assert len(df) == 8


def test_load_aggregates_empty_list_returns_empty_frame(tmp_path):
    from cdsbi.analysis.figures.data_io.aggregates import load_aggregates
    df = load_aggregates([])
    assert df.empty


def test_load_aggregates_skips_non_ok_runs(tmp_path):
    from cdsbi.analysis.figures.data_io.aggregates import load_aggregates
    root = make_sweep(tmp_path / "sweep")
    # Corrupt one run's STATUS so it is excluded. Use a default + assert so a
    # fixture-naming drift surfaces as a clear failure, not a bare StopIteration.
    bad = next(root.glob("method=cd_sbi,*seed=0*"), None)
    assert bad is not None, "fixture run-dir naming changed; update this glob"
    (bad / "STATUS").write_text("FAILED")
    df = load_aggregates([str(root)])
    assert len(df) == 3
