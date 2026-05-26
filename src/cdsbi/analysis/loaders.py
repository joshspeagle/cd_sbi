"""Run-dir loaders: glob over per-run index_row.parquet files, return one DataFrame."""
from __future__ import annotations

import glob as _glob
from pathlib import Path

import pandas as pd


def load_run(run_dir: str) -> dict:
    """Load a single run directory: read STATUS and index_row.parquet."""
    rd = Path(run_dir)
    status = (rd / "STATUS").read_text().strip() if (rd / "STATUS").exists() else "UNKNOWN"
    row = pd.read_parquet(rd / "index_row.parquet").iloc[0].to_dict()
    row["status"] = status
    row["run_dir"] = str(rd)
    return row


def load_runs(pattern: str) -> pd.DataFrame:
    """Glob pattern matches run dirs; concatenate their index_row.parquet (status=OK only).

    Skips any run directory that:
    - Has no STATUS file or STATUS != "OK"
    - Has no index_row.parquet

    Returns empty DataFrame if no valid runs found.
    """
    rows = []
    for path in _glob.glob(pattern):
        rd = Path(path)
        status_file = rd / "STATUS"
        if not status_file.exists() or status_file.read_text().strip() != "OK":
            continue
        index_row = rd / "index_row.parquet"
        if not index_row.exists():
            continue
        rows.append(pd.read_parquet(index_row))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
