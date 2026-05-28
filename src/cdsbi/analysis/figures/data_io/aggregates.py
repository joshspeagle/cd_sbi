"""Load index_row.parquet aggregates from sweep roots into one tidy DataFrame.

Wraps the existing cdsbi.analysis.loaders.load_runs (status=OK filtering,
glob-based) so figures share the same loading semantics as paper tables.
"""
from __future__ import annotations

import os

import pandas as pd

from cdsbi.analysis.loaders import load_runs


def load_aggregates(roots: list[str]) -> pd.DataFrame:
    """Concatenate per-run index rows under each sweep root.

    `roots` are sweep directories (each containing method=...,seed=... subdirs).
    Returns an empty DataFrame if `roots` is empty or no OK runs are found.
    """
    frames = []
    for root in roots:
        pattern = os.path.join(root, "*")
        df = load_runs(pattern)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
