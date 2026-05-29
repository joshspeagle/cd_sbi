"""Loaders that turn a run-dir into the arrays the E-figures consume.

Coverage curve/tile come from the standard diagnostics/coverage.parquet.
Raw PIT / Jacobian come from the *_raw.parquet companions written by
run.py when the diagnostics carry raw arrays in their result.meta (F2 Phase A).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cdsbi.analysis.figures.data_io.checkpoints import load_checkpoint


def load_sweep(sweep_root: str) -> pd.DataFrame:
    """Aggregate index rows across ALL timestamp dirs under a sweep root.

    The repo splits a section's runs across multiple timestamped sweep dirs
    (method-specific re-runs). This globs every run-dir under `sweep_root`
    (handling both <root>/<timestamp>/<run-dir> and a flat <root>/<run-dir>
    layout), keeps only STATUS==OK runs, and deduplicates on
    (method, budget_name, seed) keeping the LATEST path (re-runs supersede).
    """
    rundirs = set()
    for pat in (os.path.join(sweep_root, "*", "index_row.parquet"),
                os.path.join(sweep_root, "*", "*", "index_row.parquet")):
        for p in glob.glob(pat):
            rundirs.add(os.path.dirname(p))
    rows = []
    for rd in sorted(rundirs):
        status = os.path.join(rd, "STATUS")
        if os.path.exists(status) and open(status).read().strip() != "OK":
            continue
        df = pd.read_parquet(os.path.join(rd, "index_row.parquet"))
        df["_src"] = rd
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    alldf = pd.concat(rows, ignore_index=True).sort_values("_src")
    keys = [c for c in ("method", "budget_name", "seed") if c in alldf.columns]
    return alldf.drop_duplicates(keys, keep="last").drop(columns=["_src"])


def load_pit_values(run_dir: str, coord: int = 0) -> np.ndarray:
    """Raw PIT values from diagnostics/marginal_pit_raw.parquet.

    For d=1 the column is 'u'; for d>1 columns are 'u0','u1',...; `coord` picks one.
    """
    df = pd.read_parquet(Path(run_dir) / "diagnostics" / "marginal_pit_raw.parquet")
    col = "u" if "u" in df.columns else f"u{coord}"
    return df[col].to_numpy()


def load_joint_mahalanobis_sq(run_dir: str) -> dict[str, np.ndarray]:
    """Per-θ_0 squared-norm arrays from joint_mahalanobis_raw.parquet."""
    df = pd.read_parquet(Path(run_dir) / "diagnostics" / "joint_mahalanobis_raw.parquet")
    return {k: g["r_sq"].to_numpy() for k, g in df.groupby("theta_0_repr")}


def load_jacobian_matrices(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """(J_emp_mean, J_true) matrices from jacobian_recovery_raw.parquet."""
    df = pd.read_parquet(Path(run_dir) / "diagnostics" / "jacobian_recovery_raw.parquet")
    d = int(df["i"].max()) + 1
    j_emp = np.zeros((d, d)); j_true = np.zeros((d, d))
    for _, row in df.iterrows():
        j_emp[int(row["i"]), int(row["j"])] = row["j_emp"]
        j_true[int(row["i"]), int(row["j"])] = row["j_true"]
    return j_emp, j_true


def load_coverage_curve(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """(nominal, empirical) averaged over θ_0, one point per α — sorted by α."""
    df = pd.read_parquet(Path(run_dir) / "diagnostics" / "coverage.parquet")
    g = df.groupby("alpha", as_index=False).agg(nominal=("nominal", "first"),
                                                 empirical=("empirical", "mean"))
    g = g.sort_values("alpha")
    return g["nominal"].to_numpy(), g["empirical"].to_numpy()


def load_coverage_tile(run_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(theta0, alpha, error), error[i,j] = |empirical-nominal| at (grid point i, α_j).

    Keys rows on the FULL θ_0 grid point (every theta_0_* column), so d>1 grids
    whose points share a first coordinate (e.g. §8.2's [[-3,-3],[-3,0],...]) are
    NOT conflated. For d=1 the row axis is the θ_0 value; for d>1 it is the
    grid-point index 0..G-1.
    """
    df = pd.read_parquet(Path(run_dir) / "diagnostics" / "coverage.parquet")
    theta_cols = sorted(c for c in df.columns if c.startswith("theta_0_"))
    alpha = np.sort(df["alpha"].unique())
    grid = df[theta_cols].drop_duplicates().sort_values(theta_cols).to_numpy()
    G = grid.shape[0]
    theta_arr = df[theta_cols].to_numpy()
    error = np.zeros((G, len(alpha)))
    for i in range(G):
        mask_i = np.all(theta_arr == grid[i], axis=1)
        for j, a in enumerate(alpha):
            sub = df[mask_i & (df["alpha"] == a)]
            error[i, j] = float((sub["empirical"] - sub["nominal"]).abs().mean())
    theta0 = grid[:, 0] if len(theta_cols) == 1 else np.arange(G, dtype=float)
    return theta0, alpha, error


def load_loss_tail_mean(run_dir: str) -> float:
    """Mean of the persisted loss_history_tail from model.pt."""
    ck = load_checkpoint(run_dir)
    tail = ck.loss_history_tail
    if tail is None:
        return float(ck.final_loss)
    return float(np.mean(np.asarray(tail)))
