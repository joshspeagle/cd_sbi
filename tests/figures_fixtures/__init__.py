"""Synthetic mini-run-dir builders for figure tests.

Mirrors the real run-dir layout written by cdsbi.experiments.run._write_index_row:
a per-run directory with STATUS, index_row.parquet, and model.pt.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch


def make_run_dir(
    root: Path,
    method: str,
    budget_name: str,
    seed: int,
    coverage_error_max: float,
    final_loss: float,
    actual_params_total: int = 5000,
    loss_history_tail: list[float] | None = None,
) -> Path:
    """Create one synthetic run-dir under `root` and return its path.

    The subdir name mimics the real Hydra run-dir template
    (method=...,budget=...,seed=...) closely enough for glob-based loaders.
    """
    name = f"method={method},budget={budget_name},seed={seed}"
    rd = root / name
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "STATUS").write_text("OK")
    row = {
        "experiment": "test_exp",
        "method": method,
        "budget_name": budget_name,
        "seed": seed,
        "actual_params_total": actual_params_total,
        "coverage_error_max": coverage_error_max,
        "marginal_ks": 0.013,
        "pivot_rmse": 0.04 if method == "cd_sbi" else None,
        "final_loss": final_loss,
    }
    pd.DataFrame([row]).to_parquet(rd / "index_row.parquet")
    tail = loss_history_tail if loss_history_tail is not None else [final_loss + 0.1, final_loss]
    torch.save(
        {"arch_metadata": {"loss_history_tail": tail}, "final_loss": final_loss},
        rd / "model.pt",
    )
    return rd


def make_sweep(root: Path) -> Path:
    """Create a small multi-method, multi-seed sweep root and return it.

    Two methods x two seeds = four run-dirs. Enough to exercise grouping.
    """
    root.mkdir(parents=True, exist_ok=True)
    for method, cov in (("cd_sbi", 0.025), ("npe", 0.07)):
        for seed in (0, 1):
            make_run_dir(
                root, method=method, budget_name="medium", seed=seed,
                coverage_error_max=cov + 0.001 * seed, final_loss=1.0,
            )
    return root


def write_marginal_pit_raw(run_dir: Path, u) -> None:
    """Write a diagnostics/marginal_pit_raw.parquet with a 1-D 'u' column."""
    import numpy as _np
    diag = run_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"u": _np.asarray(u)}).to_parquet(diag / "marginal_pit_raw.parquet")


def write_jacobian_raw(run_dir: Path, j_emp, j_true) -> None:
    """Write a diagnostics/jacobian_recovery_raw.parquet (i, j, j_emp, j_true)."""
    import numpy as _np
    j_emp = _np.asarray(j_emp); j_true = _np.asarray(j_true)
    d = j_emp.shape[0]
    diag = run_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    rows = [{"i": i, "j": j, "j_emp": float(j_emp[i, j]), "j_true": float(j_true[i, j])}
            for i in range(d) for j in range(d)]
    pd.DataFrame(rows).to_parquet(diag / "jacobian_recovery_raw.parquet")


def write_coverage(run_dir: Path, theta0_list, alpha_list) -> None:
    """Write a d=1 diagnostics/coverage.parquet (theta_0_0, alpha, nominal, empirical)."""
    import numpy as _np
    diag = run_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    rng = _np.random.default_rng(0)
    rows = []
    for t in theta0_list:
        for a in alpha_list:
            rows.append({"theta_0_0": float(t), "alpha": float(a), "nominal": float(a),
                         "empirical": float(a) + rng.normal(0, 0.01), "n_eval": 2000})
    pd.DataFrame(rows).to_parquet(diag / "coverage.parquet")


def write_coverage_2d(run_dir: Path, grid_points, alpha_list) -> None:
    """Write a d=2 coverage.parquet (theta_0_0, theta_0_1, alpha, nominal, empirical).
    `grid_points` is a list of (t0, t1) tuples (several may share a first coord)."""
    import numpy as _np
    diag = run_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    rng = _np.random.default_rng(0)
    rows = []
    for (t0, t1) in grid_points:
        for a in alpha_list:
            rows.append({"theta_0_0": float(t0), "theta_0_1": float(t1),
                         "alpha": float(a), "nominal": float(a),
                         "empirical": float(a) + rng.normal(0, 0.01), "n_eval": 2000})
    pd.DataFrame(rows).to_parquet(diag / "coverage.parquet")
