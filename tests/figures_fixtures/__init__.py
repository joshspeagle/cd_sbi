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
