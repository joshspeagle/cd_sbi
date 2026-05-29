"""E6 — §8.4 (R2) ablation: tail-mean loss, R1+R2 vs R1-only across budgets."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

from cdsbi.analysis.figures import style, panels
from cdsbi.analysis.figures.manifest import FigureSpec
from cdsbi.analysis.figures.data_io.figure_data import load_loss_tail_mean
from cdsbi.simulators.exp_rate import ExponentialRate

BUDGETS = ["small", "medium", "large", "xlarge"]


def _budget_of(run_dir: str):
    for tok in Path(run_dir).name.split(","):
        if tok.startswith("budget="):
            return tok.split("=", 1)[1]
    return None


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt
    # source_runs[0] = R1+R2 arm (cd_sbi/doubly_monotone sweep); [1] = R1-only ablation
    arms = {"R1+R2": spec.source_runs[0], "R1 only": spec.source_runs[1]}
    bars = {}
    for arm, root in arms.items():
        per_budget = {b: [] for b in BUDGETS}
        for rd in glob.glob(str(Path(root) / "*")):
            if not os.path.isdir(rd) or not (Path(rd) / "model.pt").exists():
                continue
            # Both arms are CDSBI runs (R1+R2 = doubly_monotone, R1-only =
            # joint_umnn). The R1+R2 sweep dir also holds the 4 baseline
            # methods — filter to method=cd_sbi so we average only the
            # doubly-monotone arm, not unrelated NPE/NLE/NRE/LF2I losses.
            if "method=cd_sbi" not in Path(rd).name:
                continue
            b = _budget_of(rd)
            if b in per_budget:
                per_budget[b].append(load_loss_tail_mean(rd))
        bars[arm] = np.array([np.mean(per_budget[b]) if per_budget[b] else np.nan
                              for b in BUDGETS])
    fig, ax = plt.subplots(figsize=style.SIZES["single_column"])
    panels.loss_bar_with_floor(ax, bars,
                               floor=float(ExponentialRate().entropy_lower_bound()),
                               group_labels=["S", "M", "L", "XL"])
    ax.legend(fontsize=7)
    ax.set_title("§8.4 (R2) ablation: final loss vs entropy floor")
    fig.tight_layout()
    return fig
