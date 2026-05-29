"""E7 — §8.4 catastrophic folding: R1-only loss tail dips below the entropy floor.

Source is a folding-regen DIRECTORY containing folding_tail.parquet
(columns step, loss, entropy_floor). CDSBIRunner.fit persists only the final
100 training steps; that window already sits below the floor — the folding
signature — so this plots the persisted tail (not a full trajectory).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cdsbi.analysis.figures import style, panels
from cdsbi.analysis.figures.manifest import FigureSpec


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt
    df = pd.read_parquet(Path(spec.source_runs[0]) / "folding_tail.parquet")
    floor = float(df["entropy_floor"].iloc[0])
    fig, ax = plt.subplots(figsize=style.SIZES["single_column"])
    panels.loss_trajectory_with_floor(
        ax, {"R1 only (long recipe, final 100 steps)": df["loss"].to_numpy()},
        floor=floor)
    ax.set_xlabel("training step (final 100)")
    ax.legend(fontsize=7)
    ax.set_title("§8.4 catastrophic folding")
    fig.tight_layout()
    return fig
