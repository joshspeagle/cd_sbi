"""Training-loss panels for the (R2) ablation figures."""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures.panels.primitives import noise_floor_band


def loss_trajectory_with_floor(ax, trajectories, *, floor=None):
    """Plot NF-MLE loss vs training step for one or more runs.

    `trajectories` maps label -> 1-D loss array. With `floor`, draws the
    entropy lower bound as a dashed horizontal line.
    """
    for label, losses in trajectories.items():
        losses = np.asarray(losses)
        ax.plot(np.arange(len(losses)), losses, label=label)
    ax.set_xlabel("training step")
    ax.set_ylabel("NF-MLE loss")
    if floor is not None:
        noise_floor_band(ax, floor, label="entropy floor")
    return ax


def loss_bar_with_floor(ax, bars, *, floor=None, group_labels=None):
    """Grouped bar chart of a (tail-mean) loss per series across budget groups.

    `bars` maps series_label -> 1-D array, one value per group. With `floor`,
    draws the entropy lower bound as a dashed horizontal line.
    """
    series = list(bars.keys())
    n_groups = len(next(iter(bars.values())))
    x = np.arange(n_groups)
    width = 0.8 / len(series)
    for i, s in enumerate(series):
        ax.bar(x + i * width, np.asarray(bars[s]), width=width, label=s)
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    if group_labels is not None:
        ax.set_xticklabels(group_labels)
    ax.set_ylabel("final NF-MLE loss")
    if floor is not None:
        noise_floor_band(ax, floor, label="entropy floor")
    return ax
