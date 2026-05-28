"""Cross-method comparison panels."""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures.panels.primitives import noise_floor_band


def _ordered(methods_present):
    """Methods present, in canonical best->worst order."""
    return [m for m in style.CANONICAL_ORDER if m in methods_present]


def boxplot_per_method(ax, data, *, ylabel="coverage error (max)"):
    """Boxplot of a per-method metric distribution (across seeds/budgets).

    `data` maps method -> 1-D array of values. Boxes appear in canonical order,
    each filled with its method colour.
    """
    methods = _ordered(data)
    values = [np.asarray(data[m]) for m in methods]
    bp = ax.boxplot(values, patch_artist=True,
                    tick_labels=[style.METHOD_STYLE[m]["label"] for m in methods])
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(style.METHOD_STYLE[m]["color"])
        patch.set_alpha(0.7)
    ax.set_ylabel(ylabel)
    return ax


def cross_method_summary_log_y(ax, data, *, floor=None, ylabel="coverage error (max)"):
    """One log-y line per method across experiments.

    `data` maps method -> {experiment_label: scalar metric}. All methods must
    share the same experiment labels (insertion order of the first method sets
    the x-axis order).
    """
    methods = _ordered(data)
    experiments = list(next(iter(data.values())).keys())
    x = np.arange(len(experiments))
    for m in methods:
        ys = [data[m][e] for e in experiments]
        st = style.METHOD_STYLE[m]
        ax.plot(x, ys, marker=st["marker"], color=st["color"], label=st["label"])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.set_ylabel(ylabel)
    if floor is not None:
        noise_floor_band(ax, floor)
    return ax


def metric_vs_budget(ax, series, *, xlabel="parameters", ylabel="coverage error (max)"):
    """One log-x line per method: metric vs total parameter budget.

    `series` maps method -> (budgets_array, metric_array).
    """
    methods = _ordered(series)
    for m in methods:
        budgets, metric = series[m]
        st = style.METHOD_STYLE[m]
        ax.plot(budgets, metric, marker=st["marker"], color=st["color"], label=st["label"])
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax
