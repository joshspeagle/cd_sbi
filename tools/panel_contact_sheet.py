"""Render every F1 panel with synthetic data onto one contact sheet.

A developer/agent verification aid (NOT a manuscript figure): run it, then
view the PNG to confirm each primitive renders legibly with the shared style.

    python tools/panel_contact_sheet.py [output_path]

Defaults to ./panel_contact_sheet.png (gitignored).
"""
from __future__ import annotations

import sys

import numpy as np

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures import panels


def build():
    style.apply_style()
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes = axes.ravel()

    panels.pit_histogram(axes[0], rng.uniform(0, 1, 1000), bins=20)
    axes[0].set_title("pit_histogram")

    nominal = np.array([0.5, 0.68, 0.9, 0.95, 0.99])
    panels.diagonal_reference(axes[1], lo=0.5, hi=1.0)
    panels.coverage_curve(axes[1], nominal, nominal + rng.normal(0, 0.01, 5),
                          color=style.METHOD_STYLE["cd_sbi"]["color"], marker="o",
                          label="CD-SBI")
    axes[1].set_title("coverage_curve + diagonal")

    theta0 = np.linspace(-2, 2, 5)
    alpha = np.array([0.5, 0.68, 0.9, 0.95, 0.99])
    panels.coverage_tile(axes[2], theta0, alpha, np.abs(rng.normal(0, 0.02, (5, 5))))
    axes[2].set_title("coverage_tile")

    panels.boxplot_per_method(axes[3], {
        "cd_sbi": rng.normal(0.025, 0.003, 20),
        "lf2i_bff": rng.normal(0.07, 0.01, 20),
        "nle": rng.normal(0.11, 0.02, 20),
        "npe": rng.normal(0.08, 0.02, 20),
        "nre": rng.normal(0.15, 0.03, 20),
    })
    axes[3].set_title("boxplot_per_method")

    panels.cross_method_summary_log_y(axes[4], {
        "cd_sbi": {"8.1": 0.025, "8.2": 0.025, "8.3": 0.025, "8.4": 0.031},
        "lf2i_bff": {"8.1": 0.06, "8.2": 0.11, "8.3": 0.12, "8.4": 0.078},
        "nle": {"8.1": 0.025, "8.2": 0.08, "8.3": 0.11, "8.4": 0.21},
    }, floor=0.02)
    axes[4].legend(fontsize=6)
    axes[4].set_title("cross_method_summary_log_y")

    panels.metric_vs_budget(axes[5], {
        "cd_sbi": (np.array([1e3, 5e3, 25e3, 1e5]), np.array([0.03, 0.025, 0.025, 0.025])),
        "npe": (np.array([1e3, 5e3, 25e3, 1e5]), np.array([0.07, 0.06, 0.05, 0.05])),
    })
    axes[5].set_title("metric_vs_budget")

    steps = np.arange(200)
    panels.loss_trajectory_with_floor(axes[6], {
        "R1+R2": 0.99 + 0.4 * np.exp(-steps / 40),
        "R1 only": 1.40 + 0.0 * steps,
    }, floor=0.99)
    axes[6].legend(fontsize=6)
    axes[6].set_title("loss_trajectory_with_floor")

    panels.loss_bar_with_floor(axes[7], {
        "R1+R2": np.array([0.986, 0.985, 0.983, 0.983]),
        "R1 only": np.array([1.419, 1.406, 1.391, 1.372]),
    }, floor=0.99, group_labels=["S", "M", "L", "XL"])
    axes[7].legend(fontsize=6)
    axes[7].set_title("loss_bar_with_floor")

    L_inv = np.array([[1.0, 0.0], [-0.4, 0.9]])
    panels.jacobian_recovery_scatter(axes[8], L_inv + rng.normal(0, 0.02, (2, 2)), L_inv)
    axes[8].set_title("jacobian_recovery_scatter")

    panels.position_table_as_axes(axes[9],
        ["Target", "1-stage", "Coverage"],
        ["CD-SBI", "NPE", "LF2I"],
        [["CD", "yes", "yes"], ["posterior", "yes", "no"], ["set", "no", "yes"]])
    axes[9].set_title("position_table_as_axes")

    axes[10].axis("off")
    axes[11].axis("off")
    fig.tight_layout()
    return fig


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "panel_contact_sheet.png"
    fig = build()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
