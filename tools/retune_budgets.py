"""retune_budgets.py — compute per-flow-type hidden widths closest to budget targets.

Purpose
-------
The four method families in this codebase (CDSBI, NPE/NLE/LF2I-stage-1, NRE,
LF2I-calibration) all have different parameter-count formulas at the same hidden
width, so a single ``flow_hidden`` field cannot produce matched budgets for all
of them simultaneously.  This script enumerates candidate widths for each
domain, picks the width whose actual n_params() is closest to each target, and
prints a summary table.

Usage
-----
    python tools/retune_budgets.py

The output is a summary table that can be copy-pasted into the budget YAML files
(``configs/budget/*.yaml``).  The script does NOT write those files; editing them
is a manual (or automated) follow-up step.

Domains
-------
- cdsbi_flow_hidden : AdditiveFlow1D backbone (two UMNNBlock MLPs; excludes the
  two log_alpha scalar parameters which add <4 params total).
- maf_hidden        : MAFAdapter(features=1, context_features=1, hidden=H, num_layers=2).
                      Shared by NPE / NLE / LF2I-stage-1.
- classifier_hidden : build_classifier_mlp(input_dim=2, hidden=H, depth=2) — used by NRE.
- quantile_hidden   : MultiQuantileMLP(input_dim=1, hidden=H, depth=2, n_quantiles=4)
                      — used by LF2I-BFF's calibration stage. Tuned against the
                      residual (target − classifier_params(classifier_hidden));
                      for budgets where the backbone already exceeds target, the
                      head is shrunk to the minimum candidate H.

Candidates
----------
All even integers from 4 to 512, inclusive.  The even-step grid gives a smooth
parameter trajectory without the large gaps in the standard {4,6,8,12,...} list.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cdsbi.flows.additive import AdditiveFlow1D
from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.flows.triangular_additive import TriangularAdditiveFlow
from cdsbi.methods.lf2i import MultiQuantileMLP
from cdsbi.methods.nre import build_classifier_mlp

CANDIDATES: list[int] = list(range(4, 513, 2))
TARGETS: dict[str, int] = {
    "small": 1000,
    "medium": 5000,
    "large": 25000,
    "xlarge": 100000,
}
# Canonical alpha_grid_len for budget bookkeeping. The multi-quantile head's
# parameter count varies only weakly with this (~n_q × (H+1)), so picking 4
# matches the manuscript's §8.1 default and stays close for any reasonable len.
CANONICAL_ALPHA_GRID_LEN: int = 4


def cdsbi_flow_params(H: int) -> int:
    m = AdditiveFlow1D(hidden=H)
    return m.a.n_params() + m.b.n_params()


def triangular_additive_2d_params(H: int) -> int:
    m = TriangularAdditiveFlow(d=2, hidden=H)
    return m.n_params()


def maf_params(H: int) -> int:
    m = MAFAdapter(features=1, context_features=1, hidden=H, num_layers=2)
    return m.n_params()


def classifier_params(H: int) -> int:
    net = build_classifier_mlp(input_dim=2, hidden=H, depth=2)
    return sum(p.numel() for p in net.parameters())


def multi_quantile_params(H: int, n_q: int = CANONICAL_ALPHA_GRID_LEN) -> int:
    net = MultiQuantileMLP(input_dim=1, hidden=H, depth=2, n_quantiles=n_q)
    return sum(p.numel() for p in net.parameters())


# Standalone-tunable domains (each fills the full budget alone).
STANDALONE_DOMAINS: dict[str, object] = {
    "cdsbi_flow_hidden": cdsbi_flow_params,
    "maf_hidden": maf_params,
    "classifier_hidden": classifier_params,
}


def pick_best(target: int, fn, candidates: list[int]) -> tuple[int, int, float]:
    """Return (best_H, actual_params, rel_err)."""
    counts = [(H, fn(H)) for H in candidates]
    best_H, best_n = min(counts, key=lambda hc: abs(hc[1] - target))
    rel_err = abs(best_n - target) / target
    return best_H, best_n, rel_err


def pick_quantile_residual(
    target: int, backbone_H: int, candidates: list[int]
) -> tuple[int, int, int, float]:
    """Tune the LF2I-BFF multi-quantile head against the residual budget.

    The LF2I-BFF backbone is the NRE-style classifier (shared with NRE);
    the head is a small multi-quantile MLP. Returns
    (q_H, q_params, lf2i_total, lf2i_rel_err).
    """
    backbone = classifier_params(backbone_H)
    residual = target - backbone
    if residual <= 0:
        # Backbone already saturates target; use smallest candidate for the head.
        q_H = candidates[0]
        q_params = multi_quantile_params(q_H)
        total = backbone + q_params
        return q_H, q_params, total, abs(total - target) / target
    # Pick the head H closest to the residual.
    q_H, q_params, _ = pick_best(residual, multi_quantile_params, candidates)
    total = backbone + q_params
    return q_H, q_params, total, abs(total - target) / target


def status_label(rel_err: float) -> str:
    if rel_err <= 0.10:
        return "matched"
    elif rel_err <= 0.15:
        return "matched_with_warning"
    return "unreachable"


def main() -> None:
    print(f"{'budget':<8}  {'domain':<24}  {'H':>5}  {'actual':>8}  {'target':>8}  {'rel_err':>8}  status")
    print("-" * 79)

    for budget_name, target in TARGETS.items():
        first = True
        for domain_key, fn in STANDALONE_DOMAINS.items():
            H, n, err = pick_best(target, fn, CANDIDATES)
            label = status_label(err)
            prefix = f"{budget_name:<8}" if first else " " * 8
            first = False
            print(f"{prefix}  {domain_key:<24}  {H:>5}  {n:>8}  {target:>8}  {err:>7.1%}  {label}")
        # v1 d=2 variant of the same domain — for paper §8.2.
        H_d2, n_d2, e_d2 = pick_best(target, triangular_additive_2d_params, CANDIDATES)
        print(f"{'':<8}  {'cdsbi_flow_hidden (d=2)':<24}  {H_d2:>5}  {n_d2:>8}  "
              f"{target:>8}  {e_d2:>7.1%}  {status_label(e_d2)}")
        # LF2I-BFF composite: classifier backbone (shared with NRE) + quantile head.
        cls_H = pick_best(target, classifier_params, CANDIDATES)[0]
        q_H, q_n, lf2i_total, lf2i_err = pick_quantile_residual(target, cls_H, CANDIDATES)
        print(f"{'':<8}  {'quantile_hidden (BFF)':<24}  {q_H:>5}  {q_n:>8}  "
              f"{max(target - classifier_params(cls_H), 0):>8}  {'':>7}  "
              f"head-only fits residual")
        print(f"{'':<8}  {'  └ LF2I-BFF total':<24}  {'':>5}  {lf2i_total:>8}  {target:>8}  "
              f"{lf2i_err:>7.1%}  {status_label(lf2i_err)}")
        print()

    print("\nYAML snippet (paste into configs/budget/<name>.yaml):")
    for budget_name, target in TARGETS.items():
        print(f"\n  # {budget_name}.yaml")
        print(f"  name: {budget_name}")
        print(f"  target_params: {target}")
        for domain_key, fn in STANDALONE_DOMAINS.items():
            H, n, err = pick_best(target, fn, CANDIDATES)
            print(f"  {domain_key}: {H}  # actual={n} ({err:.1%})")
        H_d2, n_d2, e_d2 = pick_best(target, triangular_additive_2d_params, CANDIDATES)
        print(f"  cdsbi_flow_hidden_d2: {H_d2}  # TriangularAdditiveFlow(d=2) actual={n_d2} ({e_d2:.1%})")
        cls_H = pick_best(target, classifier_params, CANDIDATES)[0]
        q_H, q_n, lf2i_total, lf2i_err = pick_quantile_residual(target, cls_H, CANDIDATES)
        print(f"  quantile_hidden: {q_H}  # LF2I-BFF multi-quantile head actual={q_n}; "
              f"LF2I-BFF total={lf2i_total} ({lf2i_err:.1%}, {status_label(lf2i_err)})")


if __name__ == "__main__":
    main()
