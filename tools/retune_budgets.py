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
- classifier_hidden : build_classifier_mlp(input_dim=2, hidden=H, depth=2) — used by NRE.
- quantile_hidden   : 4 × build_classifier_mlp(input_dim=1, hidden=H, depth=2) — used by
  LF2I's calibration stage (v0 alpha_grid has 4 values).

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
from cdsbi.methods.nre import build_classifier_mlp

CANDIDATES: list[int] = list(range(4, 513, 2))
TARGETS: dict[str, int] = {
    "small": 1000,
    "medium": 5000,
    "large": 25000,
    "xlarge": 100000,
}


def cdsbi_flow_params(H: int) -> int:
    m = AdditiveFlow1D(hidden=H)
    return m.a.n_params() + m.b.n_params()


def maf_params(H: int) -> int:
    m = MAFAdapter(features=1, context_features=1, hidden=H, num_layers=2)
    return m.n_params()


def classifier_params(H: int) -> int:
    net = build_classifier_mlp(input_dim=2, hidden=H, depth=2)
    return sum(p.numel() for p in net.parameters())


def quantile_params(H: int, alpha_grid_len: int = 4) -> int:
    net = build_classifier_mlp(input_dim=1, hidden=H, depth=2)
    per_net = sum(p.numel() for p in net.parameters())
    return alpha_grid_len * per_net


DOMAINS: dict[str, object] = {
    "cdsbi_flow_hidden": cdsbi_flow_params,
    "maf_hidden": maf_params,
    "classifier_hidden": classifier_params,
    "quantile_hidden": quantile_params,
}


def pick_best(target: int, fn, candidates: list[int]) -> tuple[int, int, float]:
    """Return (best_H, actual_params, rel_err)."""
    counts = [(H, fn(H)) for H in candidates]
    best_H, best_n = min(counts, key=lambda hc: abs(hc[1] - target))
    rel_err = abs(best_n - target) / target
    return best_H, best_n, rel_err


def status_label(rel_err: float) -> str:
    if rel_err <= 0.10:
        return "matched"
    elif rel_err <= 0.15:
        return "matched_with_warning"
    return "unreachable"


def main() -> None:
    print(f"{'budget':<8}  {'domain':<20}  {'H':>5}  {'actual':>8}  {'target':>8}  {'rel_err':>8}  status")
    print("-" * 75)

    for budget_name, target in TARGETS.items():
        first = True
        for domain_key, fn in DOMAINS.items():
            H, n, err = pick_best(target, fn, CANDIDATES)
            label = status_label(err)
            prefix = f"{budget_name:<8}" if first else " " * 8
            first = False
            print(f"{prefix}  {domain_key:<20}  {H:>5}  {n:>8}  {target:>8}  {err:>7.1%}  {label}")
        print()

    print("\nYAML snippet (paste into configs/budget/<name>.yaml):")
    for budget_name, target in TARGETS.items():
        print(f"\n  # {budget_name}.yaml")
        print(f"  name: {budget_name}")
        print(f"  target_params: {target}")
        for domain_key, fn in DOMAINS.items():
            H, n, err = pick_best(target, fn, CANDIDATES)
            print(f"  {domain_key}: {H}  # actual={n} ({err:.1%})")


if __name__ == "__main__":
    main()
