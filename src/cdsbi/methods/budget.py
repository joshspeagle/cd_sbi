"""Closest-candidate width enumeration for matched-budget config."""
from __future__ import annotations

import warnings
from typing import Callable, List, Tuple

from cdsbi.methods.base import BudgetUnreachableError


def build_from_budget(
    target_params: int,
    n_params_fn: Callable[[int], int],
    candidates: List[int],
) -> Tuple[int, dict]:
    """Pick the width whose `n_params_fn(width)` is closest to `target_params`.

    Status tiers:
    - within ±10%: 'matched'
    - within ±15%: 'matched_with_warning' (warning emitted)
    - outside ±15%: BudgetUnreachableError
    """
    counts = [(w, n_params_fn(w)) for w in candidates]
    best_width, best_count = min(counts, key=lambda wc: abs(wc[1] - target_params))
    rel_err = abs(best_count - target_params) / target_params

    if rel_err <= 0.10:
        status = "matched"
    elif rel_err <= 0.15:
        status = "matched_with_warning"
        warnings.warn(
            f"Budget {target_params} matched at {best_count} (width={best_width}, "
            f"rel_err={rel_err:.1%}) — within ±15% but outside ±10%.",
            RuntimeWarning,
        )
    else:
        raise BudgetUnreachableError(
            f"No width in {candidates} lands within ±15% of {target_params} params. "
            f"Closest: width={best_width} ⇒ {best_count} params ({rel_err:.1%} off)."
        )
    return best_width, {"actual_params": best_count, "status": status, "rel_err": rel_err}
