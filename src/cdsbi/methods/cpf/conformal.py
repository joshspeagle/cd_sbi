"""Global split-conformal recalibration of the CPF p-field (spec §4.3.3, plan step 4).

Convention — PINNED per the 2026-06-10 audit (guards the classic off-by-(1-coverage)
tail bug): the nonconformity score is

    s = F_hat_{T|theta}(T(theta; x))  in [0, 1],   LARGE = extreme,

evaluated on calibration pairs (theta_j, x_j) ~ proposal x simulator AT THE TRUTH
(s_j = F_hat_{T|theta_j}(T(theta_j; x_j))). For target coverage `coverage` (the repo's
`alpha` = confidence level), the threshold is the

    ceil(coverage * (n + 1))-th smallest

of the n calibration scores (Vovk split conformal), +inf if that index exceeds n.
The confidence set keeps {theta : s <= q_hat}. Worked example: n = 9, coverage = 0.9
-> ceil(0.9 * 10) = 9 -> q_hat = the 9th smallest (= largest) of the 9 scores.

Guarantee (exchangeable, continuous scores): P(s_test <= q_hat) lies in
[coverage, coverage + 1/(n+1)] — marginal over the joint (theta, x) draw from the
calibration proposal ("local-marginal-under-pi", spec §3.1/§6). A single global
q_hat per coverage level is one threshold on the fixed field s(theta; x), so the
resulting sets are nested across coverage levels by construction (spec §4.3.2).
"""
from __future__ import annotations

import math

import torch


def conformal_quantile(scores: torch.Tensor, coverage: float) -> float:
    """Vovk split-conformal threshold: ceil(coverage*(n+1))-th smallest score.

    Returns +inf when the index exceeds n (the set is everything — the honest
    answer at very small n / very high coverage). `coverage` is the confidence
    level (the repo's `alpha`), e.g. 0.95.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")
    flat = scores.detach().reshape(-1)
    n = flat.numel()
    if n == 0:
        raise ValueError("conformal_quantile needs at least one calibration score")
    # Float-noise guard: coverage*(n+1) can land an ulp above an exact integer
    # (e.g. 0.9*10); snap to 9 decimals before ceil so the worked examples hold.
    k = math.ceil(round(coverage * (n + 1), 9))
    if k > n:
        return float("inf")
    return float(torch.sort(flat).values[k - 1])


class GlobalConformal:
    """Holds the calibration scores; serves the per-coverage global threshold.

    Built once on split B (frozen statistic, disjoint from flow training — the
    freeze-before-calibrate firewall, spec §5). Thresholds are monotone in the
    coverage level (order statistics are monotone in k), so super-level sets of
    the fixed field are nested across coverage levels.
    """

    def __init__(self, calibration_scores: torch.Tensor):
        flat = calibration_scores.detach().reshape(-1)
        if flat.numel() == 0:
            raise ValueError("GlobalConformal needs at least one calibration score")
        self._sorted = torch.sort(flat).values
        self._cache: dict = {}

    @property
    def n(self) -> int:
        return int(self._sorted.numel())

    def threshold(self, coverage: float) -> float:
        if coverage not in self._cache:
            self._cache[coverage] = conformal_quantile(self._sorted, coverage)
        return self._cache[coverage]
