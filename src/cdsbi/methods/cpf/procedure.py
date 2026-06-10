"""ConformalPivotProcedure — super-level sets of the calibrated p-field (plan step 2).

The CPF confidence set at confidence level `alpha` (repo convention: alpha = the
confidence level) is the super-level set of the calibrated field

    C_alpha(x) = { theta : s(theta; x) <= q_hat_alpha },
    s(theta; x) = F_hat_{T|theta}(T(theta; x)) in [0, 1],

with q_hat_alpha the GLOBAL split-conformal threshold (theta-independent constant;
cpf/conformal.py). Delegates set construction to `CriticalValueProcedure` with
`test_stat_fn = s` and a constant critical value — `contains_batch` already handles
a 0-dim critical value, and the d>1 ray-bisection tolerates non-radial fields.

DISPATCH NOTE (plan step 5 fix): `CriticalValueProcedure` already exposes
`test_statistic`, so `evaluate_coverage`'s legacy `has_stat` branch would match
this class silently. The marker `is_conformal_pivot = True` (and the subclass
identity) exists so the engine's dedicated CPF branch can be selected BEFORE the
`has_stat` test; the pivot chi^2-KS diagnostics must NOT fire for this procedure
(the field is not a pivot norm).

CAVEAT (plan step 2 test guidance): the inherited 1D `confidence_set` grid-scans
then bisects `s - q_hat`; a step/plateau-shaped field makes the bisection boundary
ill-defined. Membership queries (`contains_batch`) are exact regardless — tests
must compare against the directly-computed super-level set, not the bisection
boundary.
"""
from __future__ import annotations

from typing import Callable

import torch

from cdsbi.confidence_set.procedures import CriticalValueProcedure
from cdsbi.methods.cpf.conformal import GlobalConformal


class ConformalPivotProcedure(CriticalValueProcedure):
    is_conformal_pivot = True

    def __init__(
        self,
        score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        conformal: GlobalConformal,
        d_theta: int,
        theta_range: tuple = (-20.0, 20.0),
    ):
        """score_fn(theta_rows, x) -> (B,) calibrated scores in [0, 1] (large =
        extreme), i.e. the FROZEN F_hat_{T|theta}(T(theta; x)) — F_hat conditioned
        on the HYPOTHESIZED theta (spec §4.3.1), x entering only through T.
        `conformal` is fit on a disjoint split with score_fn frozen."""
        self.conformal = conformal

        def critical_value_fn(theta: torch.Tensor, alpha: float) -> torch.Tensor:
            # 0-dim, on the caller's device/dtype (contains_batch compares
            # against the statistic's device; new_tensor follows theta).
            return theta.new_tensor(conformal.threshold(alpha))

        super().__init__(
            test_stat_fn=score_fn,
            critical_value_fn=critical_value_fn,
            d_theta=d_theta,
            theta_range=theta_range,
        )
