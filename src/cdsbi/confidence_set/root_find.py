"""1D bisection root finder."""
from __future__ import annotations

import warnings
from typing import Callable


def bisect_1d(
    f: Callable[[float], float], a: float, b: float, tol: float = 1e-4, max_iter: int = 100
) -> float:
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        warnings.warn(
            f"bisect_1d: f({a})={fa}, f({b})={fb} same sign; returning midpoint.",
            RuntimeWarning,
        )
        return 0.5 * (a + b)
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) < tol:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)
