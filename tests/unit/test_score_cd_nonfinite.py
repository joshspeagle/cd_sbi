"""Score-CD non-finite-score handling (hardening item 1).

The previous `nan_to_num(score, 0.0)` zeroed non-finite scores, which sent the
test statistic to 0 and swept those points INSIDE the confidence set — inflating
coverage exactly on the unstable (overfit-flow) runs where Score-CD is weakest.
The fix: drop non-finite rows from Fisher estimation, and reject non-finite
statistics (→ +inf → outside the set → counted against the method), with counts
exposed.
"""
import torch

from cdsbi.methods.score_cd import _finite_rows, _reject_nonfinite


def test_finite_rows_drops_nonfinite_for_fisher():
    U = torch.tensor(
        [[1.0, 2.0], [float("inf"), 0.0], [3.0, 4.0], [0.0, float("nan")], [5.0, 6.0]]
    )
    Uf, dropped = _finite_rows(U)
    assert dropped == 2
    assert torch.equal(Uf, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


def test_finite_rows_all_finite_keeps_all():
    U = torch.randn(7, 3)
    Uf, dropped = _finite_rows(U)
    assert dropped == 0 and torch.equal(Uf, U)


def test_reject_nonfinite_maps_to_plus_inf_and_counts():
    T = torch.tensor([0.5, float("inf"), 2.0, float("nan"), -float("inf")])
    Tc, n = _reject_nonfinite(T)
    assert n == 3
    assert Tc[0] == 0.5 and Tc[2] == 2.0          # finite untouched
    assert torch.isinf(Tc[1]) and Tc[1] > 0       # +inf preserved
    assert torch.isinf(Tc[3]) and Tc[3] > 0       # nan -> +inf
    assert torch.isinf(Tc[4]) and Tc[4] > 0       # -inf -> +inf (conservative)


def test_reject_noop_on_finite():
    T = torch.tensor([1.0, 2.0, 3.0])
    Tc, n = _reject_nonfinite(T)
    assert n == 0 and torch.equal(Tc, T)


def test_fix_is_conservative_where_old_guard_was_anti_conservative():
    """The crux: a non-finite score must EXCLUDE the point, not include it."""
    c = 3.841  # chi2_1 at 0.95
    nan_T = torch.tensor([float("nan")])

    # FIX: non-finite -> +inf -> (inf <= c) is False -> theta OUTSIDE the set
    Tc, n = _reject_nonfinite(nan_T)
    assert n == 1
    assert bool((Tc <= c)[0]) is False

    # OLD BUG: nan_to_num(score, 0) -> T=0 -> (0 <= c) is True -> theta INSIDE the set
    old = torch.nan_to_num(nan_T, nan=0.0)
    assert bool((old <= c)[0]) is True
