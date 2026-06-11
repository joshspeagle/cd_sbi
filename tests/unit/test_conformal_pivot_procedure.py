"""CPF plan step 2 — ConformalPivotProcedure (super-level sets of the p-field).

Per the plan's review fixes: membership is asserted against the DIRECTLY-computed
super-level set {theta : s(theta;x) <= q_hat} — never against the inherited 1D
bisection boundary (ill-defined on plateaus) — and the step/plateau field case is
tested explicitly. x batches are held in variables for the duration of each test
(the contains_batch cache keys on id(x_batch); letting tensors die and recycling
ids with the same theta_0 could stale-hit).
"""
import torch

from cdsbi.confidence_set.procedures import CriticalValueProcedure
from cdsbi.methods.cpf.conformal import GlobalConformal
from cdsbi.methods.cpf.procedure import ConformalPivotProcedure


def _smooth_score(theta, x):
    # s = |theta - x| / 10, clamped to [0,1]: monotone field, set = {|t-x| <= 10q}.
    return (theta[:, 0] - x[:, 0]).abs().div(10.0).clamp(0.0, 1.0)


def _step_score(theta, x):
    # Piecewise-constant (plateau) field: s = floor(|theta - x|) / 10.
    return (theta[:, 0] - x[:, 0]).abs().floor().div(10.0).clamp(0.0, 1.0)


def _make_procedure(score_fn):
    # 9 calibration scores, max 0.3 -> threshold(0.9) = 0.3 (9th smallest of 9);
    # threshold(0.5) = 5th smallest = midpoint 0.165.
    conformal = GlobalConformal(torch.linspace(0.03, 0.3, 9))
    return ConformalPivotProcedure(
        score_fn=score_fn, conformal=conformal, d_theta=1, theta_range=(-20.0, 20.0)
    )


def test_marker_and_delegation():
    proc = _make_procedure(_smooth_score)
    assert proc.is_conformal_pivot is True
    assert isinstance(proc, CriticalValueProcedure)  # reuses set construction


def test_membership_matches_direct_superlevel_set():
    proc = _make_procedure(_smooth_score)
    x = torch.tensor([[0.0], [1.0], [-2.0], [5.0]])
    q = proc.conformal.threshold(0.9)  # = 0.3 -> {|theta - x| <= 3}
    for theta0 in (-4.0, -2.5, 0.0, 2.5, 2.99, 3.01, 4.0, 8.5):
        got = proc.contains_batch(theta0, x, alpha=0.9)
        want = _smooth_score(torch.full((4, 1), theta0), x) <= q
        assert torch.equal(got, want), f"mismatch at theta0={theta0}"


def test_plateau_field_membership_exact():
    proc = _make_procedure(_step_score)
    x = torch.tensor([[0.0], [0.4]])
    q = proc.conformal.threshold(0.9)
    for theta0 in (-3.6, -1.0, 0.0, 0.5, 1.0, 2.5, 3.0, 3.5, 4.0, 7.0):
        got = proc.contains_batch(theta0, x, alpha=0.9)
        want = _step_score(torch.full((2, 1), theta0), x) <= q
        assert torch.equal(got, want), f"mismatch at theta0={theta0}"


def test_sets_nested_across_alpha():
    proc = _make_procedure(_smooth_score)
    x = torch.tensor([[0.0]])
    grid = torch.linspace(-6.0, 6.0, 121)
    for theta0 in grid.tolist():
        in_50 = bool(proc.contains_batch(theta0, x, alpha=0.5)[0])
        in_90 = bool(proc.contains_batch(theta0, x, alpha=0.9)[0])
        assert (not in_50) or in_90  # C_0.5 subset of C_0.9


def test_confidence_set_1d_interior_points_only():
    # Sanity on the inherited 1D set construction, evaluated AWAY from the
    # boundary (the bisection boundary itself is not under test).
    proc = _make_procedure(_smooth_score)
    x = torch.tensor([[0.0]])
    cs = proc.confidence_set(x, alpha=0.9)  # set ~ {|theta| <= 3}
    assert cs.contains(0.0) and cs.contains(2.5) and cs.contains(-2.5)
    assert not cs.contains(3.5) and not cs.contains(-3.5)


def test_threshold_inf_means_everything():
    # Tiny calibration set at high coverage -> q_hat = +inf -> all theta in set.
    conformal = GlobalConformal(torch.tensor([0.1, 0.2, 0.3]))
    proc = ConformalPivotProcedure(
        score_fn=_smooth_score, conformal=conformal, d_theta=1
    )
    x = torch.tensor([[0.0]])
    for theta0 in (-15.0, 0.0, 15.0):
        assert bool(proc.contains_batch(theta0, x, alpha=0.9)[0])
