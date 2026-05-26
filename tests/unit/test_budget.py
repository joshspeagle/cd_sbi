import pytest
from cdsbi.methods.budget import build_from_budget, BudgetUnreachableError


def test_build_from_budget_picks_closest_within_tolerance():
    def n_params_for_width(w: int) -> int:
        return 10 * w * w
    widths = [1, 2, 4, 8, 16, 32]
    # target 170: width 4 → 160 params (5.9% off) → matched
    chosen, info = build_from_budget(170, n_params_for_width, widths)
    assert chosen == 4
    assert info["actual_params"] == 160
    assert info["status"] == "matched"


def test_build_from_budget_warning_band():
    def n_params_for_width(w: int) -> int:
        return 10 * w * w
    widths = [4, 8]  # 160 or 640
    # target 200: 160 is 20% off ⇒ outside ±15% ⇒ raise
    with pytest.raises(BudgetUnreachableError):
        build_from_budget(200, n_params_for_width, widths)


def test_build_from_budget_within_warning_band():
    def n_params_for_width(w: int) -> int:
        return 10 * w * w
    widths = [4, 8]  # 160 or 640
    # target 180: 160 is 11.1% off (between 10% and 15%) ⇒ matched_with_warning
    chosen, info = build_from_budget(180, n_params_for_width, widths)
    assert info["status"] == "matched_with_warning"
