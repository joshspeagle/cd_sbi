"""EnergyCalibrationLoss: group energy score vs N(0,I); minimized at calibration."""
import torch
from cdsbi.flows.base import Guarantee


def _loss():
    from cdsbi.losses.energy_calibration import EnergyCalibrationLoss
    return EnergyCalibrationLoss(n_ref=512)


def test_required_guarantees_is_r1_only():
    assert _loss().required_guarantees == frozenset({Guarantee.R1})


def test_score_lower_for_calibrated_group():
    torch.manual_seed(0)
    loss = _loss()
    calibrated = torch.randn(1, 400, 2)
    misscaled = 2.5 * torch.randn(1, 400, 2)
    shifted = torch.randn(1, 400, 2) + torch.tensor([2.0, 2.0])
    s_cal = loss.score(calibrated, rng_seed=1)
    s_mis = loss.score(misscaled, rng_seed=1)
    s_shift = loss.score(shifted, rng_seed=1)
    assert s_cal < s_mis, f"calibrated {s_cal:.3f} should beat misscaled {s_mis:.3f}"
    assert s_cal < s_shift, f"calibrated {s_cal:.3f} should beat shifted {s_shift:.3f}"


def test_score_is_differentiable_wrt_r():
    loss = _loss()
    r = torch.randn(2, 64, 2, requires_grad=True)
    s = loss.score(r, rng_seed=0)
    s.backward()
    assert r.grad is not None and torch.isfinite(r.grad).all()


def test_score_averages_over_groups():
    loss = _loss()
    r = torch.randn(4, 50, 2)
    s = loss.score(r, rng_seed=0)
    assert s.ndim == 0
