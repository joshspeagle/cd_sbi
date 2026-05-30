"""FloorIntegrity: arm-aware floor check (NF-MLE vs other losses)."""
import math


class _Trained:
    def __init__(self, final_loss, loss_class):
        self.final_loss = final_loss
        self.arch_metadata = {"loss_class": loss_class}
        self.procedure = object()


class _SimWithFloor:
    def entropy_lower_bound(self, **kw): return 0.92


def test_floor_integrity_passes_at_floor():
    from cdsbi.diagnostics.floor_integrity import FloorIntegrity
    res = FloorIntegrity()(_Trained(0.90, "NFMLELoss"), _SimWithFloor())
    row = res.value.iloc[0]
    assert not bool(row["cheats"])
    assert res.passed
    assert abs(float(row["entropy_floor"]) - 0.92) < 1e-9


def test_floor_integrity_flags_cheat_below_floor():
    from cdsbi.diagnostics.floor_integrity import FloorIntegrity
    res = FloorIntegrity()(_Trained(-5.29, "NFMLELoss"), _SimWithFloor())
    assert bool(res.value.iloc[0]["cheats"])
    assert not res.passed


def test_floor_integrity_noop_for_non_nfmle_loss():
    from cdsbi.diagnostics.floor_integrity import FloorIntegrity
    res = FloorIntegrity()(_Trained(-100.0, "EnergyCalibrationLoss"), _SimWithFloor())
    row = res.value.iloc[0]
    assert not bool(row["cheats"])
    assert res.passed
    assert math.isnan(float(row["entropy_floor"]))
