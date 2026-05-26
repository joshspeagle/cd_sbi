import torch
from cdsbi.conditioners.identity import Identity
from cdsbi.flows.additive import AdditiveFlow1D
from cdsbi.losses.nfmle import NFMLELoss
from cdsbi.methods.cd_sbi import CDSBIRunner
from cdsbi.reproducibility.seeding import seed_everything
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def test_cdsbi_smoke_trains_briefly(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    flow = AdditiveFlow1D(hidden=8)
    runner = CDSBIRunner(flow=flow, conditioner=Identity(), loss=NFMLELoss())
    cfg = {"n_train": 200, "batch_size": 50, "lr": 3e-3, "n_steps": 50}
    trained = runner.fit(simulator=sim, config=cfg, seed=seed)
    assert trained.final_loss < 5.0
    assert trained.n_steps == 50
    assert trained.procedure is not None


def test_cdsbi_n_params_structured():
    flow = AdditiveFlow1D(hidden=8)
    runner = CDSBIRunner(flow=flow, conditioner=Identity(), loss=NFMLELoss())
    np_dict = runner.n_params()
    assert np_dict["kind"] == "flow"
    assert np_dict["total"] == np_dict["backbone"] + np_dict["head"] + np_dict["calibration_stage"]
    assert np_dict["head"] == 2  # the two α scalars
    assert np_dict["calibration_stage"] == 0
