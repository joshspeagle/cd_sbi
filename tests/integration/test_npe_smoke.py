import torch
from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.methods.npe import NPERunner
from cdsbi.reproducibility.seeding import seed_everything
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def test_npe_smoke(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    runner = NPERunner(flow=flow)
    trained = runner.fit(simulator=sim, config={"n_train": 200, "n_epochs": 5}, seed=seed)
    assert trained.procedure is not None
    cs = trained.procedure.confidence_set(torch.tensor([[0.0]]), alpha=0.9)
    assert cs.boundary_repr.shape == (2,)


def test_npe_n_params_structured():
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    runner = NPERunner(flow=flow)
    d = runner.n_params()
    assert d["kind"] == "flow"
    assert d["backbone"] > 0
    assert d["calibration_stage"] == 0
