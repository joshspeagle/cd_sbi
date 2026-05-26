import torch
from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.methods.lf2i import LF2IRunner
from cdsbi.reproducibility.seeding import seed_everything
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def test_lf2i_smoke(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    runner = LF2IRunner(stat_flow=flow, quantile_hidden=16, quantile_depth=2)
    trained = runner.fit(
        simulator=sim,
        config={"n_train_stat": 200, "n_train_quantile": 200, "n_epochs_stat": 5, "n_epochs_quantile": 100, "alpha_grid": [0.5, 0.68, 0.9, 0.95]},
        seed=seed,
    )
    assert trained.procedure is not None
    cs = trained.procedure.confidence_set(torch.tensor([[0.0]]), alpha=0.9)
    assert cs.boundary_repr.shape == (2,)


def test_lf2i_n_params_structured():
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    runner = LF2IRunner(stat_flow=flow, quantile_hidden=16, quantile_depth=2)
    d = runner.n_params()
    assert d["kind"] == "two_stage"
    assert d["backbone"] > 0
    assert d["calibration_stage"] > 0
    assert d["head"] == 0
    assert d["total"] == d["backbone"] + d["calibration_stage"]
