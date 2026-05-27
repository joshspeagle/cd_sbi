import torch
from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.methods.nle import NLERunner
from cdsbi.reproducibility.seeding import seed_everything
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def _minimal_recipe():
    return {
        "lr": 1e-3,
        "batch_size": 32,
        "n_steps": 50,
        "n_train": 200,
        "fresh_batch": False,
    }


def test_nle_smoke(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    runner = NLERunner(flow=flow)
    trained = runner.fit(simulator=sim, config=_minimal_recipe(), seed=seed)
    assert trained.procedure is not None
    cs = trained.procedure.confidence_set(torch.tensor([[0.0]]), alpha=0.9)
    assert cs.boundary_repr.shape == (2,)
