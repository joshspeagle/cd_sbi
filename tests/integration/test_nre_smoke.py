import torch
import torch.nn as nn
from cdsbi.methods.nre import NRERunner, build_classifier_mlp
from cdsbi.reproducibility.seeding import seed_everything
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def test_nre_smoke(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    runner = NRERunner(classifier_hidden=16, classifier_depth=2)
    trained = runner.fit(simulator=sim, config={"n_train": 200, "n_epochs": 5}, seed=seed)
    assert trained.procedure is not None
    cs = trained.procedure.confidence_set(torch.tensor([[0.0]]), alpha=0.9)
    assert cs.boundary_repr.shape == (2,)


def test_nre_n_params_structured():
    runner = NRERunner(classifier_hidden=16, classifier_depth=2)
    d = runner.n_params()
    assert d["kind"] == "classifier"
    assert d["backbone"] == 0
    assert d["head"] > 0
    assert d["total"] == d["head"]


def test_build_classifier_mlp_param_count():
    mlp = build_classifier_mlp(input_dim=2, hidden=8, depth=2)
    # (2*8 + 8) + (8*8 + 8) + (8*1 + 1) = 24 + 72 + 9 = 105
    assert sum(p.numel() for p in mlp.parameters()) == 105
