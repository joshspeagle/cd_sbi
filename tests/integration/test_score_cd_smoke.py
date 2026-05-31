"""Score-CD smoke: both variants train an NLE flow + produce a CriticalValueProcedure
whose containment test runs and gives roughly-sane coverage."""
from __future__ import annotations

import numpy as np
import torch

from cdsbi.confidence_set.procedures import CriticalValueProcedure
from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.methods.score_cd import ScoreCDRunner
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def _recipe(**kw):
    base = {"lr": 1e-3, "batch_size": 64, "n_steps": 200, "n_train": 2000,
            "fresh_batch": False, "optimizer": "adam", "weight_decay": 0.0,
            "betas": [0.9, 0.999], "momentum": 0.9, "lr_schedule": "constant",
            "warmup_steps": 0, "lr_min_ratio": 0.0, "lr_gamma": 0.999,
            "batching": "random_replacement", "grad_clip_norm": 5.0,
            "n_train_quantile": 1000, "alpha_grid": [0.5, 0.9]}
    base.update(kw)
    return base


def _make_runner(variant):
    sim = LocationNormal1D()
    flow = MAFAdapter(features=int(sim.d_x), context_features=int(sim.d_theta),
                      hidden=16, num_layers=2)
    return sim, ScoreCDRunner(flow=flow, variant=variant, fisher_n=1000,
                              quantile_hidden=8, quantile_depth=2)


def test_score_cd_invalid_variant_raises():
    import pytest
    with pytest.raises(ValueError):
        ScoreCDRunner(flow=None, variant="bogus")


def test_score_cd_rao_smoke(seed):
    sim, runner = _make_runner("rao")
    trained = runner.fit(simulator=sim, config=_recipe(), seed=seed)
    assert isinstance(trained.procedure, CriticalValueProcedure)
    # containment runs and returns a bool mask over the X batch
    x = sim.sample_x_given_theta((0.0,), 400, np.random.default_rng(0))
    inside = trained.procedure.contains_batch((0.0,), x, 0.9)
    assert inside.shape == (400,)
    cov = float(inside.float().mean())
    assert 0.6 < cov < 1.0, f"90% Score-CD(rao) coverage implausible: {cov}"
    # the statistic is finite and the score path (autograd under no_grad) works
    t = trained.procedure.test_statistic(torch.zeros(5, 1), x[:5])
    assert torch.isfinite(t).all()


def test_score_cd_cal_smoke(seed):
    sim, runner = _make_runner("cal")
    trained = runner.fit(simulator=sim, config=_recipe(), seed=seed)
    assert isinstance(trained.procedure, CriticalValueProcedure)
    x = sim.sample_x_given_theta((0.0,), 400, np.random.default_rng(0))
    inside = trained.procedure.contains_batch((0.0,), x, 0.9)
    assert inside.shape == (400,)
    assert 0.5 < float(inside.float().mean()) <= 1.0
