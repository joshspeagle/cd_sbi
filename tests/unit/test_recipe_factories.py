"""Tests for v0.4 training recipe factories: optimizer, LR schedule, batching, dropout/layer_norm."""
import pytest
import torch
from cdsbi.conditioners.identity import Identity
from cdsbi.flows.additive import AdditiveFlow1D
from cdsbi.losses.nfmle import NFMLELoss
from cdsbi.methods.cd_sbi import CDSBIRunner
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def _base_cfg(**overrides):
    cfg = {
        "lr": 3e-3,
        "batch_size": 32,
        "n_steps": 10,
        "n_train": 100,
        "fresh_batch": True,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "momentum": 0.9,
        "lr_schedule": "constant",
        "warmup_steps": 0,
        "lr_min_ratio": 0.0,
        "lr_gamma": 0.999,
        "batching": "random_replacement",
        "grad_clip_norm": 5.0,
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.parametrize("opt", ["adam", "adamw", "sgd"])
def test_optimizer_factory(seed, opt):
    runner = CDSBIRunner(flow=AdditiveFlow1D(hidden=8), conditioner=Identity(), loss=NFMLELoss())
    trained = runner.fit(LocationNormal1D(), config=_base_cfg(optimizer=opt), seed=seed)
    assert torch.isfinite(torch.tensor(trained.final_loss))


@pytest.mark.parametrize("sched", ["constant", "cosine", "warmup_cosine", "exponential"])
def test_lr_schedule_factory(seed, sched):
    runner = CDSBIRunner(flow=AdditiveFlow1D(hidden=8), conditioner=Identity(), loss=NFMLELoss())
    cfg = _base_cfg(lr_schedule=sched, warmup_steps=2)
    trained = runner.fit(LocationNormal1D(), config=cfg, seed=seed)
    assert torch.isfinite(torch.tensor(trained.final_loss))


def test_batching_shuffle_epoch(seed):
    runner = CDSBIRunner(flow=AdditiveFlow1D(hidden=8), conditioner=Identity(), loss=NFMLELoss())
    cfg = _base_cfg(fresh_batch=False, batching="shuffle_epoch")
    trained = runner.fit(LocationNormal1D(), config=cfg, seed=seed)
    assert torch.isfinite(torch.tensor(trained.final_loss))


def test_dropout_and_layer_norm(seed):
    from cdsbi.flows.umnn import UMNNBlock
    block = UMNNBlock(context_dim=0, hidden=8, dropout=0.1, layer_norm=True)
    z = torch.randn(5, 1)
    g = block(z)
    assert g.shape == (5, 1)
    j = block.jacobian_factor(z)
    assert (j > 0).all()
