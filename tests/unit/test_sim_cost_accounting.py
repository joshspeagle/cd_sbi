"""Hardening item 7: simulator-call accounting.

The "matched budget" matches PARAMETERS only. Two-stage methods consume extra
calibration draws and Score-CD-rao draws fisher_n fresh sims per distinct
queried θ at inference — these must be counted and logged so the budget label
is honest.
"""
import numpy as np
import torch
from omegaconf import OmegaConf

from cdsbi.experiments.run import _sim_cost_accounting
from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.methods.score_cd import ScoreCDRunner
from cdsbi.simulators.location_gauss_2d_iid import LocationGaussian2D_iid


class _Trained:
    def __init__(self, proc):
        self.procedure = proc


class _Proc:
    pass


def _cfg(**method):
    return OmegaConf.create({"method": method})


def test_train_calls_fixed_set_vs_fresh():
    fit = {"n_steps": 100, "batch_size": 32, "n_train": 5000, "fresh_batch": False}
    out = _sim_cost_accounting(_cfg(), "nle", fit, _Trained(_Proc()), 2)
    assert out["sim_calls_train"] == 5000
    fit["fresh_batch"] = True
    out = _sim_cost_accounting(_cfg(), "nle", fit, _Trained(_Proc()), 2)
    assert out["sim_calls_train"] == 100 * 32


def test_two_stage_calibration_counted():
    fit = {"n_steps": 1, "batch_size": 1, "n_train": 10000, "fresh_batch": False,
           "n_train_quantile": 5000}
    out = _sim_cost_accounting(_cfg(), "score_cd_cal", fit, _Trained(_Proc()), 2)
    assert out["sim_calls_calibration"] == 5000
    assert out["sim_calls_total_method"] == 15000


def test_lf2i_bff_marginal_d_aware_floor():
    fit = {"n_steps": 1, "batch_size": 1, "n_train": 10000, "fresh_batch": False,
           "n_train_quantile": 5000}
    # configured 64 with d=5 -> floor lifts it to 640 (mirrors lf2i_bff.py)
    out = _sim_cost_accounting(_cfg(marginal_n=64), "lf2i_bff", fit, _Trained(_Proc()), 5)
    assert out["sim_calls_calibration"] == 5000 + 128 * 5
    # configured 2048 with d=2 -> configured wins
    out = _sim_cost_accounting(_cfg(marginal_n=2048), "lf2i_bff", fit, _Trained(_Proc()), 2)
    assert out["sim_calls_calibration"] == 5000 + 2048


def test_inference_counter_picked_up():
    proc = _Proc()
    proc.inference_sim_calls = {"fisher": 12000}
    fit = {"n_steps": 1, "batch_size": 1, "n_train": 10, "fresh_batch": False}
    out = _sim_cost_accounting(_cfg(), "score_cd_rao", fit, _Trained(proc), 2)
    assert out["sim_calls_inference"] == 12000
    assert out["sim_calls_total_method"] == 10 + 12000


def test_score_cd_rao_fisher_counter_exact_mode():
    """Exact per-θ MC mode (fisher_grid_per_dim=0): querying the statistic at
    two distinct θ draws 2×fisher_n; a cached θ adds nothing."""
    torch.manual_seed(0)
    sim = LocationGaussian2D_iid()
    flow = MAFAdapter(features=2, context_features=2, hidden=8, num_layers=2)
    runner = ScoreCDRunner(flow, variant="rao", fisher_n=64,
                           fisher_grid_per_dim=0, device="cpu")
    cfg = dict(optimizer="adam", lr=3e-3, batch_size=64, n_steps=20, n_train=256,
               fresh_batch=False, grad_clip_norm=5.0, alpha_grid=[0.9])
    tm = runner.fit(sim, cfg, seed=0)
    proc = tm.procedure
    x = torch.randn(8, 2)
    proc.test_statistic(torch.zeros(8, 2), x)
    assert proc.inference_sim_calls["fisher"] == 64
    proc.test_statistic(torch.ones(8, 2), x)
    assert proc.inference_sim_calls["fisher"] == 128
    proc.test_statistic(torch.zeros(8, 2), x)  # cache hit — no new draws
    assert proc.inference_sim_calls["fisher"] == 128


def test_score_cd_rao_fisher_counter_grid_mode_default():
    """Grid mode (the default): the full Fisher cost is paid at FIT
    (n_per_dim^d × fisher_n) and queries draw nothing."""
    torch.manual_seed(0)
    sim = LocationGaussian2D_iid()
    flow = MAFAdapter(features=2, context_features=2, hidden=8, num_layers=2)
    runner = ScoreCDRunner(flow, variant="rao", fisher_n=64,
                           fisher_grid_per_dim=3, device="cpu")
    cfg = dict(optimizer="adam", lr=3e-3, batch_size=64, n_steps=20, n_train=256,
               fresh_batch=False, grad_clip_norm=5.0, alpha_grid=[0.9])
    tm = runner.fit(sim, cfg, seed=0)
    proc = tm.procedure
    assert proc.inference_sim_calls["fisher"] == 3 * 3 * 64
    x = torch.randn(8, 2)
    proc.test_statistic(torch.zeros(8, 2), x)
    proc.test_statistic(torch.full((8, 2), 0.123), x)
    assert proc.inference_sim_calls["fisher"] == 3 * 3 * 64  # zero at query
