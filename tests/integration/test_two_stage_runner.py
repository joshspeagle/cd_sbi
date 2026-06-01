# tests/integration/test_two_stage_runner.py
import numpy as np
import torch
from cdsbi.conditioners.moment_regression import MomentRegressionConditioner
from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
from cdsbi.methods.cd_sbi_two_stage import TwoStageCDSBIRunner
from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar


def _make_runner():
    sim = NormalUnknownMeanVar()
    cond = MomentRegressionConditioner(n_iid=sim.n_iid, d_theta=sim.d_theta, target="theta")
    flow = SingleIndexMonotoneFlow(d=sim.d_theta, theta_signs=sim.theta_signs,
                                   feat_signs=sim.feat_signs, hidden=32)
    return sim, TwoStageCDSBIRunner(flow=flow, conditioner=cond, device="cpu")


def test_stage1_regression_recovers_theta_including_log_sigma():
    """The frozen regression summary predicts BOTH coords — crucially log σ (the
    coord M2 collapses). This is the structural no-collapse property."""
    sim, runner = _make_runner()
    cfg = {"lr": 2e-3, "stage1_steps": 1500, "stage2_steps": 0,
           "batch_size": 512, "fresh_batch": True}
    runner.fit(sim, cfg, seed=0)
    rng = np.random.default_rng(1)
    theta, x = sim.sample(4000, rng)
    with torch.no_grad():
        h, _ = runner.conditioner.encode(x)
    # corr of predicted vs true for each coord (coord 0 = log σ, coord 1 = μ)
    for k, name in [(0, "log_sigma"), (1, "mu")]:
        c = np.corrcoef(h[:, k].numpy(), theta[:, k].numpy())[0, 1]
        assert c > 0.8, f"summary failed to recover {name}: corr {c:.3f}"


def test_stage1_freezes_conditioner():
    """After Stage 1, conditioner params have requires_grad=False (cannot be moved
    by Stage 2's loss — the structural anti-collapse guarantee)."""
    sim, runner = _make_runner()
    cfg = {"lr": 2e-3, "stage1_steps": 50, "stage2_steps": 0,
           "batch_size": 256, "fresh_batch": True}
    runner.fit(sim, cfg, seed=0)
    assert all(not p.requires_grad for p in runner.conditioner.parameters())


import math
from scipy.stats import chi2


def test_full_two_stage_calibrates():
    """End-to-end: regress→freeze→NF-MLE yields a pivot calibrated at the truth
    (coverage ≈ α) — validity (Thm 1)."""
    sim = NormalUnknownMeanVar()
    cond = MomentRegressionConditioner(n_iid=sim.n_iid, d_theta=sim.d_theta, target="theta")
    flow = SingleIndexMonotoneFlow(d=sim.d_theta, theta_signs=sim.theta_signs,
                                   feat_signs=sim.feat_signs, hidden=32)
    runner = TwoStageCDSBIRunner(flow=flow, conditioner=cond, device="cpu")
    cfg = {"lr": 1e-3, "stage1_steps": 1500, "stage2_steps": 3000,
           "batch_size": 512, "fresh_batch": True}
    trained = runner.fit(sim, cfg, seed=0)
    proc = trained.procedure
    rng = np.random.default_rng(2)
    theta0 = np.array([math.log(1.0), 0.0])
    xv = sim.sample_x_given_theta(theta0, 20000, rng)
    th = torch.tensor(theta0, dtype=torch.float32).expand(20000, -1)
    r = proc.pivot_fn(th, xv)
    sq = (r ** 2).sum(-1).detach().numpy()
    for a in (0.8, 0.9):
        emp = float((sq <= chi2.ppf(a, df=2)).mean())
        assert abs(emp - a) < 0.06, f"miscovers at α={a}: {emp:.3f}"
