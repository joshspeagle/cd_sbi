"""Intensive: (μ,Σ) Stage-A — recovery + covariance/μ-Hotelling marginal-CD + floor."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.intensive
def test_replicate_mu_cov_stage_a():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    from cdsbi.conditioners.bartlett_summary import BartlettSummaryConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi import CDSBIRunner
    from cdsbi.diagnostics.multivariate_marginal_cd import MultivariateMarginalCDRecovery

    torch.manual_seed(0)
    sim = NormalBivariateUnknownCov()
    flow = SingleIndexMonotoneFlow(d=5, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=64, depth=2)
    cond = BartlettSummaryConditioner(n_iid=sim.n_iid)
    runner = CDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss())
    config = {"lr": 3e-3, "batch_size": 256, "n_steps": 12000, "n_train": 20000,
              "optimizer": "adamw", "fresh_batch": False}
    trained = runner.fit(simulator=sim, config=config, seed=0)
    assert trained.procedure.flow is not None   # set by CDSBIRunner.fit

    # (a) recovery
    rng = np.random.default_rng(123)
    theta, x = sim.sample(4000, rng)
    with torch.no_grad():
        per = ((trained.procedure.pivot(theta, x).cpu() - sim.r_star(theta, x).cpu()) ** 2).mean(0).sqrt()
    rmse = float((per ** 2).mean().sqrt())
    print(f"rmse={rmse:.3f} per-coord={per.numpy().round(3)}")
    assert rmse < 0.15, f"overall RMSE {rmse:.3f}"

    # (b) marginal-CD recovery (covariance + μ Hotelling)
    grid = [tuple(map(float, t)) for t in [
        (0.0, 0.0, 0.0, 0.0, 0.0), (0.4, -0.4, 0.8, 1.5, -1.5), (-0.4, 0.4, -0.8, -1.5, 1.5)]]
    mres = MultivariateMarginalCDRecovery(theta_0_grid=grid, n_per_theta=3000)(trained, sim).value
    print(mres[["cov1_ks", "cov2_ks", "cov3_ks", "mu_hotelling_ks"]].to_string())
    assert mres[["cov1_ks", "cov2_ks", "cov3_ks"]].to_numpy().max() < 0.07
    assert mres["mu_hotelling_ks"].max() < 0.08

    # (c) no cheat: loss respects the entropy floor
    H = sim.entropy_lower_bound()
    print(f"final_loss={trained.final_loss:.3f} floor={H:.3f}")
    assert trained.final_loss > H - 0.15

    # (d) joint Mahalanobis ‖r‖²~χ²₅
    from scipy.stats import kstest, chi2
    xv = sim.sample_x_given_theta((0.0, 0.0, 0.0, 0.0, 0.0), 3000, np.random.default_rng(7))
    with torch.no_grad():
        rr = trained.procedure.pivot(torch.zeros(3000, 5), xv).cpu().numpy()
    assert kstest(chi2.cdf((rr ** 2).sum(1), df=5), "uniform").statistic < 0.06
