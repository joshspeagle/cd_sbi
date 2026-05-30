"""Stage-A smoke: CDSBI on (μ,Σ) with the Bartlett oracle recovers r* in a band."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.intensive
def test_stage_a_mu_cov_recovers_pivot():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    from cdsbi.conditioners.bartlett_summary import BartlettSummaryConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi import CDSBIRunner

    torch.manual_seed(0)
    sim = NormalBivariateUnknownCov()
    flow = SingleIndexMonotoneFlow(d=5, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=48, depth=2)
    cond = BartlettSummaryConditioner(n_iid=sim.n_iid)
    runner = CDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss())
    config = {"lr": 3e-3, "batch_size": 256, "n_steps": 6000, "n_train": 10000,
              "optimizer": "adamw", "fresh_batch": False}
    trained = runner.fit(simulator=sim, config=config, seed=0)

    rng = np.random.default_rng(123)
    theta, x = sim.sample(4000, rng)
    with torch.no_grad():
        r_hat = trained.procedure.pivot(theta, x).cpu()
        r_true = sim.r_star(theta, x).cpu()
    per_coord = ((r_hat - r_true) ** 2).mean(0).sqrt()
    rmse = float(((r_hat - r_true) ** 2).mean().sqrt())
    print(f"mu_cov rmse={rmse:.4f}  per-coord={per_coord.numpy().round(3)}")
    assert rmse < 0.20, f"pivot RMSE {rmse:.3f} too high — Stage A did not recover r*"
    from scipy.stats import kstest, chi2
    xv = sim.sample_x_given_theta((0.0, 0.0, 0.0, 0.0, 0.0), 3000, np.random.default_rng(7))
    th = torch.zeros(3000, 5)
    with torch.no_grad():
        r = trained.procedure.pivot(th, xv).cpu().numpy()
    ks = kstest(chi2.cdf((r ** 2).sum(1), df=5), "uniform").statistic
    print(f"mu_cov joint Mahalanobis KS={ks:.3f}")
    assert ks < 0.08
