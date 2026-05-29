"""Stage-A smoke: CDSBI on (μ,σ²) with oracle summary recovers r* in a loose band."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
from cdsbi.conditioners.sufficient_stat import SufficientStatConditioner
from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
from cdsbi.losses.nfmle import NFMLELoss
from cdsbi.methods.cd_sbi import CDSBIRunner


@pytest.mark.intensive
def test_stage_a_recovers_pivot():
    torch.manual_seed(0)
    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=32, depth=2)
    cond = SufficientStatConditioner(n_iid=sim.n_iid)
    runner = CDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss())
    config = {"lr": 3e-3, "batch_size": 256, "n_steps": 5000, "n_train": 10000,
              "optimizer": "adamw", "fresh_batch": False}
    trained = runner.fit(simulator=sim, config=config, seed=0)

    rng = np.random.default_rng(123)
    theta, x = sim.sample(4000, rng)
    with torch.no_grad():
        r_hat = trained.procedure.pivot(theta, x).cpu()
        r_true = sim.r_star(theta, x).cpu()
    rmse = float(((r_hat - r_true) ** 2).mean().sqrt())
    print(f"rmse={rmse:.4f}")
    assert rmse < 0.25, f"pivot RMSE {rmse:.3f} too high — Stage A did not recover r*"
    from scipy.stats import norm as _norm
    u = _norm.cdf(r_hat.numpy())
    assert abs(u.mean() - 0.5) < 0.05
