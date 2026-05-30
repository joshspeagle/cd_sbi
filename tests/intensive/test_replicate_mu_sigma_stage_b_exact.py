"""Intensive: Arm I-A (invertible exact density) — does forcing information-
preservation recover σ² (II-A's casualty) AND calibrate near the oracle, with the
loss bounded by H(X|θ)?"""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.intensive
def test_arm_ia_exact_density_recovers_sigma_and_calibrates():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.invertible_summary import InvertibleSummaryConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.methods.cd_sbi_exact_density import ExactDensityCDSBIRunner
    from cdsbi.diagnostics.sufficiency_recovery import SufficiencyRecovery
    from cdsbi.diagnostics.floor_integrity import FloorIntegrity
    from scipy.stats import kstest, chi2

    torch.manual_seed(0)
    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=32, depth=2)
    cond = InvertibleSummaryConditioner(n_iid=sim.n_iid, d_theta=2, hidden=64, n_layers=6, depth=2)
    runner = ExactDensityCDSBIRunner(flow=flow, conditioner=cond)
    config = {"lr": 2e-3, "batch_size": 256, "n_steps": 8000, "n_train": 10000, "fresh_batch": False}
    trained = runner.fit(simulator=sim, config=config, seed=0)

    sr = SufficiencyRecovery(n_eval=4000)(trained, sim).value
    print(f"I-A sufficiency: log s²={sr['spearman_log_s2'].iloc[0]:.3f} X̄={sr['spearman_xbar'].iloc[0]:.3f}")
    assert sr["spearman_log_s2"].iloc[0] > 0.9, "σ²-information not recovered by exact density"
    assert sr["spearman_xbar"].iloc[0] > 0.9, "μ-information not recovered"

    fi = FloorIntegrity()(trained, sim).value
    print(f"I-A floor: final_loss={fi['final_loss'].iloc[0]:.3f} H(X|θ)={fi['entropy_floor'].iloc[0]:.3f}")
    assert not bool(fi["cheats"].iloc[0]), "exact-density loss fell below H(X|θ) — impossible unless a bug"

    for theta_0 in [(0.0, 0.0), (-0.69, 2.0), (0.69, -2.0)]:
        xv = sim.sample_x_given_theta(theta_0, 3000, np.random.default_rng(7))
        th = torch.tensor([list(theta_0)], dtype=xv.dtype).expand(xv.shape[0], -1)
        with torch.no_grad():
            r = trained.procedure.pivot(th, xv).cpu().numpy()
        ks = kstest(chi2.cdf((r ** 2).sum(1), df=2), "uniform").statistic
        print(f"I-A joint Mahalanobis KS @ {theta_0} = {ks:.3f}")
        assert ks < 0.08, f"I-A joint calibration KS {ks:.3f} too high at {theta_0}"
