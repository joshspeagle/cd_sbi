"""Intensive: Arm II-A (energy calibration) — co-adapted learned summary recovers
sufficiency (incl. σ²) AND calibrates near the Stage-A oracle, with no cheat channel."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.intensive
def test_arm_iia_energy_calibrates_and_recovers_sufficiency():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.energy_calibration import EnergyCalibrationLoss
    from cdsbi.methods.cd_sbi_energy import EnergyCDSBIRunner
    from cdsbi.diagnostics.sufficiency_recovery import SufficiencyRecovery
    from scipy.stats import kstest, chi2

    torch.manual_seed(0)
    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=32, depth=2)
    cond = DeepSetsConditioner(n_iid=sim.n_iid, d_out=2, hidden=32, depth=2, standardize=False)
    runner = EnergyCDSBIRunner(flow=flow, conditioner=cond, loss=EnergyCalibrationLoss(n_ref=256))
    config = {"lr": 2e-3, "n_steps": 8000,
              "group": {"n_theta_per_batch": 32, "group_size": 64, "n_ref": 256}}
    trained = runner.fit(simulator=sim, config=config, seed=0)

    print(f"\nII-A final_loss = {trained.final_loss:.4f}")

    sr = SufficiencyRecovery(n_eval=4000)(trained, sim).value
    print(f"II-A sufficiency: log s²={sr['spearman_log_s2'].iloc[0]:.3f} "
          f"X̄={sr['spearman_xbar'].iloc[0]:.3f}")
    assert sr["spearman_log_s2"].iloc[0] > 0.9, "σ²-information not recovered"
    assert sr["spearman_xbar"].iloc[0] > 0.9, "μ-information not recovered"

    for theta_0 in [(0.0, 0.0), (-0.69, 2.0), (0.69, -2.0)]:
        xv = sim.sample_x_given_theta(theta_0, 3000, np.random.default_rng(7))
        th = torch.tensor([list(theta_0)], dtype=xv.dtype).expand(xv.shape[0], -1)
        with torch.no_grad():
            r = trained.procedure.pivot(th, xv).cpu().numpy()
        ks = kstest(chi2.cdf((r ** 2).sum(1), df=2), "uniform").statistic
        print(f"II-A joint Mahalanobis KS @ {theta_0} = {ks:.3f}")
        assert ks < 0.08, f"II-A joint calibration KS {ks:.3f} too high at {theta_0}"
