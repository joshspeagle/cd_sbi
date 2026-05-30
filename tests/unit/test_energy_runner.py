"""EnergyCDSBIRunner: grouped fit trains summary+flow under the energy loss."""
import torch


def test_energy_runner_fit_trains_and_exposes_procedure():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.energy_calibration import EnergyCalibrationLoss
    from cdsbi.methods.cd_sbi_energy import EnergyCDSBIRunner

    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=16)
    cond = DeepSetsConditioner(n_iid=sim.n_iid, d_out=2, hidden=16, standardize=False)
    before = {k: v.clone() for k, v in cond.state_dict().items() if v.dtype.is_floating_point}
    runner = EnergyCDSBIRunner(flow=flow, conditioner=cond, loss=EnergyCalibrationLoss(n_ref=128),
                               device="cpu")
    trained = runner.fit(simulator=sim, config={
        "lr": 3e-3, "n_steps": 40,
        "group": {"n_theta_per_batch": 16, "group_size": 32, "n_ref": 128},
    }, seed=0)
    after = cond.state_dict()
    assert any(not torch.allclose(before[k], after[k]) for k in before if k.startswith(("phi", "rho")))
    assert trained.procedure.encode_fn is not None
    import numpy as np
    _, x = sim.sample(20, np.random.default_rng(0))
    assert trained.procedure.pivot(torch.zeros(20, 2), x).shape == (20, 2)
    assert trained.arch_metadata["loss_class"] == "EnergyCalibrationLoss"
    assert np.isfinite(trained.final_loss)
