"""ExactDensityCDSBIRunner: exact −log p(X|θ) over bijection+pivot; floor-bounded."""
import torch


def test_exact_density_runner_fit_trains_and_floor_bounded():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.invertible_summary import InvertibleSummaryConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.methods.cd_sbi_exact_density import ExactDensityCDSBIRunner

    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=16)
    cond = InvertibleSummaryConditioner(n_iid=sim.n_iid, d_theta=2, hidden=16, n_layers=4)
    runner = ExactDensityCDSBIRunner(flow=flow, conditioner=cond, loss=None, device="cpu")
    trained = runner.fit(simulator=sim, config={"lr": 3e-3, "batch_size": 128, "n_steps": 50,
                                                "n_train": 2000, "fresh_batch": False}, seed=0)
    import numpy as np
    assert np.isfinite(trained.final_loss)
    assert trained.arch_metadata["loss_class"] == "ExactDensityLoss"
    assert trained.procedure.encode_fn is not None
    _, x = sim.sample(16, np.random.default_rng(0))
    assert trained.procedure.pivot(torch.zeros(16, 2), x).shape == (16, 2)
    assert trained.final_loss > sim.data_entropy_lower_bound() - 2.0
