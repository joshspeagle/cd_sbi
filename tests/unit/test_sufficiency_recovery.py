"""encode_fn plumbing + simulator.oracle_summary."""
import torch


def test_oracle_summary_is_log_s2_and_xbar():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    sim = NormalUnknownMeanVar()
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0]])
    o = sim.oracle_summary(x)
    assert o.shape == (1, 2)
    s2 = float(x.var(dim=-1, unbiased=True))
    assert abs(float(o[0, 0]) - torch.log(torch.tensor(s2)).item()) < 1e-5   # log s²
    assert abs(float(o[0, 1]) - 2.0) < 1e-6                                   # X̄


def test_procedure_exposes_encode_fn_after_fit():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi import CDSBIRunner
    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=16)
    cond = DeepSetsConditioner(n_iid=sim.n_iid, d_out=2, hidden=16)
    trained = CDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss()).fit(
        simulator=sim, config={"lr": 1e-3, "batch_size": 64, "n_steps": 5,
                               "n_train": 500, "optimizer": "adam", "fresh_batch": False}, seed=0)
    assert trained.procedure.encode_fn is not None
    import numpy as np
    _, x = sim.sample(32, np.random.default_rng(0))
    feats = trained.procedure.encode_fn(x)
    assert feats.shape == (32, 2)
    assert not feats.requires_grad
