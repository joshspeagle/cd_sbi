"""CDSBIRunner.fit trains conditioner parameters when the conditioner is an nn.Module."""
import copy
import torch


def test_fit_updates_deep_sets_conditioner_params():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi import CDSBIRunner

    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=16)
    cond = DeepSetsConditioner(n_iid=sim.n_iid, d_out=2, hidden=16)
    before = {k: v.clone() for k, v in cond.state_dict().items() if v.dtype.is_floating_point}
    runner = CDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss())
    runner.fit(simulator=sim, config={"lr": 1e-3, "batch_size": 128, "n_steps": 30,
                                      "n_train": 2000, "optimizer": "adam",
                                      "fresh_batch": False}, seed=0)
    after = {k: v.cpu() for k, v in cond.state_dict().items()}
    changed = any(not torch.allclose(before[k], after[k])
                  for k in before if k.startswith(("phi", "rho")))
    assert changed, "conditioner parameters did not update during fit()"


def test_fit_still_works_with_frozen_conditioner():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.sufficient_stat import SufficientStatConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi import CDSBIRunner
    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=16)
    runner = CDSBIRunner(flow=flow, conditioner=SufficientStatConditioner(n_iid=sim.n_iid),
                         loss=NFMLELoss())
    trained = runner.fit(simulator=sim, config={"lr": 1e-3, "batch_size": 128,
                         "n_steps": 10, "n_train": 1000, "optimizer": "adam",
                         "fresh_batch": False}, seed=0)
    assert trained.final_loss == trained.final_loss   # ran without NaN
