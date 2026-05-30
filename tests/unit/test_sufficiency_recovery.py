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


def test_sufficiency_recovery_perfect_when_encode_is_oracle():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.diagnostics.sufficiency_recovery import SufficiencyRecovery
    sim = NormalUnknownMeanVar()

    class _Proc:
        d_theta = 2
        def __init__(self, sim): self.encode_fn = lambda x: sim.oracle_summary(x)
        def pivot(self, theta, x): return sim.r_star(theta, x)
    class _Trained:
        def __init__(self, sim): self.procedure = _Proc(sim)

    res = SufficiencyRecovery(n_eval=3000)(_Trained(sim), sim)
    df = res.value
    assert {"sufficiency_min_spearman", "spearman_log_s2", "spearman_xbar"}.issubset(df.columns)
    assert df["sufficiency_min_spearman"].iloc[0] > 0.98
    assert res.passed


def test_sufficiency_recovery_detects_collapse():
    import torch
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.diagnostics.sufficiency_recovery import SufficiencyRecovery
    sim = NormalUnknownMeanVar()

    class _Proc:
        d_theta = 2
        def __init__(self, sim):
            self.encode_fn = lambda x: x.mean(dim=-1, keepdim=True).repeat(1, 2)  # only X̄
        def pivot(self, theta, x): return sim.r_star(theta, x)
    class _Trained:
        def __init__(self, sim): self.procedure = _Proc(sim)

    res = SufficiencyRecovery(n_eval=3000)(_Trained(sim), sim)
    assert res.value["spearman_log_s2"].iloc[0] < 0.5
    assert not res.passed


def test_sufficiency_recovery_noop_without_encode_fn():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.diagnostics.sufficiency_recovery import SufficiencyRecovery
    sim = NormalUnknownMeanVar()
    class _Proc:
        d_theta = 2; encode_fn = None
        def pivot(self, theta, x): return x
    class _Trained:
        procedure = _Proc()
    res = SufficiencyRecovery(n_eval=10)(_Trained(), sim)
    assert res.passed and "reason" in res.meta
