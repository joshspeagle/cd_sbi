"""MultivariateMarginalCDRecovery: covariance direct PITs + μ Hotelling-recovery."""
import numpy as np
import torch


class _OracleProc:
    d_theta = 5
    def __init__(self, sim):
        self._sim = sim
        self.encode_fn = lambda x: sim.oracle_summary(x)
    def pivot(self, theta, x): return self._sim.r_star(theta, x)


class _Trained:
    def __init__(self, sim): self.procedure = _OracleProc(sim)


def test_covariance_direct_pits_uniform_with_oracle():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    from cdsbi.diagnostics.multivariate_marginal_cd import MultivariateMarginalCDRecovery
    sim = NormalBivariateUnknownCov()
    grid = [(0.2, -0.1, 0.4, 1.0, -1.5), (-0.3, 0.2, -0.5, -1.0, 1.0)]
    res = MultivariateMarginalCDRecovery(theta_0_grid=grid, n_per_theta=3000)(_Trained(sim), sim)
    df = res.value
    assert {"cov1_ks", "cov2_ks", "cov3_ks", "mu_hotelling_ks"}.issubset(df.columns)
    assert df[["cov1_ks", "cov2_ks", "cov3_ks"]].to_numpy().max() < 0.06


def test_mu_hotelling_recovery_with_oracle():
    # With the closed-form r* pivot, the sampled μ-marginal must recover Hotelling-T²
    # (the joint CD ↦ Hotelling result, validated in the N2 prototype, KS ≈ 0.004).
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    from cdsbi.diagnostics.multivariate_marginal_cd import MultivariateMarginalCDRecovery
    sim = NormalBivariateUnknownCov()
    grid = [(0.2, -0.1, 0.4, 1.0, -1.5)]
    res = MultivariateMarginalCDRecovery(theta_0_grid=grid, n_per_theta=4000)(_Trained(sim), sim)
    assert res.value["mu_hotelling_ks"].iloc[0] < 0.06


def test_noop_without_encode_fn():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    from cdsbi.diagnostics.multivariate_marginal_cd import MultivariateMarginalCDRecovery
    sim = NormalBivariateUnknownCov()
    class _P: d_theta = 5; encode_fn = None
    class _T: procedure = _P()
    res = MultivariateMarginalCDRecovery(theta_0_grid=[(0,)*5], n_per_theta=10)(_T(), sim)
    assert res.passed and "reason" in res.meta
