"""MarginalCDRecovery: σ² (χ²) and μ (Student-t) marginal-CD recovery."""
import numpy as np
import torch
from scipy.stats import norm


class _OraclePivotProcedure:
    """Wraps simulator.r_star as a PivotBasedProcedure-like object."""
    d_theta = 2
    def __init__(self, sim): self._sim = sim
    def pivot(self, theta, x): return self._sim.r_star(theta, x)


class _Trained:
    def __init__(self, sim): self.procedure = _OraclePivotProcedure(sim)


def _x_per_theta(sim, grid, n, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for th in grid:
        key = repr([float(v) for v in th])
        out[key] = sim.sample_x_given_theta(th, n, rng)
    return out


def test_sigma_branch_recovers_chi2_cd_with_oracle():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.diagnostics.marginal_cd_recovery import MarginalCDRecovery
    sim = NormalUnknownMeanVar()
    grid = [(np.log(0.5), -1.0), (np.log(2.0), 1.0)]
    diag = MarginalCDRecovery(theta_0_grid=grid, n_per_theta=3000)
    trained = _Trained(sim)
    res = diag(trained, sim, x_per_theta=_x_per_theta(sim, grid, 3000))
    df = res.value
    assert {"sigma_ks", "sigma_chi2_resid"}.issubset(df.columns)
    assert df["sigma_ks"].max() < 0.06
    assert df["sigma_chi2_resid"].max() < 1e-3


def test_noop_when_not_pivot_based():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.diagnostics.marginal_cd_recovery import MarginalCDRecovery
    sim = NormalUnknownMeanVar()
    class _NoPivot: procedure = object()
    res = MarginalCDRecovery(theta_0_grid=[(0.0, 0.0)], n_per_theta=10)(_NoPivot(), sim)
    assert res.passed and "reason" in res.meta


def test_mu_branch_recovers_student_t_with_oracle():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.diagnostics.marginal_cd_recovery import MarginalCDRecovery
    sim = NormalUnknownMeanVar()
    grid = [(np.log(0.5), -1.0), (np.log(2.0), 1.0)]
    diag = MarginalCDRecovery(theta_0_grid=grid, n_per_theta=2000)
    trained = _Trained(sim)
    res = diag(trained, sim, x_per_theta=_x_per_theta(sim, grid, 2000))
    df = res.value
    assert {"mu_ks", "mu_t_resid"}.issubset(df.columns)
    assert df["mu_ks"].max() < 0.06
    assert df["mu_t_resid"].max() < 0.02
