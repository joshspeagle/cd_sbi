"""ReducedSimulator — wraps ExponentialRate with a frozen-sum X → T reduction."""
from __future__ import annotations

import numpy as np
import torch


def test_reduced_simulator_d_x_collapses_to_one(seed):
    from cdsbi.conditioners.mlp import MLPConditioner
    from cdsbi.simulators._reduced import ReducedSimulator
    from cdsbi.simulators.exp_rate import ExponentialRate
    base = ExponentialRate()
    cond = MLPConditioner(input_dim=base.d_x, output_dim=1, mode="frozen_sum")
    reduced = ReducedSimulator(base, cond)
    assert reduced.d_theta == 1
    assert reduced.d_x == 1
    assert reduced.theta_range == base.theta_range


def test_reduced_simulator_sample_returns_T(seed):
    from cdsbi.conditioners.mlp import MLPConditioner
    from cdsbi.simulators._reduced import ReducedSimulator
    from cdsbi.simulators.exp_rate import ExponentialRate
    rng = np.random.default_rng(seed)
    base = ExponentialRate()
    cond = MLPConditioner(input_dim=base.d_x, output_dim=1, mode="frozen_sum")
    reduced = ReducedSimulator(base, cond)
    theta, T = reduced.sample(2000, rng)
    assert theta.shape == (2000, 1)
    assert T.shape == (2000, 1)
    # T = Σ X_i ~ Gamma(n=5, scale=1/θ). E[T | θ] = 5/θ.
    # Spot-check: T values are all positive (sum of exponentials).
    assert (T > 0).all()


def test_reduced_simulator_sample_x_given_theta_returns_T(seed):
    from cdsbi.conditioners.mlp import MLPConditioner
    from cdsbi.simulators._reduced import ReducedSimulator
    from cdsbi.simulators.exp_rate import ExponentialRate
    rng = np.random.default_rng(seed)
    base = ExponentialRate()
    cond = MLPConditioner(input_dim=base.d_x, output_dim=1, mode="frozen_sum")
    reduced = ReducedSimulator(base, cond)
    T = reduced.sample_x_given_theta(theta_0=1.5, n=10000, rng=rng)
    assert T.shape == (10000, 1)
    # T | θ=1.5 ~ Gamma(5, 1/1.5); mean = 5/1.5 ≈ 3.33
    np.testing.assert_allclose(float(T.mean()), 5.0 / 1.5, atol=0.05)


def test_reduced_simulator_r_star_delegates_to_base(seed):
    """r_star on the reduced simulator should give the same answer as
    calling r_star on the base simulator with the reduced T directly
    (ExponentialRate.r_star accepts either shape)."""
    from cdsbi.conditioners.mlp import MLPConditioner
    from cdsbi.simulators._reduced import ReducedSimulator
    from cdsbi.simulators.exp_rate import ExponentialRate
    rng = np.random.default_rng(seed)
    base = ExponentialRate()
    cond = MLPConditioner(input_dim=base.d_x, output_dim=1, mode="frozen_sum")
    reduced = ReducedSimulator(base, cond)
    theta, T = reduced.sample(500, rng)
    r_reduced = reduced.r_star(theta, T)
    r_base = base.r_star(theta, T)
    torch.testing.assert_close(r_reduced, r_base)
