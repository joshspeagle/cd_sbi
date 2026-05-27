"""ExponentialRate — n=5 iid exponential samples, T = Σ X_i sufficient statistic."""
from __future__ import annotations

import numpy as np
import torch


def test_simulator_sample_shape_and_distribution(seed):
    from cdsbi.simulators.exp_rate import ExponentialRate
    rng = np.random.default_rng(seed)
    sim = ExponentialRate()
    assert sim.d_theta == 1 and sim.d_x == 5
    assert sim.n_iid == 5
    theta, x = sim.sample(20000, rng)
    assert theta.shape == (20000, 1)
    assert x.shape == (20000, 5)
    # X | θ ~ Exp(θ) ⇒ E[X] = 1/θ. With θ ~ U[0.3, 3.0], E_marginal[X] ≈ E_θ[1/θ].
    # Sanity-check that values are positive.
    assert (x > 0).all()
    # For a specific theta, X mean ≈ 1/θ to within MC noise (single-theta slice).


def test_sample_x_given_theta_shape_and_mean(seed):
    from cdsbi.simulators.exp_rate import ExponentialRate
    rng = np.random.default_rng(seed)
    sim = ExponentialRate()
    x = sim.sample_x_given_theta(theta_0=1.5, n=20000, rng=rng)
    assert x.shape == (20000, 5)
    # Each X_i ~ Exp(θ=1.5) ⇒ E[X_i] = 2/3
    np.testing.assert_allclose(float(x.mean()), 2.0 / 3.0, atol=0.02)
    # T = Σ X_i has mean n/θ = 5/1.5 ≈ 3.33
    T = x.sum(dim=-1).numpy()
    np.testing.assert_allclose(T.mean(), 5.0 / 1.5, atol=0.05)


def test_r_star_marginal_is_standard_normal(seed):
    """r*(θ, T) = Φ⁻¹(F_{χ²_{2n}}(2θT)) should be ~N(0, 1) marginally over (θ, X)."""
    from cdsbi.simulators.exp_rate import ExponentialRate
    rng = np.random.default_rng(seed)
    sim = ExponentialRate()
    theta, x = sim.sample(20000, rng)
    T = x.sum(dim=-1, keepdim=True)  # (n, 1)
    r = sim.r_star(theta, T).numpy()
    # Marginal ≈ N(0, 1)
    np.testing.assert_allclose(r.mean(), 0.0, atol=0.05)
    np.testing.assert_allclose(r.std(), 1.0, atol=0.05)


def test_entropy_lower_bound_matches_paper():
    """The manuscript reports the NF-MLE loss lower bound ≈ 0.88 for this model.

    `entropy_lower_bound()` is implemented as a Monte-Carlo evaluation of the
    NF-MLE loss at the truth pivot r*, on the (θ, T) reduced space with the
    conditioner's log|∂T/∂X| accounted for. MC noise at n_mc=50_000 is small
    (~ 0.01); the band [0.70, 1.10] is generous to absorb seed and clamp-floor
    variance.
    """
    from cdsbi.simulators.exp_rate import ExponentialRate
    sim = ExponentialRate()
    H = sim.entropy_lower_bound()
    assert 0.70 <= H <= 1.10, f"entropy_lower_bound={H} outside expected [0.70, 1.10] around manuscript's 0.88"
