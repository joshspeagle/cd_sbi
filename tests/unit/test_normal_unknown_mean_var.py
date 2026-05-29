"""NormalUnknownMeanVar: X = (X_1..X_n_iid) iid N(μ, σ²); θ = (log σ, μ)."""
from __future__ import annotations

import numpy as np
import torch


def test_shapes_and_priors():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    sim = NormalUnknownMeanVar()
    assert sim.d_theta == 2 and sim.d_x == 10 and sim.n_iid == 10
    rng = np.random.default_rng(0)
    theta, x = sim.sample(20000, rng)
    assert theta.shape == (20000, 2) and x.shape == (20000, 10)
    log_sigma, mu = theta[:, 0], theta[:, 1]
    assert float(log_sigma.min()) >= np.log(0.3) - 1e-4
    assert float(log_sigma.max()) <= np.log(3.0) + 1e-4
    assert float(mu.min()) >= -5.0 - 1e-3 and float(mu.max()) <= 5.0 + 1e-3


def test_conditional_moments():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    sim = NormalUnknownMeanVar()
    rng = np.random.default_rng(1)
    x = sim.sample_x_given_theta([np.log(2.0), 1.0], 50000, rng)
    assert x.shape == (50000, 10)
    assert abs(float(x.mean()) - 1.0) < 0.02
    assert abs(float(x.std()) - 2.0) < 0.03
