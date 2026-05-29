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


def test_r_star_is_standard_normal_at_truth():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    sim = NormalUnknownMeanVar()
    rng = np.random.default_rng(2)
    theta0 = [np.log(1.5), 0.5]                  # σ = 1.5, μ = 0.5
    x = sim.sample_x_given_theta(theta0, 40000, rng)
    theta = torch.tensor(theta0, dtype=torch.float32).expand(40000, 2)
    r = sim.r_star(theta, x)
    assert r.shape == (40000, 2)
    r_np = r.numpy()
    assert np.allclose(r_np.mean(axis=0), 0.0, atol=0.03)
    assert np.allclose(r_np.std(axis=0), 1.0, atol=0.03)
    assert abs(float(np.corrcoef(r_np.T)[0, 1])) < 0.03


def test_r_star_monotonicity_signs():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    sim = NormalUnknownMeanVar()
    rng = np.random.default_rng(3)
    x = sim.sample_x_given_theta([0.0, 0.0], 1, rng)         # shape (1, 10)
    base = torch.tensor([[0.0, 0.0]])
    r0 = sim.r_star(base, x)
    # r_μ ↑ in μ (index 1): increasing μ raises r_μ
    r_mu_up = sim.r_star(torch.tensor([[0.0, 0.5]]), x)
    assert float(r_mu_up[0, 1]) > float(r0[0, 1])
    # r_σ ↑ in log σ (index 0): increasing log σ raises r_σ
    r_sig_up = sim.r_star(torch.tensor([[0.5, 0.0]]), x)
    assert float(r_sig_up[0, 0]) > float(r0[0, 0])


def test_entropy_lower_bound_runs():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    val = NormalUnknownMeanVar().entropy_lower_bound(n_mc=20000)
    assert np.isfinite(val)
