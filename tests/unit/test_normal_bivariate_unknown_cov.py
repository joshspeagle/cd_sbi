"""NormalBivariateUnknownCov: sampling, shapes, prior, signs."""
import numpy as np
import torch


def test_sample_shapes_and_dims():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    sim = NormalBivariateUnknownCov()
    assert sim.d_theta == 5 and sim.d_x == 20
    theta, x = sim.sample(64, np.random.default_rng(0))
    assert theta.shape == (64, 5) and x.shape == (64, 20)


def test_sample_x_given_theta_is_bivariate_normal():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    sim = NormalBivariateUnknownCov()
    theta0 = (0.2, -0.1, 0.5, 1.0, -2.0)
    x = sim.sample_x_given_theta(theta0, 20000, np.random.default_rng(1))
    assert x.shape == (20000, 20)
    obs = x.reshape(20000, sim.n_iid, 2).reshape(-1, 2).numpy()
    L = np.array([[np.exp(0.2), 0.0], [0.5, np.exp(-0.1)]])
    Sig = L @ L.T
    assert np.allclose(obs.mean(0), [1.0, -2.0], atol=0.05)
    assert np.allclose(np.cov(obs.T), Sig, atol=0.08)


def test_signs():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    sim = NormalBivariateUnknownCov()
    assert tuple(sim.theta_signs) == (1.0, 1.0, -1.0, 1.0, 1.0)
    assert tuple(sim.feat_signs) == (-1.0, -1.0, 1.0, -1.0, -1.0)
