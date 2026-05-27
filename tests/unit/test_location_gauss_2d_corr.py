"""LocationGaussian2D_corr — correlated-Σ 2D Gaussian simulator."""
from __future__ import annotations

import numpy as np
import torch


def test_simulator_sample_shape_and_correlation(seed):
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    rng = np.random.default_rng(seed)
    sim = LocationGaussian2D_corr()
    assert sim.d_theta == 2 and sim.d_x == 2
    theta, x = sim.sample(20000, rng)
    assert theta.shape == (20000, 2)
    assert x.shape == (20000, 2)
    # X - θ ~ N(0, Σ); empirical Σ ≈ [[1, 0.5], [0.5, 1]].
    eps = (x - theta).numpy()
    cov_emp = np.cov(eps.T)
    np.testing.assert_allclose(cov_emp, [[1.0, 0.5], [0.5, 1.0]], atol=0.06)


def test_sample_x_given_theta_shape_and_correlation(seed):
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    rng = np.random.default_rng(seed)
    sim = LocationGaussian2D_corr()
    x = sim.sample_x_given_theta(theta_0=[1.5, -0.5], n=10000, rng=rng)
    assert x.shape == (10000, 2)
    # X | θ=[1.5, -0.5] ~ N([1.5, -0.5], Σ)
    np.testing.assert_allclose(x.mean(0).numpy(), [1.5, -0.5], atol=0.05)
    cov_emp = np.cov((x - torch.tensor([1.5, -0.5])).T.numpy())
    np.testing.assert_allclose(cov_emp, [[1.0, 0.5], [0.5, 1.0]], atol=0.07)


def test_r_star_returns_decorrelated_normal(seed):
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    rng = np.random.default_rng(seed)
    sim = LocationGaussian2D_corr()
    theta, x = sim.sample(20000, rng)
    r = sim.r_star(theta, x).numpy()
    # r* = L⁻¹(θ - X) ~ N(0, I_2). Mean ≈ 0, cov ≈ I.
    np.testing.assert_allclose(r.mean(0), [0.0, 0.0], atol=0.05)
    np.testing.assert_allclose(np.cov(r.T), [[1.0, 0.0], [0.0, 1.0]], atol=0.05)


def test_r_star_jacobian_is_L_inverse():
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    sim = LocationGaussian2D_corr()
    J = sim.r_star_jacobian()
    # L⁻¹ for Σ = [[1, 0.5], [0.5, 1]]; L is lower-triangular with
    # L = [[1, 0], [0.5, sqrt(0.75)]]; L⁻¹ = [[1, 0], [-0.5/sqrt(0.75), 1/sqrt(0.75)]]
    # = [[1, 0], [-0.5774, 1.1547]] to 4 decimals.
    expected = np.array([[1.0, 0.0], [-0.5773502691896258, 1.1547005383792515]])
    np.testing.assert_allclose(J.numpy(), expected, atol=1e-6)


def test_r_star_jacobian_honors_dtype():
    import torch
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    sim = LocationGaussian2D_corr()
    J_default = sim.r_star_jacobian()
    assert J_default.dtype == torch.get_default_dtype()
    J_double = sim.r_star_jacobian(dtype=torch.float64)
    assert J_double.dtype == torch.float64
    # Numeric values must match across dtypes.
    import numpy as np
    np.testing.assert_allclose(J_default.numpy(), J_double.numpy(), atol=1e-6)


def test_log_prob_matches_independent_gaussian_formula(seed):
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    rng = np.random.default_rng(seed)
    sim = LocationGaussian2D_corr()
    theta, x = sim.sample(100, rng)
    # Reference: scipy.stats.multivariate_normal.logpdf
    from scipy.stats import multivariate_normal
    Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    expected = np.array([
        multivariate_normal(mean=t.numpy(), cov=Sigma).logpdf(xi.numpy())
        for t, xi in zip(theta, x)
    ])
    got = sim.log_prob(x, theta).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5)
