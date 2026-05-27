"""LocationGaussian2D_iid: X | θ ~ N(θ, I_2), θ ~ U[-7, 7]^2."""
import math

import numpy as np
import torch

from cdsbi.simulators.location_gauss_2d_iid import LocationGaussian2D_iid


def test_dims_are_2(seed):
    sim = LocationGaussian2D_iid()
    assert sim.d_theta == 2
    assert sim.d_x == 2


def test_sample_shape_and_distribution(seed):
    sim = LocationGaussian2D_iid()
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(n=5000, rng=rng)
    assert theta.shape == (5000, 2)
    assert x.shape == (5000, 2)
    # θ uniform on [-7, 7]^2
    assert abs(float(theta.mean()) - 0.0) < 0.2
    # X - θ ~ N(0, I_2)
    diff = (x - theta).numpy()
    assert abs(diff.mean()) < 0.05
    assert abs(diff.std() - 1.0) < 0.05


def test_r_star_componentwise(seed):
    sim = LocationGaussian2D_iid()
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(n=100, rng=rng)
    r_star = sim.r_star(theta, x)
    assert r_star.shape == (100, 2)
    assert torch.allclose(r_star, theta - x)


def test_log_prob_matches_isotropic_gaussian(seed):
    sim = LocationGaussian2D_iid()
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(n=10, rng=rng)
    ll = sim.log_prob(x, theta)
    expected = -math.log(2 * math.pi) - 0.5 * (x - theta).pow(2).sum(dim=-1)
    assert torch.allclose(ll, expected, atol=1e-6)


def test_sample_x_given_theta_shape_and_distribution(seed):
    sim = LocationGaussian2D_iid()
    rng = np.random.default_rng(seed)
    x = sim.sample_x_given_theta(theta_0=[1.0, -2.0], n=5000, rng=rng)
    assert x.shape == (5000, 2)
    assert abs(float(x[:, 0].mean()) - 1.0) < 0.1
    assert abs(float(x[:, 1].mean()) - (-2.0)) < 0.1


def test_entropy_lower_bound_is_d_times_1d_floor():
    sim = LocationGaussian2D_iid()
    # Joint entropy of N(θ, I_2) is 2 * ½ log(2πe) = log(2πe)
    assert abs(sim.entropy_lower_bound() - math.log(2 * math.pi * math.e)) < 1e-9
