import math
import numpy as np
import torch
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def test_sample_shapes():
    sim = LocationNormal1D()
    rng = np.random.default_rng(0)
    theta, x = sim.sample(100, rng)
    assert theta.shape == (100, 1)
    assert x.shape == (100, 1)


def test_sample_theta_in_range():
    sim = LocationNormal1D(theta_range=(-7.0, 7.0))
    rng = np.random.default_rng(0)
    theta, _ = sim.sample(1000, rng)
    assert (theta >= -7.0).all() and (theta <= 7.0).all()


def test_sample_x_conditional_mean():
    sim = LocationNormal1D()
    rng = np.random.default_rng(0)
    theta, x = sim.sample(50_000, rng)
    # E[X | θ] = θ ⇒ (X − θ) has mean 0
    diff = (x - theta).numpy().flatten()
    assert abs(diff.mean()) < 0.02
    assert abs(diff.std() - 1.0) < 0.02


def test_r_star_is_theta_minus_x():
    sim = LocationNormal1D()
    theta = torch.tensor([[1.0], [2.0]])
    x = torch.tensor([[0.5], [3.0]])
    r = sim.r_star(theta, x)
    assert torch.allclose(r, torch.tensor([[0.5], [-1.0]]))


def test_entropy_lower_bound():
    sim = LocationNormal1D()
    assert abs(sim.entropy_lower_bound() - 0.5 * math.log(2 * math.pi * math.e)) < 1e-12


def test_log_prob_matches_normal():
    sim = LocationNormal1D()
    theta = torch.tensor([[0.0]])
    x = torch.tensor([[1.0]])
    expected = -0.5 * math.log(2 * math.pi) - 0.5
    assert abs(sim.log_prob(x, theta).item() - expected) < 1e-6


def test_sample_x_given_theta_shape_and_distribution(seed):
    import numpy as np
    from cdsbi.simulators.location_normal_1d import LocationNormal1D
    rng = np.random.default_rng(seed)
    sim = LocationNormal1D()
    x = sim.sample_x_given_theta(theta_0=2.5, n=5000, rng=rng)
    assert x.shape == (5000, 1)
    # X | θ=2.5 ~ N(2.5, 1)
    assert abs(float(x.mean()) - 2.5) < 0.05
    assert abs(float(x.std()) - 1.0) < 0.05


def test_sample_x_given_theta_accepts_vector_theta(seed):
    import numpy as np
    from cdsbi.simulators.location_normal_1d import LocationNormal1D
    rng = np.random.default_rng(seed)
    sim = LocationNormal1D()
    # In 1D, a "vector" θ_0 is a length-1 sequence — implementation must accept both.
    x = sim.sample_x_given_theta(theta_0=[1.0], n=500, rng=rng)
    assert x.shape == (500, 1)
    assert abs(float(x.mean()) - 1.0) < 0.1
