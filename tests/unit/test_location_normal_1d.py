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
