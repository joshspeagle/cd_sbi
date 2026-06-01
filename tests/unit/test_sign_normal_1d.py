# tests/unit/test_sign_normal_1d.py
import math
import numpy as np
import torch
from cdsbi.simulators.sign_normal_1d import SignNormal1D


def test_shapes_and_attrs():
    sim = SignNormal1D(n_iid=10)
    assert sim.d_theta == 1 and sim.d_x == 10
    rng = np.random.default_rng(0)
    theta, x = sim.sample(16, rng)
    assert theta.shape == (16, 1) and x.shape == (16, 10)


def test_sign_symmetry_of_data():
    """X depends on θ only through θ² — so +θ and −θ give the same data law."""
    sim = SignNormal1D(n_iid=10)
    rng = np.random.default_rng(0)
    xp = sim.sample_x_given_theta(np.array([1.5]), 50000, rng).mean().item()
    xn = sim.sample_x_given_theta(np.array([-1.5]), 50000, rng).mean().item()
    assert abs(xp - xn) < 0.05 and abs(xp - 1.5 ** 2) < 0.05


def test_log_prob_matches_gaussian():
    sim = SignNormal1D(n_iid=3)
    x = torch.tensor([[1.0, 2.0, 0.5]])
    theta = torch.tensor([[1.0]])             # θ²=1 → mean 1
    expected = sum(-0.5 * (xi - 1.0) ** 2 - 0.5 * math.log(2 * math.pi)
                   for xi in [1.0, 2.0, 0.5])
    assert abs(sim.log_prob(x, theta).item() - expected) < 1e-5


def test_r_star():
    """X̄=0, θ²=1 ⟹ r* = √4·(0 − 1) = −2."""
    sim = SignNormal1D(n_iid=4)
    result = sim.r_star(torch.tensor([[1.0]]), torch.zeros(1, 4))
    assert torch.allclose(result, torch.tensor([[-2.0]]))
