"""AsinhTransformedSimulator (data-conditioning gate, pilot finding #2)."""
import numpy as np
import torch
from omegaconf import OmegaConf

from cdsbi.simulators.cauchy_loc_scale import CauchyLocScale
from cdsbi.simulators.transformed import AsinhTransformedSimulator


def test_draws_are_asinh_of_base():
    base, wrapped = CauchyLocScale(), AsinhTransformedSimulator(CauchyLocScale())
    xb = base.sample_x_given_theta((0.3, -1.0), 256, np.random.default_rng(7))
    yw = wrapped.sample_x_given_theta((0.3, -1.0), 256, np.random.default_rng(7))
    assert torch.allclose(yw, torch.asinh(xb), atol=1e-6)


def test_tails_are_tamed():
    wrapped = AsinhTransformedSimulator(CauchyLocScale())
    y = wrapped.sample_x_given_theta((1.0, 0.0), 100_000, np.random.default_rng(0))
    assert float(y.abs().max()) < 40.0  # raw Cauchy max would be ~1e4-1e6


def test_oracle_pivot_exact_through_the_transform():
    """r_star(theta, asinh(X)) must equal the base pivot on raw X — the
    measured floor row stays exactly calibrated under conditioning."""
    base = CauchyLocScale()
    wrapped = AsinhTransformedSimulator(CauchyLocScale())
    rng = np.random.default_rng(3)
    x = base.sample_x_given_theta((0.5, 1.0), 4096, rng)
    th = torch.tensor([[0.5, 1.0]]).expand(4096, 2)
    r_base = base.r_star(th, x)
    r_wrap = wrapped.r_star(th, torch.asinh(x))
    assert torch.allclose(r_wrap, r_base, atol=1e-4)


def test_log_prob_change_of_variables():
    """Definitional + stability: log p_Y(y) = log p_X(sinh y) + sum log cosh(y),
    finite even at large |y| (stable log-cosh)."""
    base = CauchyLocScale()
    wrapped = AsinhTransformedSimulator(CauchyLocScale())
    th = torch.tensor([[0.0, 0.0], [0.5, -1.0]])
    y = torch.tensor([[0.5] * 10, [30.0] * 10])  # incl. a deep-tail point
    lp = wrapped.log_prob(y, th)
    # Definitional check on the moderate row (direct cosh would overflow on the
    # deep-tail row — that row is checked for finiteness via the stable form).
    lp0 = base.log_prob(torch.sinh(y[:1]), th[:1]) + torch.log(torch.cosh(y[:1])).sum(-1)
    assert torch.allclose(lp[:1], lp0, atol=1e-4)
    assert torch.isfinite(lp).all()  # row 1 (|y|=30) finite via stable log-cosh


def test_build_simulator_flag():
    from cdsbi.experiments.run import _build_simulator
    cfg = OmegaConf.create({"target": {
        "name": "cauchy_loc_scale_asinh",
        "_target_": "cdsbi.simulators.cauchy_loc_scale.CauchyLocScale",
        "n_iid": 10, "asinh": True}})
    sim = _build_simulator(cfg)
    assert isinstance(sim, AsinhTransformedSimulator)
    assert sim.d_theta == 2 and sim.d_x == 10
