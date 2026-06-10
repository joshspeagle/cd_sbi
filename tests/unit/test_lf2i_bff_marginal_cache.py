"""LF2I-BFF per-x marginal cache (perf hardening #2, 2026-06-10).

The BFF marginal depends only on x; set construction probes hundreds of θ per
fixed x and was recomputing the N_grid-row marginal on every probe (~99.6% of
the statistic's cost). The cache must change NOTHING numerically.
"""
import numpy as np
import torch

from cdsbi.flows.maf_adapter import MAFAdapter  # noqa: F401 (env warm)
from cdsbi.methods.lf2i_bff import LF2IBFFRunner
from cdsbi.simulators.location_gauss_2d_iid import LocationGaussian2D_iid


def _fit_tiny():
    torch.manual_seed(0); np.random.seed(0)
    sim = LocationGaussian2D_iid()
    runner = LF2IBFFRunner(classifier_hidden=16, classifier_depth=2,
                           quantile_hidden=8, quantile_depth=2,
                           marginal_n=64, device="cpu")
    cfg = dict(optimizer="adam", lr=3e-3, batch_size=128, n_steps=30,
               n_train=512, n_train_stat=512, n_train_quantile=128,
               fresh_batch=False, grad_clip_norm=5.0,
               alpha_grid=[0.5, 0.9])
    return sim, runner.fit(sim, cfg, seed=0)


def test_statistic_value_unchanged_by_cache_and_branch():
    """{many θ, one X} early-return branch must equal the row-wise computation
    (same θ paired with copies of the same x through the equal-B branch)."""
    sim, tm = _fit_tiny()
    proc = tm.procedure
    x = sim.sample_x_given_theta((0.0, 0.0), 1, np.random.default_rng(1))
    thetas = torch.randn(32, 2)
    t_fast = proc.test_statistic(thetas, x)                  # one-X-many-θ branch
    x_rep = x.expand(32, -1).contiguous()
    t_slow = proc.test_statistic(thetas, x_rep)              # equal-B branch
    assert torch.allclose(t_fast, t_slow, atol=1e-5)


def test_marginal_computed_once_per_x_across_theta_probes():
    sim, tm = _fit_tiny()
    proc = tm.procedure
    n0 = proc.bff_marginal_evals["n"]
    x = sim.sample_x_given_theta((1.0, -1.0), 1, np.random.default_rng(2))
    for k in range(50):  # 50 θ probes at the SAME x (the ray-bisection pattern)
        proc.test_statistic(torch.randn(8, 2), x)
    assert proc.bff_marginal_evals["n"] == n0 + 1            # ONE compute, 49 hits
    x2 = sim.sample_x_given_theta((0.0, 0.0), 1, np.random.default_rng(3))
    proc.test_statistic(torch.randn(8, 2), x2)
    assert proc.bff_marginal_evals["n"] == n0 + 2            # new x -> one more
