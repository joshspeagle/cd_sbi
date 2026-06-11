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

def test_content_cache_hits_across_fresh_slice_objects():
    """The d>1 ray-bisection path re-slices a fixed expanded x into FRESH
    tensor objects every bisection iteration (id-keying misses on all of
    them) with bit-identical content — the content-keyed layer must absorb
    them: ONE unique-row compute for 40 'iterations'."""
    sim, tm = _fit_tiny()
    proc = tm.procedure
    x_base = sim.sample_x_given_theta((0.5, 0.5), 6, np.random.default_rng(4))
    x_exp = x_base.unsqueeze(1).expand(-1, 20, -1).reshape(-1, x_base.shape[-1])  # 120 rows
    n0 = proc.bff_marginal_evals["n"]
    vals = []
    for _ in range(40):  # fresh slice object each time, same content
        chunk = x_exp[0:120].clone()
        vals.append(proc.test_statistic(torch.zeros(1, 2).expand(120, -1), chunk))
    assert proc.bff_marginal_evals["n"] == n0 + 1
    assert torch.allclose(vals[0], vals[-1], atol=1e-6)


def test_row_dedup_matches_rowwise_marginal():
    """A batch where each x repeats across rays must give exactly the same
    statistic as the per-row computation on the unrepeated batch."""
    sim, tm = _fit_tiny()
    proc = tm.procedure
    x_base = sim.sample_x_given_theta((0.0, 1.0), 5, np.random.default_rng(5))
    theta = torch.randn(5, 2)
    t_single = proc.test_statistic(theta, x_base)            # (5,)
    # repeat each x 7x with matching theta rows
    x_rep = x_base.repeat_interleave(7, dim=0)
    th_rep = theta.repeat_interleave(7, dim=0)
    t_rep = proc.test_statistic(th_rep, x_rep).view(5, 7)
    assert torch.allclose(t_rep, t_single.unsqueeze(1).expand(-1, 7), atol=1e-5)
