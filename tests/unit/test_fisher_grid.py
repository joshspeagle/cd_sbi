"""Amortized grid Fisher (perf hardening): correctness + zero per-query sims."""
import numpy as np
import torch

from cdsbi.methods.score_cd import ScoreCDRunner, _FisherGrid
from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.simulators.cauchy_loc_scale import CauchyLocScale


def test_multilinear_exact_on_linear_table():
    """Multilinear interpolation reproduces any theta-LINEAR function exactly —
    build a synthetic table whose entries are linear in theta and check interior
    queries against the closed form (then the inverse of it)."""
    axes = [torch.linspace(-1.0, 1.0, 5), torch.linspace(-2.0, 2.0, 5)]
    def I_of(t0, t1):
        return torch.tensor([[2.0 + 0.5 * t0, 0.1 * t1], [0.1 * t1, 3.0 - 0.25 * t0]],
                            dtype=torch.float64)
    table = torch.stack([
        torch.stack([I_of(float(a), float(b)) for b in axes[1]]) for a in axes[0]
    ])
    grid = _FisherGrid(axes, table)
    for t0, t1 in [(-0.3, 1.7), (0.62, -1.11), (0.0, 0.0), (1.0, 2.0)]:
        got = grid.fisher_inv(torch.tensor([t0, t1]))
        want = torch.linalg.inv(I_of(t0, t1)).to(torch.float32)
        assert torch.allclose(got, want, atol=1e-5), (t0, t1)


def test_clamps_outside_box():
    axes = [torch.linspace(0.0, 1.0, 3)]
    table = torch.stack([torch.eye(1, dtype=torch.float64) * (1 + k) for k in range(3)])
    grid = _FisherGrid(axes, table)
    assert torch.allclose(grid.fisher_inv(torch.tensor([99.0])),
                          grid.fisher_inv(torch.tensor([1.0])))


def test_grid_mode_zero_sims_at_inference():
    """The point of the fix: after fit, set construction must draw ZERO
    additional simulator calls (the old exact-MC path drew 4000 per probed θ)."""
    torch.manual_seed(0); np.random.seed(0)
    sim = CauchyLocScale()
    flow = MAFAdapter(features=10, context_features=2, hidden=16, num_layers=2)
    runner = ScoreCDRunner(flow, variant="rao", fisher_n=200, fisher_grid_per_dim=4)
    cfg = dict(optimizer="adam", lr=3e-3, batch_size=128, n_steps=30, n_train=512,
               fresh_batch=False, grad_clip_norm=5.0, alpha_grid=[0.9])
    tm = runner.fit(sim, cfg, seed=0)
    proc = tm.procedure
    after_fit = proc.inference_sim_calls["fisher"]
    assert after_fit == 4 * 4 * 200  # n_per_dim^d grid points x fisher_n
    x = sim.sample_x_given_theta((0.0, 0.0), 4, np.random.default_rng(1))
    proc.test_statistic(torch.zeros(4, 2), x)
    proc.confidence_set(x[:1], alpha=0.9)
    assert proc.inference_sim_calls["fisher"] == after_fit  # ZERO new draws
