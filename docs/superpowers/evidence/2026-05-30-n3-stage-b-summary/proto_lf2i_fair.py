"""FAIR LF2I-BFF baseline on (μ,Σ) d=5 after the marginal fix: BFF marginal is now
MC over draws from the TRUE prior (simulator.sample), d-aware count marginal_n=2048
(floored at 128·d=640 → 2048). Removes both the d-blind N=64 sparsity AND the
box-uniform extrapolation confound. Compare to the unfixed run (0.20–0.39) and to
CD-SBI Stage-A oracle (~0.026)."""
import numpy as np
import torch
from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
from cdsbi.methods.lf2i_bff import LF2IBFFRunner

torch.manual_seed(0)
sim = NormalBivariateUnknownCov()   # now exposes theta_range (bounding box) natively

runner = LF2IBFFRunner(classifier_hidden=128, classifier_depth=2,
                       quantile_hidden=64, quantile_depth=2,
                       marginal_n=2048, device="auto")
alpha_grid = [0.5, 0.68, 0.9, 0.95]
config = {"lr": 1e-3, "batch_size": 256, "n_steps": 10000, "n_train": 40000,
          "fresh_batch": True, "optimizer": "adam", "weight_decay": 0.0,
          "betas": [0.9, 0.999], "momentum": 0.9, "lr_schedule": "constant",
          "warmup_steps": 0, "lr_min_ratio": 0.0, "lr_gamma": 0.999,
          "batching": "random_replacement", "grad_clip_norm": 5.0,
          "n_train_stat": 40000, "n_train_quantile": 20000, "alpha_grid": alpha_grid}
trained = runner.fit(simulator=sim, config=config, seed=0)
proc = trained.procedure
print(f"FIXED LF2I-BFF — marginal_n_effective={trained.arch_metadata['marginal_n_effective']}, "
      f"estimator={trained.arch_metadata['marginal_estimator']}, wall {trained.wall_clock_sec:.0f}s\n")

grid = [(0., 0., 0., 0., 0.), (0.6, -0.5, -0.9, 2.0, -2.0), (-0.6, 0.5, 1.0, -2.0, 2.5)]
print("coverage (empirical vs nominal α); err = |emp-α|")
worst = 0.0
for th0 in grid:
    xv = sim.sample_x_given_theta(th0, 2000, np.random.default_rng(7))
    row = []
    for a in alpha_grid:
        inside = proc.contains_batch(th0, xv, a).float().mean().item()
        err = abs(inside - a); worst = max(worst, err)
        row.append(f"α={a}: {inside:.3f}(e{err:.3f})")
    print(f"  θ0={tuple(round(t,1) for t in th0)}: " + "  ".join(row))
print(f"\ncoverage_error_max (3-pt grid) = {worst:.3f}  [unfixed was 0.393]")

rng = np.random.default_rng(2)
lows = np.array([np.log(0.4), np.log(0.4), -1.5, -3, -3])
highs = np.array([np.log(2.5), np.log(2.5), 1.5, 3, 3])
lhs = lows + (highs - lows) * rng.random((16, 5))
worst_lhs = 0.0
for th0 in lhs:
    xv = sim.sample_x_given_theta(tuple(th0), 1500, np.random.default_rng(7))
    for a in alpha_grid:
        inside = proc.contains_batch(tuple(th0), xv, a).float().mean().item()
        worst_lhs = max(worst_lhs, abs(inside - a))
print(f"coverage_error_max (16-pt LHS over true prior) = {worst_lhs:.3f}  [unfixed was 0.201]")
print("\nCD-SBI Stage-A (oracle Bartlett) coverage ~0.026 for reference.")
