"""Clean (no-overfit) read of the I-A structural limit: fresh_batch=True (infinite
data → loss cannot dip below H(X|θ) by memorization). Base arch (h64, 6 layers).
Report sufficiency (5 dirs) + per-coord PIT (center+2 extremes) + loss-vs-floor.
This is the fair characterization of whether I-A recovers the 2-D covariance."""
import numpy as np
import torch
from scipy.stats import spearmanr, kstest, norm

from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
from cdsbi.conditioners.invertible_summary import InvertibleSummaryConditioner
from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
from cdsbi.methods.cd_sbi_exact_density import ExactDensityCDSBIRunner

ORACLE = ["log_D11", "log_D22", "D21", "xbar1", "xbar2"]

torch.manual_seed(0)
sim = NormalBivariateUnknownCov()
flow = SingleIndexMonotoneFlow(d=5, theta_signs=list(sim.theta_signs),
                               feat_signs=list(sim.feat_signs), hidden=64, depth=2)
cond = InvertibleSummaryConditioner(n_iid=sim.d_x, d_theta=5, hidden=64, n_layers=6, depth=2)
runner = ExactDensityCDSBIRunner(flow=flow, conditioner=cond)
config = {"lr": 2e-3, "n_steps": 15000, "batch_size": 256, "fresh_batch": True,
          "grad_clip_norm": 5.0}
tr = runner.fit(simulator=sim, config=config, seed=0)
proc = tr.procedure

rng = np.random.default_rng(123)
_, x = sim.sample(4000, rng)
with torch.no_grad():
    S = proc.encode_fn(x).cpu().numpy()
    oracle = sim.oracle_summary(x).cpu().numpy()
print("=== sufficiency (fresh-batch, no overfit) ===")
for k, nm in enumerate(ORACLE):
    sps = [abs(spearmanr(oracle[:, k], S[:, j]).statistic) for j in range(5)]
    print(f"  {nm:8s} {np.nanmax(sps):.3f} (S[{int(np.nanargmax(sps))}])")
print(f"final_loss={tr.final_loss:.3f}  floor H(X|θ)={sim.data_entropy_lower_bound():.3f}  "
      f"{'no-cheat' if tr.final_loss > sim.data_entropy_lower_bound() else 'CHEAT'}")

grid = [(0., 0., 0., 0., 0.), (0.6, -0.5, -0.9, 2.0, -2.0), (-0.6, 0.5, 1.0, -2.0, 2.5)]
ks = np.zeros((3, 5))
for i, th0 in enumerate(grid):
    xv = sim.sample_x_given_theta(th0, 4000, np.random.default_rng(7))
    thv = torch.tensor([th0], dtype=xv.dtype).expand(4000, 5)
    with torch.no_grad():
        r = proc.pivot(thv, xv).cpu().numpy()
    ks[i] = [kstest(norm.cdf(r[:, k]), "uniform").statistic for k in range(5)]
print("=== per-coord PIT KS (rows center/extreme/extreme; cols r1..r5) ===")
print(ks.round(3))
