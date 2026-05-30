"""N3 fork diagnosis. The base risk-gate showed I-A recovers xbar1/2 + log_D11 but
COLLAPSES log_D22 & D21 (both → S[2]), while aggregate χ²₅ calibration passes. Three
decisive questions:
  (A) Capacity/training: does a BIGGER bijection (hidden128, 10 layers) + more steps
      (20k) recover log_D22 & D21? (under-fit vs structural)
  (B) Routing: is the covariance info merely in A (latent coords 5..19), not S? Check
      max|Spearman| of each oracle coord vs ALL 20 latent z coords. If log_D22/D21
      live in A, the issue is the exact-density loss not ROUTING them to the front.
  (C) Genuine calibration? Per-coordinate PIT at center + 2 EXTREME θ₀ (the N2 lesson:
      aggregate χ²₅ masks per-coord miscalibration). If r-coords for the unrecovered
      covariance dirs are mis-PIT at extremes, the χ²₅-pass was trivial."""
import numpy as np
import torch
from scipy.stats import spearmanr, kstest, norm, chi2

from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
from cdsbi.conditioners.invertible_summary import InvertibleSummaryConditioner
from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
from cdsbi.methods.cd_sbi_exact_density import ExactDensityCDSBIRunner

ORACLE = ["log_D11", "log_D22", "D21", "xbar1", "xbar2"]


def train(hidden, n_layers, n_steps):
    torch.manual_seed(0)
    sim = NormalBivariateUnknownCov()
    flow = SingleIndexMonotoneFlow(d=5, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=64, depth=2)
    cond = InvertibleSummaryConditioner(n_iid=sim.d_x, d_theta=5, hidden=hidden,
                                        n_layers=n_layers, depth=2)
    runner = ExactDensityCDSBIRunner(flow=flow, conditioner=cond)
    config = {"lr": 2e-3, "n_steps": n_steps, "batch_size": 256, "n_train": 20000,
              "fresh_batch": False, "grad_clip_norm": 5.0}
    return sim, cond, runner.fit(simulator=sim, config=config, seed=0)


def suff(proc, sim, cond, full=False):
    rng = np.random.default_rng(123)
    _, x = sim.sample(4000, rng)
    with torch.no_grad():
        S = proc.encode_fn(x).cpu().numpy()
        zfull, _ = cond.transform(x.to(next(cond.parameters()).device))
        zfull = zfull.cpu().numpy()
        oracle = sim.oracle_summary(x).cpu().numpy()
    target = zfull if full else S
    out = {}
    for k, nm in enumerate(ORACLE):
        sps = [abs(spearmanr(oracle[:, k], target[:, j]).statistic) for j in range(target.shape[1])]
        out[nm] = (float(np.nanmax(sps)), int(np.nanargmax(sps)))
    return out


def percoord_pit(proc, sim):
    grid = [(0., 0., 0., 0., 0.), (0.6, -0.5, -0.9, 2.0, -2.0), (-0.6, 0.5, 1.0, -2.0, 2.5)]
    ks = np.zeros((3, 5))
    for i, th0 in enumerate(grid):
        xv = sim.sample_x_given_theta(th0, 4000, np.random.default_rng(7))
        thv = torch.tensor([th0], dtype=xv.dtype).expand(4000, 5)
        with torch.no_grad():
            r = proc.pivot(thv, xv).cpu().numpy()
        ks[i] = [kstest(norm.cdf(r[:, k]), "uniform").statistic for k in range(5)]
    return ks


print("===== (A) base (hidden64, 6 layers, 12k) =====")
sim, cond, tr = train(64, 6, 12000)
s = suff(tr.procedure, sim, cond)
for nm in ORACLE: print(f"  {nm:8s} S-recovery {s[nm][0]:.3f} (S[{s[nm][1]}])")
print("===== (A') bigger (hidden128, 10 layers, 20k) =====")
sim2, cond2, tr2 = train(128, 10, 20000)
s2 = suff(tr2.procedure, sim2, cond2)
for nm in ORACLE: print(f"  {nm:8s} S-recovery {s2[nm][0]:.3f} (S[{s2[nm][1]}])")
print(f"  final_loss={tr2.final_loss:.3f} floor={sim2.data_entropy_lower_bound():.3f}")
print("===== (B) bigger: recovery vs ALL 20 latent coords (is cov info in A?) =====")
sf = suff(tr2.procedure, sim2, cond2, full=True)
for nm in ORACLE: print(f"  {nm:8s} full-z recovery {sf[nm][0]:.3f} (z[{sf[nm][1]}])")
print("===== (C) bigger: per-coord PIT KS [rows θ₀ center/extreme/extreme] =====")
ks = percoord_pit(tr2.procedure, sim2)
print("   cols = r1(logD11) r2(logD22) r3(D21) r4(mu1) r5(mu2)")
print(ks.round(3))
print("\nINTERPRET: A'→if cov dirs still <0.9 = structural not capacity.")
print("  B→if log_D22/D21 high in full-z but low in S = ROUTING failure (loss leaves them in A).")
print("  C→if r2/r3 PIT KS large at extremes = calibration was trivial/aggregate-masked.")
