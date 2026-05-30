"""N3 risk-gate: does the I-A invertible learned summary (ℝ²⁰→ℝ⁵) recover ALL
FIVE sufficient directions for (μ,Σ) — crucially the 3 COVARIANCE directions
(log D₁₁, log D₂₂, and the cross-covariance D₂₁), the analog of σ² in 1-D?

Mirrors the verified 1-D Arm I-A (AffineCouplingBijection + InvertibleSummary +
ExactDensityCDSBIRunner) at d_x=20, d_theta=5. Checks:
  (1) SufficiencyRecovery: per-oracle-coord max|Spearman| over the 5 learned S
      features (the verdict metric; >0.9 each = recovered).
  (2) calibration: joint Mahalanobis ‖r‖²~χ²₅ KS at truth.
  (3) no-cheat: final loss > data_entropy_lower_bound H(X|θ).
If (1) passes on the covariance dirs, the N3 design is grounded → write the plan.
If D₂₁ collapses, that's a fork (architecture: more layers / different mask / aug)."""
import numpy as np
import torch
from scipy.stats import spearmanr, kstest, chi2

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

# A bit longer than the 1-D recipe (d_x 10→20, d_theta 2→5 is harder).
config = {"lr": 2e-3, "n_steps": 12000, "batch_size": 256, "n_train": 20000,
          "fresh_batch": False, "grad_clip_norm": 5.0}
trained = runner.fit(simulator=sim, config=config, seed=0)
proc = trained.procedure

# (1) SufficiencyRecovery
rng = np.random.default_rng(123)
_, x = sim.sample(4000, rng)
with torch.no_grad():
    S = proc.encode_fn(x).cpu().numpy()                 # (n,5) learned summary
    oracle = sim.oracle_summary(x).cpu().numpy()        # (n,5) Bartlett features
print("=== (1) SufficiencyRecovery: max|Spearman(oracle_k, S_j)| over j ===")
sp_min = 1.0
for k, nm in enumerate(ORACLE):
    sps = [abs(spearmanr(oracle[:, k], S[:, j]).statistic) for j in range(5)]
    best = int(np.nanargmax(sps)); val = float(np.nanmax(sps))
    sp_min = min(sp_min, val)
    print(f"  {nm:8s}: {val:.3f}  (best matches S[{best}])")
print(f"  --> min over 5 dirs = {sp_min:.3f}  {'PASS' if sp_min > 0.9 else 'FAIL'} (>0.9)")

# (2) calibration: ‖r‖²~χ²₅ at truth
xv = sim.sample_x_given_theta((0., 0., 0., 0., 0.), 3000, np.random.default_rng(7))
with torch.no_grad():
    rr = proc.pivot(torch.zeros(3000, 5), xv).cpu().numpy()
ks_chi5 = kstest(chi2.cdf((rr ** 2).sum(1), df=5), "uniform").statistic
print(f"=== (2) joint Mahalanobis ‖r‖²~χ²₅ KS = {ks_chi5:.3f}  {'PASS' if ks_chi5 < 0.06 else 'soft'} (<0.06) ===")

# (3) no-cheat floor
H = sim.data_entropy_lower_bound()
print(f"=== (3) final_loss={trained.final_loss:.3f}  H(X|θ)={H:.3f}  "
      f"{'PASS (no cheat)' if trained.final_loss > H else 'BELOW FLOOR (cheat!)'} ===")
print("\nVERDICT: grounded if covariance dirs (log_D11/log_D22/D21) all >0.9 AND no cheat.")
