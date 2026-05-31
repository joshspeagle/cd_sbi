"""Verify the eval engine on the exact μΣ cases that broke the legacy path:
 - CD-SBI μΣ (single_index pivot) — legacy OOM'd at 21 GiB on the n_eval precompute.
 - Score-CD μΣ (MAF, rao) — legacy took ~93 min/run.
Train briefly (verifying the ENGINE's speed/memory, not final coverage quality), then
engine-evaluate on the full 16-θ₀ grid × 2000 X with chunking. Success = completes
fast, NO OOM, valid (finite) coverage."""
import time, numpy as np, torch
from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
from cdsbi.conditioners.bartlett_summary import BartlettSummaryConditioner
from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.losses.nfmle import NFMLELoss
from cdsbi.methods.cd_sbi import CDSBIRunner
from cdsbi.methods.score_cd import ScoreCDRunner
from cdsbi.diagnostics.engine import evaluate_coverage

sim = NormalBivariateUnknownCov()
GRID = [(0.271,0.084,0.180,-2.631,-1.055),(-0.219,0.274,0.801,-1.329,1.149),
        (-0.093,-0.344,-1.098,-0.763,2.726),(0.094,0.588,-1.227,-0.487,2.091),
        (0.455,0.788,1.374,0.507,-1.731),(-0.502,0.115,0.941,-0.257,0.506),
        (-0.422,-0.846,1.287,2.354,-1.322),(0.194,-0.743,0.396,-2.600,0.991),
        (-0.295,0.421,-0.111,-2.002,-2.772),(-0.904,0.890,-1.429,1.843,-0.687),
        (0.826,0.545,-0.914,-1.522,0.249),(0.555,-0.052,0.226,1.039,-2.270),
        (0.755,-0.595,-0.205,2.032,1.763),(-0.650,-0.252,-0.552,2.863,-0.040),
        (-0.759,-0.221,0.667,0.017,-2.062),(0.639,-0.529,-0.749,1.144,2.452)]
ALPHAS = [0.5,0.68,0.9,0.95]
cfg = {"lr":3e-3,"batch_size":256,"n_steps":3000,"n_train":20000,"optimizer":"adamw","fresh_batch":False}

print("=== CD-SBI μΣ (single_index pivot) — the OOM case ===", flush=True)
torch.manual_seed(0)
flow = SingleIndexMonotoneFlow(d=5, theta_signs=list(sim.theta_signs), feat_signs=list(sim.feat_signs), hidden=48, depth=2)
cond = BartlettSummaryConditioner(n_iid=sim.n_iid)
tr = CDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss()).fit(simulator=sim, config=cfg, seed=0)
t0=time.time()
out = evaluate_coverage(tr.procedure, sim, GRID, ALPHAS, n_per_theta=2000, chunk_size=256, seed=0)
print(f"  engine eval: {time.time()-t0:.1f}s  cov_err_max={out['coverage_error_max']:.3f}  "
      f"chi2_ks_max={out['joint_chi2_ks_max']:.3f}  NO OOM ✓", flush=True)
print(f"  peak GPU MiB: {torch.cuda.max_memory_allocated()/2**20:.0f}" if torch.cuda.is_available() else "", flush=True)

print("=== Score-CD μΣ (MAF, rao) — the 93-min case ===", flush=True)
if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
torch.manual_seed(0)
maf = MAFAdapter(features=sim.d_x, context_features=sim.d_theta, hidden=64, num_layers=4)
tr2 = ScoreCDRunner(flow=maf, variant="rao", fisher_n=2000).fit(simulator=sim, config=cfg, seed=0)
t0=time.time()
out2 = evaluate_coverage(tr2.procedure, sim, GRID, ALPHAS, n_per_theta=2000, chunk_size=256, seed=0)
print(f"  engine eval: {time.time()-t0:.1f}s  cov_err_max={out2['coverage_error_max']:.3f}  (vs legacy ~80 min eval)", flush=True)
print(f"  peak GPU MiB: {torch.cuda.max_memory_allocated()/2**20:.0f}" if torch.cuda.is_available() else "", flush=True)
