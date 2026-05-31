"""Decisive correctness check: engine coverage == legacy Coverage diagnostic on a REAL
trained model (CD-SBI pivot and Score-CD critical-value), same theta0 grid + same X.
If they agree, the engine is faithful and the muSigma numbers are real (not a bug)."""
import numpy as np, torch
from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
from cdsbi.conditioners.bartlett_summary import BartlettSummaryConditioner
from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.losses.nfmle import NFMLELoss
from cdsbi.methods.cd_sbi import CDSBIRunner
from cdsbi.methods.score_cd import ScoreCDRunner
from cdsbi.diagnostics.engine import evaluate_coverage
from cdsbi.diagnostics.coverage import Coverage

sim = NormalBivariateUnknownCov()
GRID = [(0.0,0.0,0.0,0.0,0.0),(0.4,-0.4,0.8,1.5,-1.5),(-0.4,0.4,-0.8,-1.5,1.5)]
ALPHAS = [0.5,0.68,0.9,0.95]
cfg = {"lr":3e-3,"batch_size":256,"n_steps":4000,"n_train":20000,"optimizer":"adamw","fresh_batch":False}


def compare(name, trained):
    rng = np.random.default_rng(0); xpt = {}
    for t0 in GRID:
        rep = str(list(map(float, t0)))
        xpt[rep] = sim.sample_x_given_theta(t0, 1500, rng)
    leg = Coverage(theta_0_grid=GRID, alpha_grid=ALPHAS, n_per_theta=1500)(trained, sim, x_per_theta=xpt)
    legdf = leg.value
    eng = evaluate_coverage(trained.procedure, sim, GRID, ALPHAS, n_per_theta=1500, chunk_size=256, seed=0)
    engdf = eng["coverage"]
    keys = [c for c in legdf.columns if c.startswith("theta_0_")] + ["alpha"]
    m = legdf.merge(engdf, on=keys, suffixes=("_leg","_eng"))
    d = (m["empirical_leg"] - m["empirical_eng"]).abs()
    print(f"[{name}] max|delta|={d.max():.4f} mean|delta|={d.mean():.4f} (pairs={len(m)})  "
          f"legacy_cov_err={(legdf['empirical']-legdf['nominal']).abs().max():.3f} "
          f"engine_cov_err={eng['coverage_error_max']:.3f}", flush=True)


torch.manual_seed(0)
f = SingleIndexMonotoneFlow(d=5, theta_signs=list(sim.theta_signs), feat_signs=list(sim.feat_signs), hidden=48, depth=2)
tr = CDSBIRunner(flow=f, conditioner=BartlettSummaryConditioner(n_iid=sim.n_iid), loss=NFMLELoss()).fit(simulator=sim, config=cfg, seed=0)
compare("CD-SBI muSigma (pivot)", tr)

torch.manual_seed(0)
m = MAFAdapter(features=sim.d_x, context_features=sim.d_theta, hidden=32, num_layers=2)
tr2 = ScoreCDRunner(flow=m, variant="rao", fisher_n=2000).fit(simulator=sim, config={**cfg,"lr":1e-3,"optimizer":"adam"}, seed=0)
compare("Score-CD muSigma (rao)", tr2)
