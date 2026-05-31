"""Full (μ,Σ) comparison via the new eval engine — CD-SBI vs Score-CD(rao,cal) across
budgets. Training via the real runners at each budget's model size; coverage via the
chunked engine (eval ~1s, no OOM). fresh_batch=False, n_train=20000 (μΣ regime)."""
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
BUD = {"small":(20,14),"medium":(48,32),"large":(110,76),"xlarge":(222,156)}
SEEDS = [0,1,2]


def recipe(lr, opt, **extra):
    r = {"lr":lr,"batch_size":256,"n_steps":12000,"n_train":20000,"fresh_batch":False,
         "optimizer":opt,"weight_decay":0.0,"betas":[0.9,0.999],"momentum":0.9,
         "lr_schedule":"constant","warmup_steps":0,"lr_min_ratio":0.0,"lr_gamma":0.999,
         "batching":"random_replacement","grad_clip_norm":5.0}
    r.update(extra); return r


def ev(proc, seed):
    return evaluate_coverage(proc, sim, GRID, ALPHAS, n_per_theta=2000, chunk_size=256, seed=seed)["coverage_error_max"]


res = {}
for budget, (ch, mh) in BUD.items():
    for seed in SEEDS:
        t0 = time.time()
        # CD-SBI (single_index pivot)
        torch.manual_seed(seed)
        f = SingleIndexMonotoneFlow(d=5, theta_signs=list(sim.theta_signs), feat_signs=list(sim.feat_signs), hidden=ch, depth=2)
        tr = CDSBIRunner(flow=f, conditioner=BartlettSummaryConditioner(n_iid=sim.n_iid), loss=NFMLELoss()).fit(simulator=sim, config=recipe(3e-3,"adamw"), seed=seed)
        res.setdefault(("cd_sbi",budget),[]).append(ev(tr.procedure, seed))
        # Score-CD rao
        torch.manual_seed(seed)
        m = MAFAdapter(features=sim.d_x, context_features=sim.d_theta, hidden=mh, num_layers=2)
        tr = ScoreCDRunner(flow=m, variant="rao", fisher_n=2000).fit(simulator=sim, config=recipe(1e-3,"adam",n_train_quantile=10000,alpha_grid=ALPHAS), seed=seed)
        res.setdefault(("score_cd_rao",budget),[]).append(ev(tr.procedure, seed))
        # Score-CD cal
        torch.manual_seed(seed)
        m = MAFAdapter(features=sim.d_x, context_features=sim.d_theta, hidden=mh, num_layers=2)
        tr = ScoreCDRunner(flow=m, variant="cal", fisher_n=2000).fit(simulator=sim, config=recipe(1e-3,"adam",n_train_quantile=10000,alpha_grid=ALPHAS), seed=seed)
        res.setdefault(("score_cd_cal",budget),[]).append(ev(tr.procedure, seed))
        print(f"[{budget} seed{seed}] cd_sbi={res[('cd_sbi',budget)][-1]:.3f} "
              f"rao={res[('score_cd_rao',budget)][-1]:.3f} cal={res[('score_cd_cal',budget)][-1]:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

print("\n=== (μ,Σ) coverage_error_max (mean over seeds), by budget ===", flush=True)
print(f"{'method':14s} " + " ".join(f"{b:>7s}" for b in BUD), flush=True)
for meth in ["cd_sbi","score_cd_rao","score_cd_cal"]:
    print(f"{meth:14s} " + " ".join(f"{np.mean(res[(meth,b)]):7.3f}" for b in BUD), flush=True)
