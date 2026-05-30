"""Neural-copula battery, script 1 — (μ,Σ) d=5. From ONE trained NLE:
 PROBE 3: full-latent pivot ‖ε(X;θ)‖²~χ²_dx (exact if flow exact, but d_x=20 — inefficient).
 PROBE 1: CALIBRATED score — stat=‖U(θ;X)‖², critical value c_α(θ) learned by quantile
          regression (LF2I-style, grid-free). Does calibration close the score's 0.06→~0.03 gap?
Compare to: asymptotic score 0.060, oracle Bartlett 0.026."""
import numpy as np, torch
from scipy.stats import chi2
from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.methods.lf2i import MultiQuantileMLP, train_multi_quantile_head

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
sim = NormalBivariateUnknownCov()
flow = MAFAdapter(features=sim.d_x, context_features=sim.d_theta, hidden=128, num_layers=8).to(DEV)
opt = torch.optim.Adam(flow.parameters(), lr=1e-3); rng = np.random.default_rng(0)
flow.train()
for _ in range(15000):
    th, x = sim.sample(256, rng)
    loss = -flow.log_prob(x.to(DEV), context=th.to(DEV)).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
flow.eval(); print(f"NLE final -logq={loss.item():.3f}")
alphas = [0.5, 0.68, 0.9, 0.95]
grid = [(0.,0.,0.,0.,0.), (0.6,-0.5,-0.9,2.0,-2.0), (-0.6,0.5,1.0,-2.0,2.5)]


def score(theta_rows, x):  # paired: theta_rows (n,5), x (n,20) -> U (n,5)
    th = theta_rows.clone().detach().to(DEV).requires_grad_(True)
    lp = flow.log_prob(x.to(DEV), context=th)
    g, = torch.autograd.grad(lp.sum(), th); return g.detach()


# PROBE 3: full-latent χ²_20
print("\n=== PROBE 3: full-latent ‖ε‖²~χ²_20 (exact-but-inefficient) ===")
w3 = 0.0
for th0 in grid:
    xv = sim.sample_x_given_theta(th0, 3000, np.random.default_rng(7)).to(DEV)
    thr = torch.tensor([th0], dtype=xv.dtype, device=DEV).expand(3000, 5)
    with torch.no_grad():
        z = flow.flow.transform_to_noise(xv, context=thr).cpu().numpy()
    stat = (z**2).sum(1)
    for a in alphas: w3 = max(w3, abs((stat <= chi2.ppf(a, df=sim.d_x)).mean() - a))
print(f"  full-latent coverage_error_max (3-pt) = {w3:.3f}  [d_x=20 pivot]")

# PROBE 1: calibrated score-norm
print("\n=== PROBE 1: CALIBRATED score-norm stat=‖U‖², learned c_α(θ) ===")
th_cal, x_cal = sim.sample(40000, np.random.default_rng(11))
U = score(th_cal, x_cal); stat_cal = (U**2).sum(1)        # (40000,)
qnet = MultiQuantileMLP(input_dim=5, hidden=128, depth=2, n_quantiles=len(alphas)).to(DEV)
qcfg = {"lr":1e-3,"n_steps":4000,"batch_size":512,"optimizer":"adam","lr_schedule":"constant",
        "grad_clip_norm":5.0,"warmup_steps":0,"lr_min_ratio":0.0,"lr_gamma":0.999,
        "weight_decay":0.0,"betas":[0.9,0.999],"momentum":0.9}
train_multi_quantile_head(qnet, th_cal.to(DEV), stat_cal.to(DEV), alphas, qcfg, DEV)
qnet.eval()
w1 = 0.0
for th0 in grid:
    xv = sim.sample_x_given_theta(th0, 3000, np.random.default_rng(7))
    thr = torch.tensor([th0], dtype=xv.dtype).expand(3000, 5)
    stat = (score(thr, xv)**2).sum(1).cpu().numpy()
    with torch.no_grad():
        cvals = qnet(torch.tensor([th0], dtype=torch.float32, device=DEV))[0].cpu().numpy()
    for k, a in enumerate(alphas): w1 = max(w1, abs((stat <= cvals[k]).mean() - a))
print(f"  calibrated-score coverage_error_max (3-pt) = {w1:.3f}  [asymptotic was 0.060; oracle 0.026]")
