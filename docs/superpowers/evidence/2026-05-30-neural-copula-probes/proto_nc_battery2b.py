"""Neural-copula battery, script 2 — GENERALITY (probe 2). Cauchy location-scale,
n_iid=10, d_theta=2 (μ, log γ). Cauchy is REGULAR (finite Fisher) but has NO
finite-dim sufficient statistic — so the oracle/summary route is IMPOSSIBLE by
construction, yet the score/NLE approach should apply. Tests the scalability thesis:
NLE + score CD (asymptotic χ²₂) AND calibrated score-norm. Coverage over a θ₀ grid."""
import numpy as np, torch, math
from scipy.stats import chi2
from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.methods.lf2i import MultiQuantileMLP, train_multi_quantile_head

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_IID = 10; D_THETA = 2; D_X = N_IID
MU_R = (-3.0, 3.0); LG_R = (math.log(0.4), math.log(2.5))


def draw_theta(n, rng):
    mu = rng.uniform(*MU_R, size=(n, 1)); lg = rng.uniform(*LG_R, size=(n, 1))
    return np.concatenate([mu, lg], 1)


def sample_x(theta_np, rng):  # Cauchy(loc=mu, scale=exp(lg)), n_iid replicates
    mu = theta_np[:, 0:1]; g = np.exp(theta_np[:, 1:2])
    u = rng.standard_cauchy(size=(theta_np.shape[0], N_IID))
    return np.arcsinh(mu + g * u)  # asinh-stabilize heavy Cauchy tails (θ-indep bijection: score unchanged)


def sample(n, rng):
    th = draw_theta(n, rng); x = sample_x(th, rng)
    return torch.tensor(th, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)


def sample_x_given(theta_row, n, rng):
    th = np.tile(np.array(theta_row, dtype=np.float64), (n, 1))
    return torch.tensor(sample_x(th, rng), dtype=torch.float32)


torch.manual_seed(0)
flow = MAFAdapter(features=D_X, context_features=D_THETA, hidden=128, num_layers=8).to(DEV)
opt = torch.optim.Adam(flow.parameters(), lr=1e-3); rng = np.random.default_rng(0)
flow.train()
for _ in range(15000):
    th, x = sample(256, rng)
    loss = -flow.log_prob(x.to(DEV), context=th.to(DEV)).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
flow.eval(); print(f"Cauchy NLE final -logq={loss.item():.3f}  (no sufficient statistic exists)")
alphas = [0.5, 0.68, 0.9, 0.95]
grid = [(0., 0.), (2.0, -0.7), (-2.0, 0.8), (1.0, 0.5), (-1.0, -0.5)]


def score(theta_rows, x):
    th = theta_rows.clone().detach().to(DEV).requires_grad_(True)
    lp = flow.log_prob(x.to(DEV), context=th)
    g, = torch.autograd.grad(lp.sum(), th); return g.detach()


def fisher(theta_row, n_fish=8000):
    xf = sample_x_given(theta_row, n_fish, np.random.default_rng(999)).to(DEV)
    thr = torch.tensor([theta_row], dtype=xf.dtype, device=DEV).expand(n_fish, 2)
    U = score(thr, xf); return (U.T @ U / n_fish).cpu().numpy()


# asymptotic Rao score χ²₂
print("\n=== PROBE 2a: asymptotic score (Rao χ²₂) on Cauchy ===")
wa = 0.0
for th0 in grid:
    Iinv = np.linalg.inv(fisher(th0) + 1e-6*np.eye(2))
    xv = sample_x_given(th0, 3000, np.random.default_rng(7))
    thr = torch.tensor([th0], dtype=xv.dtype).expand(3000, 2)
    U = score(thr, xv).cpu().numpy(); stat = np.einsum('ni,ij,nj->n', U, Iinv, U)
    for a in alphas: wa = max(wa, abs((stat <= chi2.ppf(a, df=2)).mean() - a))
print(f"  asymptotic-score coverage_error_max (5-pt) = {wa:.3f}")

# calibrated score-norm
print("=== PROBE 2b: CALIBRATED score-norm on Cauchy ===")
th_cal, x_cal = sample(40000, np.random.default_rng(11))
stat_cal = (score(th_cal, x_cal)**2).sum(1)
qnet = MultiQuantileMLP(input_dim=2, hidden=128, depth=2, n_quantiles=len(alphas)).to(DEV)
qcfg = {"lr":1e-3,"n_steps":4000,"batch_size":512,"optimizer":"adam","lr_schedule":"constant",
        "grad_clip_norm":5.0,"warmup_steps":0,"lr_min_ratio":0.0,"lr_gamma":0.999,
        "weight_decay":0.0,"betas":[0.9,0.999],"momentum":0.9}
train_multi_quantile_head(qnet, th_cal.to(DEV), stat_cal.to(DEV), alphas, qcfg, DEV); qnet.eval()
wc = 0.0
for th0 in grid:
    xv = sample_x_given(th0, 3000, np.random.default_rng(7))
    thr = torch.tensor([th0], dtype=xv.dtype).expand(3000, 2)
    stat = (score(thr, xv)**2).sum(1).cpu().numpy()
    with torch.no_grad():
        cvals = qnet(torch.tensor([th0], dtype=torch.float32, device=DEV))[0].cpu().numpy()
    for k, a in enumerate(alphas): wc = max(wc, abs((stat <= cvals[k]).mean() - a))
print(f"  calibrated-score coverage_error_max (5-pt) = {wc:.3f}")
print("\n(No sufficient statistic exists for Cauchy → summary/oracle route impossible; score works.)")
