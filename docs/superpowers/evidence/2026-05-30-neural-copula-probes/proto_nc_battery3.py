"""Neural-copula probing round 2: dimensional scaling + a hard benchmark.
 A) HIGHER-D (d_θ=10): 5 independent Gaussians, unknown mean+logvar each, n_iid=10
    (regular, suff-stat exists but d_θ=10) — tests parameter-dimension scaling.
 B) SLCP (d_θ=5, n_iid=4): canonical SBI benchmark. Nonlinear map θ→(mean,cov) with
    SIGN symmetry (θ3,θ4 enter squared → multimodal) and Fisher degeneracy near 0 —
    a real regularity stress test for the score/Rao CD.
For each: NLE conditional flow + score CD (asymptotic χ²_{d_θ}) + grid-free calibrated
score-norm. Report coverage_error_max. (Rough; single seed.)"""
import numpy as np, torch, math
from scipy.stats import chi2
from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.methods.lf2i import MultiQuantileMLP, train_multi_quantile_head

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALPHAS = [0.5, 0.68, 0.9, 0.95]
QCFG = {"lr":1e-3,"n_steps":4000,"batch_size":512,"optimizer":"adam","lr_schedule":"constant",
        "grad_clip_norm":5.0,"warmup_steps":0,"lr_min_ratio":0.0,"lr_gamma":0.999,
        "weight_decay":0.0,"betas":[0.9,0.999],"momentum":0.9}


def run(name, sample, sample_x_given, d_theta, d_x, grid, n_steps=12000, asinh=False):
    torch.manual_seed(0)
    flow = MAFAdapter(features=d_x, context_features=d_theta, hidden=128, num_layers=8).to(DEV)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3); rng = np.random.default_rng(0)
    flow.train()
    for _ in range(n_steps):
        th, x = sample(256, rng)
        if asinh: x = torch.asinh(x)
        loss = -flow.log_prob(x.to(DEV), context=th.to(DEV)).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
    flow.eval()
    print(f"\n===== {name} (d_θ={d_theta}, d_x={d_x}) — NLE -logq={loss.item():.2f} =====")

    def score(theta_rows, x):
        if asinh: x = torch.asinh(x)
        th = theta_rows.clone().detach().to(DEV).requires_grad_(True)
        lp = flow.log_prob(x.to(DEV), context=th)
        g, = torch.autograd.grad(lp.sum(), th); return g.detach()

    def fisher(th0, n=8000):
        xf = sample_x_given(th0, n, np.random.default_rng(999))
        thr = torch.tensor([th0], dtype=torch.float32).expand(n, d_theta)
        U = score(thr, xf); return (U.T @ U / n).cpu().numpy()

    # asymptotic Rao χ²
    wa = 0.0
    for th0 in grid:
        Iinv = np.linalg.inv(fisher(th0) + 1e-5*np.eye(d_theta))
        xv = sample_x_given(th0, 3000, np.random.default_rng(7))
        thr = torch.tensor([th0], dtype=torch.float32).expand(3000, d_theta)
        U = score(thr, xv).cpu().numpy(); stat = np.einsum('ni,ij,nj->n', U, Iinv, U)
        for a in ALPHAS: wa = max(wa, abs((stat <= chi2.ppf(a, df=d_theta)).mean() - a))
    print(f"  asymptotic-score coverage_error_max = {wa:.3f}")

    # calibrated score-norm
    th_cal, x_cal = sample(40000, np.random.default_rng(11))
    stat_cal = (score(th_cal, x_cal)**2).sum(1)
    qnet = MultiQuantileMLP(input_dim=d_theta, hidden=128, depth=2, n_quantiles=len(ALPHAS)).to(DEV)
    train_multi_quantile_head(qnet, th_cal.to(DEV), stat_cal.to(DEV), ALPHAS, QCFG, DEV); qnet.eval()
    wc = 0.0
    for th0 in grid:
        xv = sample_x_given(th0, 3000, np.random.default_rng(7))
        thr = torch.tensor([th0], dtype=torch.float32).expand(3000, d_theta)
        stat = (score(thr, xv)**2).sum(1).cpu().numpy()
        with torch.no_grad():
            cv = qnet(torch.tensor([th0], dtype=torch.float32, device=DEV))[0].cpu().numpy()
        for k, a in enumerate(ALPHAS): wc = max(wc, abs((stat <= cv[k]).mean() - a))
    print(f"  calibrated-score coverage_error_max = {wc:.3f}")


# ---- A) higher-d: 5 indep Gaussians, unknown (mean, logvar) each → d_θ=10 ----
D = 5
def gA_sample_x(theta_np, rng):
    mu = theta_np[:, :D]; sig = np.exp(theta_np[:, D:])
    z = rng.standard_normal((theta_np.shape[0], 10, D))
    return (mu[:, None, :] + sig[:, None, :] * z).reshape(theta_np.shape[0], 10 * D)
def gA_draw(n, rng):
    mu = rng.uniform(-3, 3, (n, D)); lg = rng.uniform(math.log(0.4), math.log(2.5), (n, D))
    return np.concatenate([mu, lg], 1)
def gA_sample(n, rng):
    th = gA_draw(n, rng); return torch.tensor(th, dtype=torch.float32), torch.tensor(gA_sample_x(th, rng), dtype=torch.float32)
def gA_xgiven(th0, n, rng):
    th = np.tile(np.array(th0, dtype=np.float64), (n, 1)); return torch.tensor(gA_sample_x(th, rng), dtype=torch.float32)
gridA = [tuple([0.]*10), tuple([1.5]*D + [-0.5]*D), tuple([-1.5]*D + [0.6]*D),
         tuple(list(np.linspace(-2,2,D)) + list(np.linspace(-0.6,0.6,D)))]
run("A: higher-d 5×Gaussian(μ,logσ)", gA_sample, gA_xgiven, 10, 50, gridA)

# ---- B) SLCP (d_θ=5, n_iid=4, d_x=8) ----
def slcp_x(theta_np, rng):
    n = theta_np.shape[0]; t = theta_np
    m = t[:, :2]; s1 = t[:, 2]**2; s2 = t[:, 3]**2; rho = np.tanh(t[:, 4])
    cov = np.zeros((n, 2, 2)); cov[:,0,0]=s1**2; cov[:,1,1]=s2**2; cov[:,0,1]=rho*s1*s2; cov[:,1,0]=cov[:,0,1]
    L = np.linalg.cholesky(cov + 1e-6*np.eye(2))
    z = rng.standard_normal((n, 4, 2))
    x = m[:, None, :] + np.einsum('nij,nkj->nki', L, z)
    return x.reshape(n, 8)
def slcp_draw(n, rng): return rng.uniform(-3, 3, (n, 5))
def slcp_sample(n, rng):
    th = slcp_draw(n, rng); return torch.tensor(th, dtype=torch.float32), torch.tensor(slcp_x(th, rng), dtype=torch.float32)
def slcp_xgiven(th0, n, rng):
    th = np.tile(np.array(th0, dtype=np.float64), (n, 1)); return torch.tensor(slcp_x(th, rng), dtype=torch.float32)
gridB = [(0.5,-0.5,1.2,1.0,0.3), (-1.0,1.0,1.5,0.8,-0.4), (1.5,1.5,0.8,1.2,0.5),
         (0.,0.,1.0,1.0,0.), (-1.5,-1.0,1.3,0.9,-0.2)]
run("B: SLCP benchmark", slcp_sample, slcp_xgiven, 5, 8, gridB)
print("\n(SLCP: θ3,θ4 enter squared → sign-symmetry multimodality + Fisher degeneracy near 0.)")
