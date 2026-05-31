"""Score-CD with a zuko NSF density model (GPU-efficient, sbi's backend). Repeat SLCP
(MAF baseline 0.115/0.426) + (μ,Σ) (MAF 0.060/0.038). Unbuffered + progress prints."""
import sys, time, numpy as np, torch, zuko
from scipy.stats import chi2
from cdsbi.methods.lf2i import MultiQuantileMLP, train_multi_quantile_head

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALPHAS = [0.5, 0.68, 0.9, 0.95]
QCFG = {"lr":1e-3,"n_steps":4000,"batch_size":512,"optimizer":"adam","lr_schedule":"constant",
        "grad_clip_norm":5.0,"warmup_steps":0,"lr_min_ratio":0.0,"lr_gamma":0.999,
        "weight_decay":0.0,"betas":[0.9,0.999],"momentum":0.9}


def run(name, sample, sample_x_given, d_theta, d_x, grid, n_steps=8000, bs=1024, asinh=True):
    torch.manual_seed(0)
    flow = zuko.flows.NSF(features=d_x, context=d_theta, transforms=6,
                          hidden_features=[128, 128], bins=8).to(DEV)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3); rng = np.random.default_rng(0)
    t0 = time.time(); flow.train()
    for step in range(n_steps):
        th, x = sample(bs, rng)
        if asinh: x = torch.asinh(x)
        loss = -flow(th.to(DEV)).log_prob(x.to(DEV)).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
        if step % 2000 == 0:
            print(f"  [{name}] step {step}/{n_steps}  -logq={loss.item():.2f}  ({time.time()-t0:.0f}s)", flush=True)
    flow.eval()
    print(f"===== {name} (zuko-NSF; d_θ={d_theta}, d_x={d_x}) — NLE -logq={loss.item():.2f}  ({time.time()-t0:.0f}s) =====", flush=True)

    def score(theta_rows, x):
        if asinh: x = torch.asinh(x)
        th = theta_rows.clone().detach().to(DEV).requires_grad_(True)
        g, = torch.autograd.grad(flow(th).log_prob(x.to(DEV)).sum(), th); return g.detach()

    def fisher(th0, n=8000):
        xf = sample_x_given(th0, n, np.random.default_rng(999))
        thr = torch.tensor([th0], dtype=torch.float32).expand(n, d_theta)
        U = score(thr, xf); return (U.T @ U / n).cpu().numpy()

    wa = 0.0
    for th0 in grid:
        Iinv = np.linalg.inv(fisher(th0) + 1e-5*np.eye(d_theta))
        xv = sample_x_given(th0, 3000, np.random.default_rng(7))
        thr = torch.tensor([th0], dtype=torch.float32).expand(3000, d_theta)
        U = score(thr, xv).cpu().numpy(); stat = np.einsum('ni,ij,nj->n', U, Iinv, U)
        for a in ALPHAS: wa = max(wa, abs((stat <= chi2.ppf(a, df=d_theta)).mean() - a))
    print(f"  asymptotic-score coverage_error_max = {wa:.3f}", flush=True)
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
    print(f"  calibrated-score coverage_error_max = {wc:.3f}", flush=True)


def slcp_x(t, rng):
    n = t.shape[0]; m = t[:, :2]; s1 = t[:, 2]**2; s2 = t[:, 3]**2; rho = np.tanh(t[:, 4])
    cov = np.zeros((n, 2, 2)); cov[:,0,0]=s1**2; cov[:,1,1]=s2**2; cov[:,0,1]=rho*s1*s2; cov[:,1,0]=cov[:,0,1]
    L = np.linalg.cholesky(cov + 1e-6*np.eye(2)); z = rng.standard_normal((n, 4, 2))
    return (m[:, None, :] + np.einsum('nij,nkj->nki', L, z)).reshape(n, 8)
def slcp_sample(n, rng):
    th = rng.uniform(-3, 3, (n, 5)); return torch.tensor(th, dtype=torch.float32), torch.tensor(slcp_x(th, rng), dtype=torch.float32)
def slcp_xgiven(t0, n, rng):
    return torch.tensor(slcp_x(np.tile(np.array(t0, float), (n, 1)), rng), dtype=torch.float32)
slcp_grid = [(0.5,-0.5,1.2,1.0,0.3), (-1.0,1.0,1.5,0.8,-0.4), (1.5,1.5,0.8,1.2,0.5), (0.,0.,1.0,1.0,0.), (-1.5,-1.0,1.3,0.9,-0.2)]
run("SLCP", slcp_sample, slcp_xgiven, 5, 8, slcp_grid)
print("  [MAF baseline: asymptotic 0.115 / calibrated 0.426]\n", flush=True)

from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
sim = NormalBivariateUnknownCov()
run("(μ,Σ) d=5", lambda n, r: sim.sample(n, r), lambda t0, n, r: sim.sample_x_given_theta(t0, n, r), 5, 20,
    [(0.,0.,0.,0.,0.), (0.6,-0.5,-0.9,2.0,-2.0), (-0.6,0.5,1.0,-2.0,2.5)])
print("  [MAF baseline: asymptotic 0.060 / calibrated 0.038; oracle 0.026]", flush=True)
