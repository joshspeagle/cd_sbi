"""Neural-copula (NLE+score) on the PREVIOUS test cases — comparison points vs the
known CD-SBI/oracle coverage. Same recipe: conditional MAF q(X|θ) + score CD
(asymptotic χ²_{d_θ}) + grid-free calibrated score-norm. asinh data-stabilization on
(θ-indep → score unchanged). Rough, single seed."""
import numpy as np, torch
from scipy.stats import chi2
from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.methods.lf2i import MultiQuantileMLP, train_multi_quantile_head

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALPHAS = [0.5, 0.68, 0.9, 0.95]
QCFG = {"lr":1e-3,"n_steps":4000,"batch_size":512,"optimizer":"adam","lr_schedule":"constant",
        "grad_clip_norm":5.0,"warmup_steps":0,"lr_min_ratio":0.0,"lr_gamma":0.999,
        "weight_decay":0.0,"betas":[0.9,0.999],"momentum":0.9}


def run(name, sim, known, n_steps=10000, asinh=True):
    d_theta = int(sim.d_theta); d_x = int(sim.d_x)
    nl = 6 if d_x <= 4 else 8
    torch.manual_seed(0)
    flow = MAFAdapter(features=max(d_x, 2), context_features=d_theta, hidden=64, num_layers=nl).to(DEV)
    pad = max(d_x, 2) - d_x  # MAF needs ≥2 features; pad 1-d data with a constant-0 channel
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3); rng = np.random.default_rng(0)

    def prep(x):
        if asinh: x = torch.asinh(x)
        if pad: x = torch.cat([x, torch.zeros(x.shape[0], pad)], -1)
        return x

    flow.train()
    for _ in range(n_steps):
        th, x = sim.sample(256, rng)
        loss = -flow.log_prob(prep(x).to(DEV), context=th.to(DEV)).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
    flow.eval()
    print(f"\n===== {name} (d_θ={d_theta}, d_x={d_x}) — NLE -logq={loss.item():.2f}  [CD-SBI≈{known}] =====")

    def score(theta_rows, x):
        th = theta_rows.clone().detach().to(DEV).requires_grad_(True)
        lp = flow.log_prob(prep(x).to(DEV), context=th)
        g, = torch.autograd.grad(lp.sum(), th); return g.detach()

    def fisher(th0, n=8000):
        xf = sim.sample_x_given_theta(th0, n, np.random.default_rng(999))
        thr = torch.tensor([th0], dtype=torch.float32).expand(n, d_theta)
        U = score(thr, xf); return (U.T @ U / n).cpu().numpy()

    grid = [tuple(map(float, t)) for t in sim.sample(5, np.random.default_rng(321))[0].numpy()]
    wa = 0.0
    for th0 in grid:
        Iinv = np.linalg.inv(fisher(th0) + 1e-5*np.eye(d_theta))
        xv = sim.sample_x_given_theta(th0, 3000, np.random.default_rng(7))
        thr = torch.tensor([th0], dtype=torch.float32).expand(3000, d_theta)
        U = score(thr, xv).cpu().numpy(); stat = np.einsum('ni,ij,nj->n', U, Iinv, U)
        for a in ALPHAS: wa = max(wa, abs((stat <= chi2.ppf(a, df=d_theta)).mean() - a))
    th_cal, x_cal = sim.sample(40000, np.random.default_rng(11))
    stat_cal = (score(th_cal, x_cal)**2).sum(1)
    qnet = MultiQuantileMLP(input_dim=d_theta, hidden=128, depth=2, n_quantiles=len(ALPHAS)).to(DEV)
    train_multi_quantile_head(qnet, th_cal.to(DEV), stat_cal.to(DEV), ALPHAS, QCFG, DEV); qnet.eval()
    wc = 0.0
    for th0 in grid:
        xv = sim.sample_x_given_theta(th0, 3000, np.random.default_rng(7))
        thr = torch.tensor([th0], dtype=torch.float32).expand(3000, d_theta)
        stat = (score(thr, xv)**2).sum(1).cpu().numpy()
        with torch.no_grad():
            cv = qnet(torch.tensor([th0], dtype=torch.float32, device=DEV))[0].cpu().numpy()
        for k, a in enumerate(ALPHAS): wc = max(wc, abs((stat <= cv[k]).mean() - a))
    print(f"  asymptotic-score={wa:.3f}   calibrated-score={wc:.3f}   (vs CD-SBI {known})")


from cdsbi.simulators.location_normal_1d import LocationNormal1D
from cdsbi.simulators.location_gauss_2d_iid import LocationGaussian2D_iid
from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
from cdsbi.simulators.exp_rate import ExponentialRate
from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar

run("§8.1 LocationNormal1D", LocationNormal1D(), "0.025")
run("§8.2 LocationGaussian2D_iid", LocationGaussian2D_iid(), "0.025")
run("§8.3 LocationGaussian2D_corr", LocationGaussian2D_corr(), "0.025")
run("§8.4 ExponentialRate", ExponentialRate(), "0.028-0.034")
run("(μ,σ²) NormalUnknownMeanVar", NormalUnknownMeanVar(), "0.026")
print("\n[(μ,Σ) d=5 already: score 0.060 / calibrated 0.038, oracle 0.026]")
