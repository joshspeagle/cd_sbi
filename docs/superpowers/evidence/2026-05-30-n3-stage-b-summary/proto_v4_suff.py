"""Decisive: is log_D22's low Spearman a SUFFICIENCY failure or a PARAMETERIZATION
artifact? Train the winner (big plain affine, fresh-batch) and measure recovery of
the covariance info three honest ways:
 (1) per-coord max|Spearman| vs CHOLESKY oracle (logD11,logD22,D21)  [current metric]
 (2) per-coord max|Spearman| + multiple-regression R²(S→·) vs RAW 2nd moments
     A11=Σ(x1-x̄1)², A22=Σ(x2-x̄2)², A12=Σ(x1-x̄1)(x2-x̄2)  (the actual sufficient stat)
 (3) canonical correlations between S(5) and the oracle-5  (joint info overlap)
 (4) per-coord PIT (center+2 extremes) + joint χ²₅  (does it calibrate?)
If RAW moments + R² + canonical corr are high, S IS sufficient and log_D22 Spearman
is the artifact (the N2 RMSE-vs-r* lesson) — milestone effectively met by big plain affine."""
import math
import numpy as np
import torch
from numpy.linalg import lstsq
from scipy.stats import spearmanr, kstest, norm, chi2

from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
from cdsbi.flows.affine_coupling import AffineCouplingBijection

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
sim = NormalBivariateUnknownCov()
bij = AffineCouplingBijection(20, hidden=128, n_layers=10).to(DEV)
flow = SingleIndexMonotoneFlow(d=5, theta_signs=list(sim.theta_signs),
                               feat_signs=list(sim.feat_signs), hidden=64, depth=2).to(DEV)
params = list(flow.parameters()) + list(bij.parameters())
opt = torch.optim.Adam(params, lr=2e-3)
rng = np.random.default_rng(0)
const = 0.5 * sim.d_x * math.log(2 * math.pi)
flow.train(); bij.train(); last = 0.0
for _ in range(25000):
    th, x = sim.sample(256, rng); th = th.to(DEV); x = x.to(DEV)
    z, bld = bij(x); S = z[:, :5]; A = z[:, 5:]
    r, pld = flow.forward(th, context=S)
    loss = (0.5 * (r.pow(2).sum(-1) + A.pow(2).sum(-1)) + const - pld - bld).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step(); last = loss.item()
flow.eval(); bij.eval()


def raw_moments(x):
    obs = x.view(x.shape[0], sim.n_iid, 2).cpu().numpy()
    xc = obs - obs.mean(1, keepdims=True)
    A11 = (xc[:, :, 0] ** 2).sum(1); A22 = (xc[:, :, 1] ** 2).sum(1)
    A12 = (xc[:, :, 0] * xc[:, :, 1]).sum(1)
    xbar = obs.mean(1)
    return np.stack([A11, A22, A12, xbar[:, 0], xbar[:, 1]], 1)


rng = np.random.default_rng(123); _, x = sim.sample(6000, rng)
with torch.no_grad():
    S = bij(x.to(DEV))[0][:, :5].cpu().numpy()
oracle_chol = sim.oracle_summary(x).cpu().numpy()
raw = raw_moments(x)
print(f"loss={last:.3f} floor={sim.data_entropy_lower_bound():.3f}\n")

CHOL = ["log_D11", "log_D22", "D21", "xbar1", "xbar2"]
RAW = ["A11", "A22", "A12", "xbar1", "xbar2"]
Ss = (S - S.mean(0)) / (S.std(0) + 1e-9)
print("(1) vs CHOLESKY:   max|Spearman|")
for k, nm in enumerate(CHOL):
    print(f"    {nm:8s} {max(abs(spearmanr(oracle_chol[:,k], S[:,j]).statistic) for j in range(5)):.3f}")
print("(2) vs RAW 2nd moments:  max|Spearman|   R²(all 5 S → coord)")
for k, nm in enumerate(RAW):
    sp = max(abs(spearmanr(raw[:, k], S[:, j]).statistic) for j in range(5))
    y = (raw[:, k] - raw[:, k].mean()) / (raw[:, k].std() + 1e-9)
    coef, *_ = lstsq(np.c_[Ss, np.ones(len(Ss))], y, rcond=None)
    r2 = 1 - ((y - np.c_[Ss, np.ones(len(Ss))] @ coef) ** 2).mean() / y.var()
    print(f"    {nm:8s} {sp:.3f}        {r2:.3f}")
# (3) canonical correlations S(5) vs oracle-chol(5)
Xc = (S - S.mean(0)); Yc = (oracle_chol - oracle_chol.mean(0))
Xc /= (Xc.std(0) + 1e-9); Yc /= (Yc.std(0) + 1e-9)
Sxx = Xc.T @ Xc / len(Xc); Syy = Yc.T @ Yc / len(Yc); Sxy = Xc.T @ Yc / len(Xc)
import numpy.linalg as la
M = la.inv(Sxx + 1e-6 * np.eye(5)) @ Sxy @ la.inv(Syy + 1e-6 * np.eye(5)) @ Sxy.T
cc = np.sqrt(np.clip(np.sort(np.real(la.eigvals(M)))[::-1], 0, 1))
print(f"(3) canonical correlations S↔oracle5: {np.round(cc,3)}")
# (4) calibration
grid = [(0., 0., 0., 0., 0.), (0.6, -0.5, -0.9, 2.0, -2.0), (-0.6, 0.5, 1.0, -2.0, 2.5)]
ks = np.zeros((3, 5))
for i, th0 in enumerate(grid):
    xv = sim.sample_x_given_theta(th0, 4000, np.random.default_rng(7))
    thv = torch.tensor([th0], dtype=xv.dtype).expand(4000, 5)
    with torch.no_grad():
        S2 = bij(xv.to(DEV))[0][:, :5]
        r = flow.forward(thv.to(DEV), context=S2)[0].cpu().numpy()
    ks[i] = [kstest(norm.cdf(r[:, k]), "uniform").statistic for k in range(5)]
print("(4) per-coord PIT KS (rows center/extreme/extreme; r1..r5):\n", ks.round(3))
xv = sim.sample_x_given_theta((0., 0., 0., 0., 0.), 3000, np.random.default_rng(7))
with torch.no_grad():
    S2 = bij(xv.to(DEV))[0][:, :5]
    rr = flow.forward(torch.zeros(3000, 5, device=DEV), context=S2)[0].cpu().numpy()
print(f"    joint ‖r‖²~χ²₅ KS = {kstest(chi2.cdf((rr**2).sum(1), df=5), 'uniform').statistic:.3f}")
