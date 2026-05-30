"""Hypothesis: A22 isn't ROUTED into S because plain RealNVP has fixed masks + no
learned channel mixing. Glow's fix = invertible 1×1 linear (LU) layers between
couplings. Add LinearLU(20) before each affine coupling; test if A22 (the missing
5th sufficient dim) now routes into S. Winner config: 10 layers, hidden128, fresh."""
import math
import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import lstsq
import numpy.linalg as la
from scipy.stats import spearmanr, kstest, norm, chi2

from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
from cdsbi.flows.affine_coupling import _tanh_mlp

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class LinearLU(nn.Module):
    """Invertible linear via PLU: W=P·L·U, L unit-lower, U upper. log|det|=Σlog|diag U|."""
    def __init__(self, d):
        super().__init__()
        q, _ = la.qr(np.random.default_rng(0).standard_normal((d, d)))
        P, L, U = _plu(q)
        self.register_buffer("P", torch.tensor(P, dtype=torch.float32))
        self.L = nn.Parameter(torch.tensor(L, dtype=torch.float32))
        self.U = nn.Parameter(torch.tensor(U, dtype=torch.float32))
        self.register_buffer("l_mask", torch.tril(torch.ones(d, d), -1))
        self.register_buffer("u_mask", torch.triu(torch.ones(d, d), 1))
        self.register_buffer("eye", torch.eye(d))
        s = np.diag(U).copy()
        self.log_s = nn.Parameter(torch.tensor(np.log(np.abs(s)), dtype=torch.float32))
        self.register_buffer("sign_s", torch.tensor(np.sign(s), dtype=torch.float32))

    def forward(self, x):
        L = self.L * self.l_mask + self.eye
        U = self.U * self.u_mask + torch.diag(self.sign_s * torch.exp(self.log_s))
        W = self.P @ L @ U
        return x @ W.T, self.log_s.sum().expand(x.shape[0])


def _plu(W):
    import scipy.linalg as sla
    P, L, U = sla.lu(W)
    return P, L, U


class GlowCoupling(nn.Module):
    def __init__(self, d, hidden=128, n_layers=10, depth=2, cap=2.0):
        super().__init__()
        self.d = d; self.cap = cap
        masks = []
        for i in range(n_layers):
            m = torch.zeros(d); m[i % 2::2] = 1.0; masks.append(m)
        self.register_buffer("_masks", torch.stack(masks))
        self.lin = nn.ModuleList([LinearLU(d) for _ in range(n_layers)])
        self.snet = nn.ModuleList([_tanh_mlp(d, hidden, d, depth) for _ in range(n_layers)])
        self.tnet = nn.ModuleList([_tanh_mlp(d, hidden, d, depth) for _ in range(n_layers)])
        for net in list(self.snet) + list(self.tnet):
            nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)

    def forward(self, x):
        z = x; ld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for i in range(len(self.snet)):
            z, d1 = self.lin[i](z); ld = ld + d1
            m = self._masks[i]; zm = z * m
            s = self.cap * torch.tanh(self.snet[i](zm)) * (1 - m)
            t = self.tnet[i](zm) * (1 - m)
            z = zm + (1 - m) * (z * torch.exp(s) + t)
            ld = ld + s.sum(-1)
        return z, ld
    def n_params(self): return sum(p.numel() for p in self.parameters())


torch.manual_seed(0)
sim = NormalBivariateUnknownCov()
bij = GlowCoupling(20, hidden=128, n_layers=10).to(DEV)
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
    return np.stack([(xc[:, :, 0]**2).sum(1), (xc[:, :, 1]**2).sum(1),
                     (xc[:, :, 0]*xc[:, :, 1]).sum(1), obs.mean(1)[:, 0], obs.mean(1)[:, 1]], 1)


rng = np.random.default_rng(123); _, x = sim.sample(6000, rng)
with torch.no_grad():
    S = bij(x.to(DEV))[0][:, :5].cpu().numpy()
raw = raw_moments(x); oc = sim.oracle_summary(x).cpu().numpy()
print(f"GLOW loss={last:.3f} floor={sim.data_entropy_lower_bound():.3f}")
Ss = (S - S.mean(0)) / (S.std(0) + 1e-9)
print("vs RAW 2nd moments: max|Spearman|  R²")
for k, nm in enumerate(["A11", "A22", "A12", "xbar1", "xbar2"]):
    sp = max(abs(spearmanr(raw[:, k], S[:, j]).statistic) for j in range(5))
    y = (raw[:, k] - raw[:, k].mean()) / (raw[:, k].std() + 1e-9)
    coef, *_ = lstsq(np.c_[Ss, np.ones(len(Ss))], y, rcond=None)
    r2 = 1 - ((y - np.c_[Ss, np.ones(len(Ss))] @ coef)**2).mean() / y.var()
    print(f"  {nm:6s} {sp:.3f}  {r2:.3f}")
Xc = (S - S.mean(0)); Yc = (oc - oc.mean(0)); Xc /= Xc.std(0)+1e-9; Yc /= Yc.std(0)+1e-9
Sxx = Xc.T@Xc/len(Xc); Syy = Yc.T@Yc/len(Yc); Sxy = Xc.T@Yc/len(Xc)
M = la.inv(Sxx+1e-6*np.eye(5))@Sxy@la.inv(Syy+1e-6*np.eye(5))@Sxy.T
cc = np.sqrt(np.clip(np.sort(np.real(la.eigvals(M)))[::-1], 0, 1))
print(f"canonical corr S↔oracle5: {np.round(cc,3)}")
grid = [(0., 0., 0., 0., 0.), (0.6, -0.5, -0.9, 2.0, -2.0), (-0.6, 0.5, 1.0, -2.0, 2.5)]
ks = np.zeros((3, 5))
for i, th0 in enumerate(grid):
    xv = sim.sample_x_given_theta(th0, 4000, np.random.default_rng(7))
    thv = torch.tensor([th0], dtype=xv.dtype).expand(4000, 5)
    with torch.no_grad():
        r = flow.forward(thv.to(DEV), context=bij(xv.to(DEV))[0][:, :5])[0].cpu().numpy()
    ks[i] = [kstest(norm.cdf(r[:, k]), "uniform").statistic for k in range(5)]
print("per-coord PIT KS:\n", ks.round(3))
