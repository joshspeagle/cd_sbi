"""N3 asinh risk-gate (user's steer: asinh transform for ALL covariance components —
variances AND signed off-diagonals). asinh-affine coupling = affine in asinh-space:
   b' = sinh(a(u)·asinh(b) + d(u)),  a>0
sign-preserving, log-tails + linear-near-0, invertible, tractable log-det:
   log|db'/db| = log a + logcosh(a·asinh(b)+d) − ½ log(b²+1).
Variants vs plain-affine baseline (3/5; log_D22 0.53, D21 0.65):
  V1  AsinhAffineCoupling(20)  — transform expressivity alone (no equivariance)
  V2  Equivariant-asinh couplings + asinh router — full perm-equiv + asinh
"""
import math
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, kstest, norm, chi2

from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow

ORACLE = ["log_D11", "log_D22", "D21", "xbar1", "xbar2"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def mlp(i, h, o, depth=2):
    L = [nn.Linear(i, h), nn.Tanh()]
    for _ in range(depth - 1):
        L += [nn.Linear(h, h), nn.Tanh()]
    L += [nn.Linear(h, o)]
    nn.init.zeros_(L[-1].weight); nn.init.zeros_(L[-1].bias)
    return nn.Sequential(*L)


def logcosh(x):
    ax = x.abs()
    return ax + torch.log1p(torch.exp(-2 * ax)) - math.log(2.0)


def asinh_affine(b, log_a, d):
    """b'=sinh(a·asinh(b)+d); returns b', per-elem log|db'/db|. a=exp(log_a)."""
    ab = torch.asinh(b)
    arg = torch.exp(log_a) * ab + d
    arg = arg.clamp(-15, 15)
    bp = torch.sinh(arg)
    ld = log_a + logcosh(arg) - 0.5 * torch.log(b * b + 1)
    return bp, ld


class AsinhAffineCoupling(nn.Module):
    def __init__(self, d, hidden=64, n_layers=6, cap=1.5):
        super().__init__()
        self.d = d; self.cap = cap
        masks = []
        for i in range(n_layers):
            m = torch.zeros(d); m[i % 2::2] = 1.0; masks.append(m)
        self.register_buffer("_masks", torch.stack(masks))
        self.anet = nn.ModuleList([mlp(d, hidden, d) for _ in range(n_layers)])
        self.dnet = nn.ModuleList([mlp(d, hidden, d) for _ in range(n_layers)])

    def forward(self, x):
        z = x; ld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for i in range(len(self.anet)):
            m = self._masks[i]; zm = z * m
            log_a = self.cap * torch.tanh(self.anet[i](zm)) * (1 - m)
            d = self.dnet[i](zm) * (1 - m)
            bp, eld = asinh_affine(z, log_a, d)
            z = zm + (1 - m) * bp
            ld = ld + ((1 - m) * eld).sum(-1)
        return z, ld

    def n_params(self): return sum(p.numel() for p in self.parameters())


class EquivAsinh(nn.Module):
    """Equivariant asinh coupling over replicates; mask channel mc."""
    def __init__(self, ctx=8, hidden=32, cap=1.5, mc=0):
        super().__init__()
        self.mc = mc; self.cap = cap
        self.phi = mlp(1, hidden, ctx)
        self.anet = mlp(1 + ctx, hidden, 1)
        self.dnet = mlp(1 + ctx, hidden, 1)

    def forward(self, x):                                 # x:(B,n,2)
        B, n, p = x.shape
        a = x[:, :, self.mc:self.mc + 1]; uc = 1 - self.mc
        b = x[:, :, uc:uc + 1]
        c = self.phi(a).mean(1, keepdim=True).expand(B, n, -1)
        inp = torch.cat([a, c], -1)
        log_a = self.cap * torch.tanh(self.anet(inp))
        d = self.dnet(inp)
        bp, eld = asinh_affine(b, log_a, d)
        out = x.clone(); out[:, :, uc:uc + 1] = bp
        return out, eld.sum((1, 2))


class V1Cond(nn.Module):
    def __init__(self, d_x=20, d_theta=5, hidden=64):
        super().__init__(); self.d_theta = d_theta
        self.bij = AsinhAffineCoupling(d_x, hidden=hidden, n_layers=6)
    def transform(self, x): return self.bij(x)
    def encode(self, x):
        z, ld = self.bij(x); return z[:, :self.d_theta], ld
    def n_params(self): return self.bij.n_params()


class V2Cond(nn.Module):
    def __init__(self, n_iid=10, p=2, d_theta=5, k=8, ctx=8, hidden=64):
        super().__init__(); self.n_iid = n_iid; self.p = p; self.d_theta = d_theta
        self.eq = nn.ModuleList([EquivAsinh(ctx=ctx, hidden=hidden, mc=i % 2) for i in range(k)])
        self.router = AsinhAffineCoupling(n_iid * p, hidden=hidden, n_layers=6)
    def transform(self, x):
        B = x.shape[0]; z = x.view(B, self.n_iid, self.p)
        ld = torch.zeros(B, device=x.device, dtype=x.dtype)
        for L in self.eq:
            z, e = L(z); ld = ld + e
        z, e2 = self.router(z.reshape(B, self.n_iid * self.p))
        return z, ld + e2
    def encode(self, x):
        z, ld = self.transform(x); return z[:, :self.d_theta], ld
    def n_params(self): return sum(p.numel() for p in self.parameters())


def run(tag, cond, sim, n_steps=15000, lr=2e-3):
    flow = SingleIndexMonotoneFlow(d=5, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=64, depth=2).to(DEV)
    cond = cond.to(DEV)
    params = list(flow.parameters()) + list(cond.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(0)
    const = 0.5 * sim.d_x * math.log(2 * math.pi)
    flow.train(); cond.train(); last = 0.0
    for _ in range(n_steps):
        th, x = sim.sample(256, rng); th = th.to(DEV); x = x.to(DEV)
        z, bld = cond.transform(x)
        S = z[:, :5]; A = z[:, 5:]
        r, pld = flow.forward(th, context=S)
        loss = (0.5 * (r.pow(2).sum(-1) + A.pow(2).sum(-1)) + const - pld - bld).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step(); last = loss.item()
    flow.eval(); cond.eval()
    rng = np.random.default_rng(123)
    _, x = sim.sample(4000, rng)
    with torch.no_grad():
        S = cond.encode(x.to(DEV))[0].cpu().numpy()
        oracle = sim.oracle_summary(x).cpu().numpy()
    print(f"===== {tag} =====")
    mn = 1.0
    for k, nm in enumerate(ORACLE):
        sps = [abs(spearmanr(oracle[:, k], S[:, j]).statistic) for j in range(5)]
        v = float(np.nanmax(sps)); mn = min(mn, v)
        print(f"  {nm:8s} {v:.3f} (S[{int(np.nanargmax(sps))}])")
    print(f"  min5={mn:.3f} {'PASS' if mn > 0.9 else 'FAIL'}  loss={last:.3f} floor={sim.data_entropy_lower_bound():.3f}"
          f" {'no-cheat' if last > sim.data_entropy_lower_bound() else 'CHEAT'}")

    def piv(th, xx): return flow.forward(th.to(DEV), context=cond.encode(xx.to(DEV))[0])[0]
    grid = [(0., 0., 0., 0., 0.), (0.6, -0.5, -0.9, 2.0, -2.0), (-0.6, 0.5, 1.0, -2.0, 2.5)]
    ks = np.zeros((3, 5))
    for i, th0 in enumerate(grid):
        xv = sim.sample_x_given_theta(th0, 4000, np.random.default_rng(7))
        thv = torch.tensor([th0], dtype=xv.dtype).expand(4000, 5)
        with torch.no_grad():
            r = piv(thv, xv).cpu().numpy()
        ks[i] = [kstest(norm.cdf(r[:, k]), "uniform").statistic for k in range(5)]
    print("  per-coord PIT KS:\n", ks.round(3))


torch.manual_seed(0)
sim = NormalBivariateUnknownCov()
run("V1 asinh-affine (no equiv)", V1Cond(), sim)
torch.manual_seed(0)
run("V2 equiv-asinh + asinh router", V2Cond(), sim)
