"""N3 perm-equivariant risk-gate. Build a minimal EQUIVARIANT coupling bijection on
the set of 10 exchangeable bivariate replicates, then compose with the existing
affine coupling for routing. Question: does it recover log_D22 & D21 (which the
plain affine coupling could NOT)?

Equivariant layer (mask one channel m; alternate per layer):
  a_i = x_i[m] (masked, kept fixed);  b_i = x_i[~m] (transformed)
  c   = mean_i φ(a_i)            # permutation-INVARIANT pooled context (∈ ℝ^h)
  b_i'= b_i·exp(s(a_i,c)) + t(a_i,c)
  log_det = Σ_i s_i             # triangular (b' depends on b diag + fixed a,c)
The pooled c can carry Σa, Σa² → second moments; exp(s)·b forms cross-products.
Then flatten (B,10,2)→(B,20) and route through AffineCouplingBijection(20).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, kstest, norm, chi2

from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
from cdsbi.flows.affine_coupling import AffineCouplingBijection
from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
from cdsbi.confidence_set.procedures import PivotBasedProcedure

ORACLE = ["log_D11", "log_D22", "D21", "xbar1", "xbar2"]


def mlp(i, h, o, depth=2):
    L = [nn.Linear(i, h), nn.Tanh()]
    for _ in range(depth - 1):
        L += [nn.Linear(h, h), nn.Tanh()]
    L += [nn.Linear(h, o)]
    return nn.Sequential(*L)


class EquivariantCoupling(nn.Module):
    """One equivariant coupling layer over a set of replicates, mask channel `mc`."""
    def __init__(self, p=2, ctx=8, hidden=32, scale_cap=2.0, mc=0):
        super().__init__()
        self.mc = mc; self.p = p; self.scale_cap = scale_cap
        self.phi = mlp(1, hidden, ctx)
        self.s = mlp(1 + ctx, hidden, 1)
        self.t = mlp(1 + ctx, hidden, 1)
        nn.init.zeros_(self.s[-1].weight); nn.init.zeros_(self.s[-1].bias)
        nn.init.zeros_(self.t[-1].weight); nn.init.zeros_(self.t[-1].bias)

    def forward(self, x):           # x: (B, n, p)
        B, n, p = x.shape
        a = x[:, :, self.mc:self.mc + 1]                         # (B,n,1) masked
        uc = 1 - self.mc                                          # the other channel
        b = x[:, :, uc:uc + 1]                                    # (B,n,1)
        c = self.phi(a).mean(dim=1, keepdim=True).expand(B, n, -1)  # (B,n,ctx) invariant
        inp = torch.cat([a, c], dim=-1)
        s = self.scale_cap * torch.tanh(self.s(inp))
        t = self.t(inp)
        b2 = b * torch.exp(s) + t
        out = x.clone()
        out[:, :, uc:uc + 1] = b2
        return out, s.sum(dim=(1, 2))                             # log_det (B,)


class EquivariantSummary(nn.Module):
    """K equivariant layers (alt mask) ∘ AffineCoupling(20) router. transform/encode
    match InvertibleSummaryConditioner's contract."""
    def __init__(self, n_iid=10, p=2, d_theta=5, k_equiv=8, ctx=8, hidden=64):
        super().__init__()
        self.n_iid = n_iid; self.p = p; self.d_theta = d_theta
        self.eq = nn.ModuleList([EquivariantCoupling(p=p, ctx=ctx, hidden=hidden, mc=i % 2)
                                 for i in range(k_equiv)])
        self.router = AffineCouplingBijection(n_iid * p, hidden=hidden, n_layers=6, depth=2)

    def transform(self, x):         # x: (B, 20)
        B = x.shape[0]
        z = x.view(B, self.n_iid, self.p)
        ld = torch.zeros(B, device=x.device, dtype=x.dtype)
        for layer in self.eq:
            z, d = layer(z); ld = ld + d
        z = z.reshape(B, self.n_iid * self.p)
        z, d2 = self.router(z)
        return z, ld + d2

    def encode(self, x):
        z, ld = self.transform(x)
        return z[:, :self.d_theta], ld

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def train(cond, sim, n_steps=15000, lr=2e-3, fresh=True):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    flow = SingleIndexMonotoneFlow(d=5, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=64, depth=2).to(dev)
    cond = cond.to(dev)
    import math
    params = list(flow.parameters()) + list(cond.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(0)
    const = 0.5 * sim.d_x * math.log(2 * math.pi)
    if not fresh:
        th_all, x_all = sim.sample(20000, rng); th_all = th_all.to(dev); x_all = x_all.to(dev)
    flow.train(); cond.train()
    last = 0.0
    for step in range(n_steps):
        if fresh:
            th, x = sim.sample(256, rng); th = th.to(dev); x = x.to(dev)
        else:
            idx = torch.randint(0, 20000, (256,)); th = th_all[idx]; x = x_all[idx]
        z, bij_ld = cond.transform(x)
        S = z[:, :5]; A = z[:, 5:]
        r, piv_ld = flow.forward(th, context=S)
        nll = (0.5 * (r.pow(2).sum(-1) + A.pow(2).sum(-1)) + const - piv_ld - bij_ld)
        loss = nll.mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        last = loss.item()
    flow.eval(); cond.eval()
    return flow, cond, last, dev


torch.manual_seed(0)
sim = NormalBivariateUnknownCov()
cond = EquivariantSummary(n_iid=10, p=2, d_theta=5, k_equiv=8, ctx=8, hidden=64)
flow, cond, last, dev = train(cond, sim, n_steps=15000, fresh=True)

rng = np.random.default_rng(123)
_, x = sim.sample(4000, rng)
with torch.no_grad():
    S = cond.encode(x.to(dev))[0].cpu().numpy()
    oracle = sim.oracle_summary(x).cpu().numpy()
print("=== EQUIVARIANT sufficiency (fresh-batch) ===")
mn = 1.0
for k, nm in enumerate(ORACLE):
    sps = [abs(spearmanr(oracle[:, k], S[:, j]).statistic) for j in range(5)]
    v = float(np.nanmax(sps)); mn = min(mn, v)
    print(f"  {nm:8s} {v:.3f} (S[{int(np.nanargmax(sps))}])")
print(f"  min over 5 = {mn:.3f}  {'PASS' if mn > 0.9 else 'FAIL'} (>0.9)")
print(f"final_loss={last:.3f}  floor={sim.data_entropy_lower_bound():.3f}  "
      f"{'no-cheat' if last > sim.data_entropy_lower_bound() else 'CHEAT'}")


def pivot_fn(th, xx):
    S = cond.encode(xx.to(dev))[0]
    return flow.forward(th.to(dev), context=S)[0]


grid = [(0., 0., 0., 0., 0.), (0.6, -0.5, -0.9, 2.0, -2.0), (-0.6, 0.5, 1.0, -2.0, 2.5)]
ks = np.zeros((3, 5))
for i, th0 in enumerate(grid):
    xv = sim.sample_x_given_theta(th0, 4000, np.random.default_rng(7))
    thv = torch.tensor([th0], dtype=xv.dtype).expand(4000, 5)
    with torch.no_grad():
        r = pivot_fn(thv, xv).cpu().numpy()
    ks[i] = [kstest(norm.cdf(r[:, k]), "uniform").statistic for k in range(5)]
print("=== per-coord PIT KS (rows center/extreme/extreme; cols r1..r5) ===")
print(ks.round(3))
xv = sim.sample_x_given_theta((0., 0., 0., 0., 0.), 3000, np.random.default_rng(7))
with torch.no_grad():
    rr = pivot_fn(torch.zeros(3000, 5), xv).cpu().numpy()
print(f"joint ‖r‖²~χ²₅ KS = {kstest(chi2.cdf((rr**2).sum(1), df=5), 'uniform').statistic:.3f}")
