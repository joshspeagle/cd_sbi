"""Fair asinh test + capacity control.
V3  AFFINE transform (proven, stable) but conditioner nets see ASINH features of
    the masked coords: input = [zm, asinh(zm)]. Sign-aware log-linear basis for the
    scale/shift → can build signed cross-products (Σx₁x₂) and log-scales uniformly,
    WITHOUT the unstable sinh output tails. (user's asinh idea, stable realization)
V4  plain affine, BIGGER (10 layers/128h), fresh-batch (no overfit possible) — is
    the covariance failure capacity-bound or structural?
Baseline to beat: plain affine 3/5 (logD11 0.97, logD22 0.53, D21 0.65)."""
import math
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, kstest, norm

from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
from cdsbi.flows.affine_coupling import AffineCouplingBijection

ORACLE = ["log_D11", "log_D22", "D21", "xbar1", "xbar2"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def mlp(i, h, o, depth=2):
    L = [nn.Linear(i, h), nn.Tanh()]
    for _ in range(depth - 1):
        L += [nn.Linear(h, h), nn.Tanh()]
    L += [nn.Linear(h, o)]
    nn.init.zeros_(L[-1].weight); nn.init.zeros_(L[-1].bias)
    return nn.Sequential(*L)


class AsinhFeatAffineCoupling(nn.Module):
    """Affine coupling z·exp(s)+t, but s,t nets see [zm, asinh(zm)] (2d inputs)."""
    def __init__(self, d, hidden=64, n_layers=6, cap=2.0):
        super().__init__()
        self.d = d; self.cap = cap
        masks = []
        for i in range(n_layers):
            m = torch.zeros(d); m[i % 2::2] = 1.0; masks.append(m)
        self.register_buffer("_masks", torch.stack(masks))
        self.snet = nn.ModuleList([mlp(2 * d, hidden, d) for _ in range(n_layers)])
        self.tnet = nn.ModuleList([mlp(2 * d, hidden, d) for _ in range(n_layers)])

    def forward(self, x):
        z = x; ld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for i in range(len(self.snet)):
            m = self._masks[i]; zm = z * m
            feat = torch.cat([zm, torch.asinh(zm)], -1)
            s = self.cap * torch.tanh(self.snet[i](feat)) * (1 - m)
            t = self.tnet[i](feat) * (1 - m)
            z = zm + (1 - m) * (z * torch.exp(s) + t)
            ld = ld + s.sum(-1)
        return z, ld
    def n_params(self): return sum(p.numel() for p in self.parameters())


class Cond(nn.Module):
    def __init__(self, bij, d_theta=5):
        super().__init__(); self.bij = bij; self.d_theta = d_theta
    def transform(self, x): return self.bij(x)
    def encode(self, x):
        z, ld = self.bij(x); return z[:, :self.d_theta], ld
    def n_params(self): return self.bij.n_params()


def run(tag, bij, sim, n_steps=15000, lr=2e-3):
    cond = Cond(bij).to(DEV)
    flow = SingleIndexMonotoneFlow(d=5, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=64, depth=2).to(DEV)
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
    rng = np.random.default_rng(123); _, x = sim.sample(4000, rng)
    with torch.no_grad():
        S = cond.encode(x.to(DEV))[0].cpu().numpy(); oracle = sim.oracle_summary(x).cpu().numpy()
    print(f"===== {tag} =====")
    mn = 1.0
    for k, nm in enumerate(ORACLE):
        sps = [abs(spearmanr(oracle[:, k], S[:, j]).statistic) for j in range(5)]
        v = float(np.nanmax(sps)); mn = min(mn, v)
        print(f"  {nm:8s} {v:.3f}")
    print(f"  min5={mn:.3f} {'PASS' if mn > 0.9 else 'FAIL'}  loss={last:.3f} floor={sim.data_entropy_lower_bound():.3f}")


torch.manual_seed(0)
sim = NormalBivariateUnknownCov()
run("V3 asinh-feature affine coupling", AsinhFeatAffineCoupling(20, hidden=64, n_layers=6), sim)
torch.manual_seed(0)
run("V4 plain affine BIG (10L/128h) fresh", AffineCouplingBijection(20, hidden=128, n_layers=10), sim, n_steps=25000)
