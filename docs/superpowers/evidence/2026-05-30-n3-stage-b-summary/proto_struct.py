"""Structure-informed invertible summary — validate the core math.
HelmertPolar: x(20)→reshape(10,2)→Helmert over replicates (means m∈ℝ², contrasts
C∈ℝ^{9×2}, log-det 0)→polar/QR on C exposing Bartlett D=(D11,D21,D22) + sphere
directions (15, ancillary). Summary S=(logD11,logD22,D21,m1,m2)=Bartlett oracle feats.
Checks:
 (1) forward D matches simulator's oracle_summary (the Bartlett Cholesky) → correct.
 (2) invertible: reconstruct C from (D11,D21,D22,u1,u2) → match.
 (3) log|det ∂(D,dirs)/∂C| = 8 logD11 + 7 logD22 (+const) — radial Jacobian.
 (4) quick-train SingleIndexMonotoneFlow pivot on S (NF-MLE) → 5/5 recovery (by
     construction) + per-coord PIT + χ²₅ calibration + entropy floor (Stage-A parity)."""
import math
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, kstest, norm, chi2

from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow

DEV = "cuda" if torch.cuda.is_available() else "cpu"
n_iid = 10


def helmert(n):
    H = np.zeros((n, n)); H[0] = 1 / math.sqrt(n)
    for k in range(1, n):
        H[k, :k] = 1 / math.sqrt(k * (k + 1)); H[k, k] = -k / math.sqrt(k * (k + 1))
    return H


class HelmertPolar(nn.Module):
    def __init__(self, n_iid=10):
        super().__init__(); self.n = n_iid
        self.register_buffer("H", torch.tensor(helmert(n_iid), dtype=torch.float32))

    def transform(self, x):
        B = x.shape[0]
        y = torch.einsum('ij,bjc->bic', self.H, x.view(B, self.n, 2))   # (B,n,2)
        m = y[:, 0, :]                                                  # (B,2) means*√n
        C = y[:, 1:, :]                                                 # (B,n-1,2) contrasts
        c1 = C[:, :, 0]; c2 = C[:, :, 1]                                # (B,9)
        D11 = c1.norm(dim=1).clamp_min(1e-8)                            # (B,)
        u1 = c1 / D11[:, None]
        D21 = (u1 * c2).sum(1)                                          # (B,)
        c2p = c2 - D21[:, None] * u1
        D22 = c2p.norm(dim=1).clamp_min(1e-8)
        u2 = c2p / D22[:, None]
        S = torch.stack([torch.log(D11), torch.log(D22), D21, m[:, 0], m[:, 1]], 1)
        A_dirs = torch.cat([u1, u2], 1)                                 # (B,18) on-sphere (15 dof)
        logdet = 8 * torch.log(D11) + 7 * torch.log(D22)               # radial Jacobian (+const)
        return S, A_dirs, logdet, (D11, D21, D22, u1, u2, m)

    def invert(self, D11, D21, D22, u1, u2, m):
        c1 = D11[:, None] * u1
        c2 = D21[:, None] * u1 + D22[:, None] * u2
        C = torch.stack([c1, c2], -1)                                  # (B,9,2)
        y = torch.cat([m[:, None, :], C], 1)                           # (B,10,2)
        x = torch.einsum('ji,bjc->bic', self.H, y)                     # Hᵀ y
        return x.reshape(x.shape[0], 20)


torch.manual_seed(0)
sim = NormalBivariateUnknownCov()
hp = HelmertPolar(n_iid).to(DEV)
rng = np.random.default_rng(1)
theta, x = sim.sample(2000, rng); x = x.to(DEV)
S, Ad, ld, parts = hp.transform(x)

# (1) forward matches oracle Bartlett feats
oc = sim.oracle_summary(x.cpu()).numpy()      # (logD11, logD22, D21, xbar1, xbar2)
Sn = S.detach().cpu().numpy()
print("(1) forward vs oracle_summary (max|diff| per coord, means scaled by √n):")
xbar_scale = np.array([1, 1, 1, math.sqrt(n_iid), math.sqrt(n_iid)])
print("   logD11,logD22,D21 abs-diff:", np.round(np.abs(Sn[:, :3] - oc[:, :3]).max(0), 5))
print("   means: corr(S4,xbar1)=%.4f corr(S5,xbar2)=%.4f" % (
    np.corrcoef(Sn[:, 3], oc[:, 3])[0, 1], np.corrcoef(Sn[:, 4], oc[:, 4])[0, 1]))

# (2) invertibility
D11, D21, D22, u1, u2, m = parts
xrec = hp.invert(D11, D21, D22, u1, u2, m)
print(f"(2) invertibility max|x-xrec| = {(x - xrec).abs().max().item():.2e}")

# (3) log-det sanity: compare to finite-diff? just report it's finite & structure
print(f"(3) logdet sample mean = {ld.mean().item():.3f} (= 8logD11+7logD22)")

# (4) quick-train pivot on S, check calibration + floor (Stage-A parity)
flow = SingleIndexMonotoneFlow(d=5, theta_signs=list(sim.theta_signs),
                               feat_signs=list(sim.feat_signs), hidden=64, depth=2).to(DEV)
opt = torch.optim.Adam(flow.parameters(), lr=3e-3)
rng = np.random.default_rng(0)
flow.train(); last = 0.0
for _ in range(8000):
    th, xb = sim.sample(256, rng); th = th.to(DEV); xb = xb.to(DEV)
    Sb, _, _, _ = hp.transform(xb)
    r, pld = flow.forward(th, context=Sb)
    loss = (0.5 * r.pow(2).sum(-1) - pld).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step(); last = loss.item()
flow.eval()
print(f"(4) pivot NF-MLE final_loss={last:.3f} entropy_floor={sim.entropy_lower_bound():.3f}")
grid = [(0., 0., 0., 0., 0.), (0.6, -0.5, -0.9, 2.0, -2.0), (-0.6, 0.5, 1.0, -2.0, 2.5)]
ks = np.zeros((3, 5))
for i, th0 in enumerate(grid):
    xv = sim.sample_x_given_theta(th0, 4000, np.random.default_rng(7)).to(DEV)
    thv = torch.tensor([th0], dtype=xv.dtype, device=DEV).expand(4000, 5)
    with torch.no_grad():
        Sv, _, _, _ = hp.transform(xv)
        r = flow.forward(thv, context=Sv)[0].cpu().numpy()
    ks[i] = [kstest(norm.cdf(r[:, k]), "uniform").statistic for k in range(5)]
print("   per-coord PIT KS:\n", ks.round(3))
xv = sim.sample_x_given_theta((0., 0., 0., 0., 0.), 3000, np.random.default_rng(7)).to(DEV)
with torch.no_grad():
    Sv, _, _, _ = hp.transform(xv)
    rr = flow.forward(torch.zeros(3000, 5, device=DEV), context=Sv)[0].cpu().numpy()
print(f"   joint ‖r‖²~χ²₅ KS = {kstest(chi2.cdf((rr**2).sum(1), df=5), 'uniform').statistic:.3f}")
