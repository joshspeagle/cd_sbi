"""NEURAL-COPULA probe #1: NLE + score (Rao) pivot — escape the d_θ summary bottleneck.
Model p(X|θ) with a scalable conditional flow (MAF; no bottleneck, info-complete,
cannot cheat — it's a proper likelihood on raw X∈ℝ²⁰). Read a d_θ=5 EFFICIENT pivot
from the SCORE U(θ;X)=∇_θ log q(X|θ): at the true θ, U|θ ~ N(0, I(θ)) (mean-zero
exactly; ≈Gaussian by CLT over the n_iid=10 replicates). Rao stat = Uᵀ I(θ)⁻¹ U ~ χ²₅.
Test on (μ,Σ) d=5: coverage vs oracle (0.026) + per-coord PIT of r = I^{-1/2} U.
Fisher I(θ₀) estimated from a SEPARATE batch (no optimism); coverage on fresh X."""
import numpy as np
import torch
from scipy.stats import chi2, norm, kstest

from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
from cdsbi.flows.maf_adapter import MAFAdapter

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
sim = NormalBivariateUnknownCov()
flow = MAFAdapter(features=sim.d_x, context_features=sim.d_theta, hidden=128, num_layers=8).to(DEV)
opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
rng = np.random.default_rng(0)

# Train NLE: -log q(X|θ)
flow.train()
for step in range(15000):
    th, x = sim.sample(256, rng); th = th.to(DEV); x = x.to(DEV)
    loss = -flow.log_prob(x, context=th).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
flow.eval()
print(f"NLE trained, final -logq={loss.item():.3f}")


def score(theta_row, x):
    """U(θ;X)=∇_θ log q(X|θ) ∈ (n,5), autograd through the MAF context."""
    n = x.shape[0]
    th = torch.tensor(theta_row, dtype=x.dtype, device=DEV).repeat(n, 1).requires_grad_(True)
    lp = flow.log_prob(x, context=th)              # (n,)
    g, = torch.autograd.grad(lp.sum(), th, create_graph=False)
    return g.detach()                               # (n,5)


def fisher(theta_row, n_fish=8000):
    xf = sim.sample_x_given_theta(tuple(theta_row), n_fish, np.random.default_rng(999)).to(DEV)
    U = score(theta_row, xf)                        # (n,5), E[U]=0 at truth
    return (U.T @ U / n_fish).cpu().numpy()         # Fisher ≈ Cov(U)


def rao_coverage(grid, alphas, n_eval=3000):
    worst = 0.0
    perpit = []
    for th0 in grid:
        I = fisher(th0)
        Iinv = np.linalg.inv(I + 1e-6 * np.eye(5))
        L = np.linalg.cholesky(np.linalg.inv(Iinv))  # I^{1/2} via inv(Iinv)
        xv = sim.sample_x_given_theta(tuple(th0), n_eval, np.random.default_rng(7)).to(DEV)
        U = score(th0, xv).cpu().numpy()             # (n,5)
        stat = np.einsum('ni,ij,nj->n', U, Iinv, U)  # Uᵀ I⁻¹ U ~ χ²₅
        for a in alphas:
            cov = float((stat <= chi2.ppf(a, df=5)).mean())
            worst = max(worst, abs(cov - a))
        # per-coord PIT of whitened score r = I^{-1/2} U  (should be ~N(0,1) each)
        Isqrt_inv = np.linalg.cholesky(Iinv)         # lower-tri, (I⁻¹)^{1/2}
        r = U @ Isqrt_inv                            # whiten
        ks = [kstest(norm.cdf(r[:, k]), "uniform").statistic for k in range(5)]
        perpit.append(ks)
    return worst, np.array(perpit)


alphas = [0.5, 0.68, 0.9, 0.95]
grid3 = [(0., 0., 0., 0., 0.), (0.6, -0.5, -0.9, 2.0, -2.0), (-0.6, 0.5, 1.0, -2.0, 2.5)]
w3, pit = rao_coverage(grid3, alphas)
print(f"Rao-score coverage_error_max (3-pt) = {w3:.3f}   [oracle Bartlett ~0.026]")
print("per-coord PIT KS of whitened score (rows θ₀; cols 5):\n", pit.round(3))

rng2 = np.random.default_rng(2)
lows = np.array([np.log(0.4), np.log(0.4), -1.5, -3, -3]); highs = np.array([np.log(2.5), np.log(2.5), 1.5, 3, 3])
lhs = [tuple(lows + (highs - lows) * rng2.random(5)) for _ in range(12)]
wl, _ = rao_coverage(lhs, alphas, n_eval=2000)
print(f"Rao-score coverage_error_max (12-pt LHS) = {wl:.3f}")
