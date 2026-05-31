"""LF2I-BFF benchmark on SLCP — a Fisher-free, calibration-based, Neyman-inverted
method (vs Score-CD which broke: asymptotic 0.111 / calibrated 0.648). SLCP prior is a
true box U(-3,3)^5 → MC-from-proposal marginal is clean. asinh data-stabilization
(feature transform; coverage validity unaffected) for a fair classifier scale."""
import numpy as np, torch, time
from cdsbi.methods.lf2i_bff import LF2IBFFRunner


class SLCP:
    d_theta = 5
    theta_range = (-3.0, 3.0)
    @property
    def d_x(self): return 8
    def _x(self, t, rng):
        n = t.shape[0]; m = t[:, :2]; s1 = t[:, 2]**2; s2 = t[:, 3]**2; rho = np.tanh(t[:, 4])
        cov = np.zeros((n, 2, 2)); cov[:,0,0]=s1**2; cov[:,1,1]=s2**2; cov[:,0,1]=rho*s1*s2; cov[:,1,0]=cov[:,0,1]
        L = np.linalg.cholesky(cov + 1e-6*np.eye(2)); z = rng.standard_normal((n, 4, 2))
        return (m[:, None, :] + np.einsum('nij,nkj->nki', L, z)).reshape(n, 8)
    def sample(self, n, rng):
        th = rng.uniform(-3, 3, (n, 5))
        return torch.tensor(th, dtype=torch.float32), torch.tensor(np.arcsinh(self._x(th, rng)), dtype=torch.float32)
    def sample_x_given_theta(self, t0, n, rng):
        th = np.tile(np.array(t0, float), (n, 1))
        return torch.tensor(np.arcsinh(self._x(th, rng)), dtype=torch.float32)


sim = SLCP()
runner = LF2IBFFRunner(classifier_hidden=128, classifier_depth=2,
                       quantile_hidden=64, quantile_depth=2, marginal_n=2048, device="auto")
alpha_grid = [0.5, 0.68, 0.9, 0.95]
config = {"lr":1e-3,"batch_size":256,"n_steps":10000,"n_train":40000,"fresh_batch":True,
          "optimizer":"adam","weight_decay":0.0,"betas":[0.9,0.999],"momentum":0.9,
          "lr_schedule":"constant","warmup_steps":0,"lr_min_ratio":0.0,"lr_gamma":0.999,
          "batching":"random_replacement","grad_clip_norm":5.0,
          "n_train_stat":40000,"n_train_quantile":20000,"alpha_grid":alpha_grid}
t0 = time.time()
trained = runner.fit(simulator=sim, config=config, seed=0)
proc = trained.procedure
print(f"LF2I-BFF on SLCP trained ({time.time()-t0:.0f}s, stage1 loss {trained.final_loss:.3f})", flush=True)

grid = [(0.5,-0.5,1.2,1.0,0.3), (-1.0,1.0,1.5,0.8,-0.4), (1.5,1.5,0.8,1.2,0.5), (0.,0.,1.0,1.0,0.), (-1.5,-1.0,1.3,0.9,-0.2)]
worst = 0.0
for th0 in grid:
    xv = sim.sample_x_given_theta(th0, 2000, np.random.default_rng(7))
    row = []
    for a in alpha_grid:
        inside = proc.contains_batch(th0, xv, a).float().mean().item()
        e = abs(inside - a); worst = max(worst, e); row.append(f"α{a}:{inside:.2f}(e{e:.02f})")
    print(f"  θ0={tuple(round(t,1) for t in th0)}: " + " ".join(row), flush=True)
print(f"\ncoverage_error_max (5-pt grid) = {worst:.3f}   [Score-CD: 0.111 asymptotic / 0.648 calibrated]", flush=True)

rng = np.random.default_rng(2)
lhs = rng.uniform(-3, 3, (16, 5))
wl = 0.0
for th0 in lhs:
    xv = sim.sample_x_given_theta(tuple(th0), 1500, np.random.default_rng(7))
    for a in alpha_grid:
        wl = max(wl, abs(proc.contains_batch(tuple(th0), xv, a).float().mean().item() - a))
print(f"coverage_error_max (16-pt LHS over the box) = {wl:.3f}", flush=True)
