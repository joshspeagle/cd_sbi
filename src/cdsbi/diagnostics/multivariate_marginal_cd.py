"""MultivariateMarginalCDRecovery: d=5 (μ,Σ) marginal-CD recovery.

(i) three covariance DIRECT PITs Φ(r_k(θ₀;X)), k=1,2,3 (the Bartlett χ²/χ²/N
    pivots) — KS vs U + max residual vs the analytic reference;
(ii) the μ JOINT Hotelling-T² recovery — sample r~N(0,I₅) per dataset, invert the
    trained pivot (autoregressive_invert) → θ-samples, take the μ-block, form its
    Hotelling F-stat vs the dataset's sample cov S, KS-compare pooled F-stats to
    F_{p,n−p}. (One r-sample per dataset; pooled over the n_per_theta datasets.)
No-ops unless the procedure exposes encode_fn and the simulator analytic_marginal_cd_pit.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, kstest, f as fdist

from cdsbi.diagnostics.base import DiagnosticResult
from cdsbi.diagnostics.ks_floor import ks_noise_floor
from cdsbi.flows.invert import autoregressive_invert


class MultivariateMarginalCDRecovery:
    name = "multivariate_marginal_cd"

    def __init__(self, theta_0_grid: Sequence, n_per_theta: int = 2000, seed: int = 0):
        self.theta_0_grid = list(theta_0_grid)
        self.n_per_theta = n_per_theta
        self.seed = seed

    def _noop(self, reason):
        return DiagnosticResult(self.name, value=pd.DataFrame(), passed=True,
                                noise_floor=0.0, n_samples=0, meta={"reason": reason})

    def __call__(self, trained, simulator, eval_data=None, x_per_theta=None) -> DiagnosticResult:
        proc = getattr(trained, "procedure", None)
        encode_fn = getattr(proc, "encode_fn", None)
        # Gate on `p` (the bivariate (μ,Σ) sim sets self.p=2). CRITICAL: the 1-D
        # NormalUnknownMeanVar ALSO has analytic_marginal_cd_pit (from M1) and
        # encode_fn is always set, so encode_fn+analytic alone is NOT a sufficient
        # gate — it would fire on a mu_sigma run and IndexError on the 2-wide r.
        # `hasattr(simulator, "p")` excludes the 1-D sim (which has no p attribute).
        if (encode_fn is None or not hasattr(simulator, "analytic_marginal_cd_pit")
                or not hasattr(simulator, "p")):
            return self._noop("not a multivariate (μ,Σ) target with a learned summary")
        n = simulator.n_iid; p = simulator.p
        floor = ks_noise_floor(self.n_per_theta)
        rng = np.random.default_rng(self.seed)
        rows = []
        for theta_0 in self.theta_0_grid:
            x = simulator.sample_x_given_theta(theta_0, self.n_per_theta, rng)
            m = x.shape[0]
            theta_t = torch.tensor([[float(v) for v in theta_0]], dtype=x.dtype).expand(m, -1)
            with torch.no_grad():
                r = proc.pivot(theta_t, x)
            analytic = simulator.analytic_marginal_cd_pit(theta_0, x)
            row = {"theta_0": repr([float(v) for v in theta_0]), "noise_floor": floor}
            for j, key in enumerate(["cov1", "cov2", "cov3"]):
                pit = norm.cdf(r[:, j].detach().cpu().numpy())
                row[f"{key}_ks"] = float(kstest(pit, "uniform").statistic)
                row[f"{key}_resid"] = float(np.abs(pit - analytic[f"{key}_pit"].numpy()).max())
            # μ Hotelling-recovery: invert one r-sample/dataset → μ-block → F-stat
            mu_ks = self._mu_hotelling(proc, simulator, x, rng, n, p)
            row["mu_hotelling_ks"] = mu_ks
            rows.append(row)
        df = pd.DataFrame(rows)
        passed = bool(df[["cov1_ks", "cov2_ks", "cov3_ks", "mu_hotelling_ks"]].to_numpy().max()
                      <= 2.0 * floor + 0.02)
        return DiagnosticResult(self.name, value=df, passed=passed, noise_floor=floor,
                                n_samples=self.n_per_theta, meta={})

    def _sample_mu_marginal(self, proc, simulator, x, n, p):
        """Sample the marginal μ-CD per dataset (one draw each). Returns μ-samples (m,p).
        Trained path: r~N(0,I_d) → autoregressive_invert(flow) → θ-samples → μ-block.
        Oracle path (flow is None): the validated closed-form Bartlett inversion —
        invert the covariance pivots from r_cov, then μ = X̄ + (1/√n)·C·z."""
        feat = proc.encode_fn(x)                                     # (m, d_theta)
        m = x.shape[0]
        flow = getattr(proc, "flow", None)
        if flow is not None:
            r_samp = torch.randn(m, simulator.d_theta, dtype=x.dtype)
            with torch.no_grad():
                theta_s = autoregressive_invert(flow, r_samp, feat)
            return theta_s[:, 3:5].cpu().numpy()
        # --- closed-form Bartlett fallback (oracle r*): the prototype-validated path ---
        from scipy.stats import chi2 as _chi2
        f = feat.cpu().numpy()
        D11 = np.exp(f[:, 0]); D22 = np.exp(f[:, 1]); D21 = f[:, 2]
        xbar = f[:, 3:5]                                             # (m,2) = (X̄₁, X̄₂)
        rc = np.random.default_rng(self.seed + 1).standard_normal((m, 3))
        z = np.random.default_rng(self.seed + 2).standard_normal((m, 2))
        T11sq = _chi2.ppf(np.clip(1 - norm.cdf(rc[:, 0]), 1e-12, 1 - 1e-12), n - 1)
        T22sq = _chi2.ppf(np.clip(1 - norm.cdf(rc[:, 1]), 1e-12, 1 - 1e-12), n - 2)
        C11 = D11 / np.sqrt(T11sq); C22 = D22 / np.sqrt(T22sq)
        L21 = (D21 - rc[:, 2] * C22) * C11 / D11
        mu = np.empty((m, 2))
        mu[:, 0] = xbar[:, 0] + (C11 * z[:, 0]) / math.sqrt(n)
        mu[:, 1] = xbar[:, 1] + (L21 * z[:, 0] + C22 * z[:, 1]) / math.sqrt(n)
        return mu

    def _mu_hotelling(self, proc, simulator, x, rng, n, p) -> float:
        mu_s = self._sample_mu_marginal(proc, simulator, x, n, p)    # (m,2) marginal-μ draws
        obs = x.reshape(x.shape[0], n, p)
        xbar = obs.mean(dim=1).cpu().numpy()
        Xc = (obs - obs.mean(dim=1, keepdim=True)).cpu().numpy()
        S = np.einsum('mki,mkj->mij', Xc, Xc) / (n - 1)
        d = mu_s - xbar
        T2 = n * np.einsum('mi,mij,mj->m', d, np.linalg.inv(S), d)
        F = T2 * (n - p) / (p * (n - 1))
        return float(kstest(F, "f", args=(p, n - p)).statistic)
