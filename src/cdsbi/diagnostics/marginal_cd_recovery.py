"""MarginalCDRecovery: validate the σ² (χ²) and μ (Student-t) marginal CDs.

σ² is direct (r_σ ⟂ μ → Φ(r_σ) is the χ²-based CD). μ requires marginalizing the
σ nuisance out of the joint confidence density → Student-t_{n−1} CD. The primary
recovery metric for each marginal is the KS statistic of its PIT against U(0,1)
(distributional recovery). The pointwise `*_resid` columns are secondary: the gap
between the trained marginalized CD and the closed-form analytic CD, summarised as
the **95th percentile over X** (a max-over-X L∞ statistic grows with sample size
and is dominated by a single tail draw, so it is a poor recovery summary). No-ops
(passed=True) when the procedure is not pivot-based or the simulator lacks a
marginal_cd_spec.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, kstest

from cdsbi.diagnostics.base import DiagnosticResult
from cdsbi.diagnostics.ks_floor import ks_noise_floor


class MarginalCDRecovery:
    name = "marginal_cd_recovery"

    def __init__(self, theta_0_grid: Sequence, n_per_theta: int = 2000):
        self.theta_0_grid = list(theta_0_grid)
        self.n_per_theta = n_per_theta

    def _noop(self, reason: str) -> DiagnosticResult:
        return DiagnosticResult(self.name, value=pd.DataFrame(), passed=True,
                                noise_floor=0.0, n_samples=0, meta={"reason": reason})

    def _marginalize_mu(self, proc, simulator, theta_0, x, sc, lc, chunk=256):
        """H_μ(μ₀|X) for each row of x, by integrating the joint CD over log σ.
        Finite-difference ∂r_σ/∂logσ on a fixed grid; no autograd (pivot_fn moves
        devices, which would detach an external grad leaf on GPU)."""
        mu0 = float(theta_0[lc])
        lo, hi = simulator.log_sigma_range
        grid = torch.linspace(lo - 2.0, hi + 2.0, 257, dtype=x.dtype, device=x.device)  # (K,)
        K = grid.shape[0]
        out = []
        with torch.no_grad():
            for start in range(0, x.shape[0], chunk):
                xb = x[start:start + chunk]                       # (b, n_iid)
                b = xb.shape[0]
                theta = torch.empty(b, K, 2, dtype=x.dtype, device=x.device)
                theta[:, :, sc] = grid.view(1, K)
                theta[:, :, lc] = mu0
                theta = theta.reshape(b * K, 2)
                x_rep = xb.repeat_interleave(K, dim=0)            # (b*K, n_iid)
                r = proc.pivot(theta, x_rep)                      # (b*K, 2)
                r_sigma = r[:, sc].reshape(b, K)
                r_mu = r[:, lc].reshape(b, K)
                grid_dev = grid.to(r_sigma.device)
                dr = torch.gradient(r_sigma, spacing=(grid_dev,), dim=1)[0]   # (b, K)
                phi = torch.exp(-0.5 * r_sigma ** 2) / np.sqrt(2 * np.pi)
                Phi_mu = 0.5 * (1.0 + torch.erf(r_mu / np.sqrt(2.0)))
                integ = phi * dr.abs() * Phi_mu                   # (b, K)
                H = torch.trapezoid(integ, grid_dev, dim=1)       # (b,)
                out.append(H.detach().cpu().numpy())
        return np.concatenate(out)

    def __call__(self, trained, simulator, eval_data=None, x_per_theta=None) -> DiagnosticResult:
        proc = getattr(trained, "procedure", None)
        if proc is None or not hasattr(proc, "pivot"):
            return self._noop("procedure is not pivot-based")
        spec = getattr(simulator, "marginal_cd_spec", None)
        if spec is None or not hasattr(simulator, "analytic_marginal_cd_pit"):
            return self._noop("simulator has no marginal_cd_spec")
        sc = spec["scale_coord"]
        floor = ks_noise_floor(self.n_per_theta)
        rows = []
        for theta_0 in self.theta_0_grid:
            key = repr([float(v) for v in theta_0])
            x = x_per_theta[key] if x_per_theta and key in x_per_theta else \
                simulator.sample_x_given_theta(theta_0, self.n_per_theta, np.random.default_rng(0))
            n = x.shape[0]
            theta_t = torch.tensor([[float(v) for v in theta_0]], dtype=x.dtype).expand(n, -1)
            with torch.no_grad():
                r = proc.pivot(theta_t, x)
            sigma_pit = norm.cdf(r[:, sc].detach().cpu().numpy())
            analytic = simulator.analytic_marginal_cd_pit(theta_0, x)
            sigma_ks = float(kstest(sigma_pit, "uniform").statistic)
            sigma_chi2_resid = float(np.abs(sigma_pit - analytic["sigma_pit"].numpy()).max())
            lc = spec["location_coord"]
            H_mu = np.clip(self._marginalize_mu(proc, simulator, theta_0, x, sc, lc), 0.0, 1.0)
            mu_ks = float(kstest(H_mu, "uniform").statistic)
            # 95th-percentile per-X residual vs the analytic t-CD (robust to a
            # single tail draw; the KS statistic above is the primary check).
            mu_t_resid = float(np.quantile(np.abs(H_mu - analytic["mu_pit"].numpy()), 0.95))
            rows.append({
                "theta_0": key, "sigma_ks": sigma_ks,
                "sigma_chi2_resid": sigma_chi2_resid, "noise_floor": floor,
                "mu_ks": mu_ks, "mu_t_resid": mu_t_resid,
            })
        df = pd.DataFrame(rows)
        passed = bool((df["sigma_ks"] <= 2.0 * floor).all() and (df["mu_ks"] <= 2.0 * floor).all())
        return DiagnosticResult(self.name, value=df, passed=passed,
                                noise_floor=floor, n_samples=self.n_per_theta,
                                meta={"scale_coord": sc})
