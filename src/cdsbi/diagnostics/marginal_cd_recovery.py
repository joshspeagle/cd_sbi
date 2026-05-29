"""MarginalCDRecovery: validate the σ² (χ²) and μ (Student-t) marginal CDs.

σ² is direct (r_σ ⟂ μ → Φ(r_σ) is the χ²-based CD). μ requires marginalizing the
σ nuisance out of the joint confidence density → Student-t_{n−1} CD (added in a
later task). No-ops (passed=True) when the procedure is not pivot-based or the
simulator lacks a marginal_cd_spec.
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
            rows.append({
                "theta_0": key, "sigma_ks": sigma_ks,
                "sigma_chi2_resid": sigma_chi2_resid, "noise_floor": floor,
            })
        df = pd.DataFrame(rows)
        passed = bool((df["sigma_ks"] <= 2.0 * floor).all())
        return DiagnosticResult(self.name, value=df, passed=passed,
                                noise_floor=floor, n_samples=self.n_per_theta,
                                meta={"scale_coord": sc})
