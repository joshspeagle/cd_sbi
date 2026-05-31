"""Scalable coverage-evaluation engine.

The legacy path looped every θ₀ inside every diagnostic and recomputed the method's
statistic each time, in one un-chunked forward — which made high-d / autograd
(Score-CD) evaluation blow up (≈93 min/run and a 21 GiB OOM at d=5). This engine
fixes that with three moves:

  1. **Statistic-once.** Per θ₀, compute the procedure's statistic a single time
     (the pivot r(θ₀;X) for pivot procedures, or T(θ₀;X) for critical-value ones)
     and derive *all* metrics (coverage at every α, joint ‖r‖²~χ²_d, per-coord PIT)
     from that one cached tensor — no per-(α, diagnostic) recomputation.
  2. **Chunked / memory-bounded.** X is streamed through the procedure in chunks,
     so peak memory is O(chunk) not O(n) — no giant precompute, no OOM.
  3. **Simulate-once.** X|θ₀ is drawn a single time per θ₀ and shared across metrics.

Score-CD's statistic enables grad internally (the score is an autograd backward), so
the engine's `torch.no_grad()` wrapper is harmless — the inner `enable_grad` overrides
it for the score and is detached on the way out.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2, norm, kstest


def _theta_vec(theta_0, d: int) -> np.ndarray:
    v = np.atleast_1d(np.asarray(theta_0, dtype=np.float64)).reshape(-1)
    if v.shape[0] != d:
        raise ValueError(f"theta_0 has {v.shape[0]} comps, expected d={d}")
    return v


def _chunked_apply(fn, theta_vec: np.ndarray, x: torch.Tensor, d: int, chunk: int) -> torch.Tensor:
    """cat_i fn(θ-broadcast_i, x_chunk_i) over chunks of x. Memory is O(chunk)."""
    outs = []
    th_row = torch.from_numpy(np.ascontiguousarray(theta_vec)).to(x.dtype)   # (d,)
    for i in range(0, x.shape[0], chunk):
        xc = x[i:i + chunk]
        th = th_row.unsqueeze(0).expand(xc.shape[0], d)
        with torch.no_grad():                      # Score-CD re-enables grad internally
            outs.append(fn(th, xc).detach())
    return torch.cat(outs, dim=0)


def evaluate_coverage(procedure, simulator, theta_grid: Sequence, alpha_grid: List[float],
                      n_per_theta: int = 2000, chunk_size: int = 512, seed: int = 0,
                      x_per_theta: dict = None) -> dict:
    """Compute coverage (and, for pivot procedures, joint χ²_d KS + per-coord PIT KS)
    over a θ₀ grid, statistic-once and chunked. Returns a dict with a `coverage`
    DataFrame (schema matches the legacy Coverage diagnostic) + summary scalars.

    `x_per_theta`: optional {θ₀-repr → X tensor} to reuse a pre-drawn X|θ₀ (shared
    with other diagnostics); falls back to simulating when a θ₀ is absent.
    """
    d = int(procedure.d_theta)
    rng = np.random.default_rng(seed)
    has_pivot = hasattr(procedure, "pivot")
    has_stat = hasattr(procedure, "test_statistic")
    rows: list = []
    chi2_ks: list = []
    pit_ks: list = []

    for theta_0 in theta_grid:
        v = _theta_vec(theta_0, d)
        x = None
        if x_per_theta is not None:
            x = x_per_theta.get(str(list(map(float, v))))
            if x is not None:
                x = x[:n_per_theta]
        if x is None:
            x = simulator.sample_x_given_theta(theta_0, n_per_theta, rng)   # simulate-once

        if has_pivot:
            r = _chunked_apply(procedure.pivot, v, x, d, chunk_size).cpu().numpy()  # (n,d) once
            sq = (r ** 2).sum(axis=1)
            for a in alpha_grid:
                emp = float((sq <= chi2.ppf(a, df=d)).mean())
                rows.append({**{f"theta_0_{k}": float(v[k]) for k in range(d)},
                             "alpha": float(a), "nominal": float(a), "empirical": emp})
            chi2_ks.append(float(kstest(chi2.cdf(sq, df=d), "uniform").statistic))
            pit_ks.append([float(kstest(norm.cdf(r[:, k]), "uniform").statistic) for k in range(d)])

        elif has_stat:
            T = _chunked_apply(procedure.test_statistic, v, x, d, chunk_size).cpu().numpy()  # (n,) once
            th1 = torch.from_numpy(np.ascontiguousarray(v)).float().unsqueeze(0)
            for a in alpha_grid:
                c = float(procedure.critical_value(th1, a).reshape(-1)[0])
                rows.append({**{f"theta_0_{k}": float(v[k]) for k in range(d)},
                             "alpha": float(a), "nominal": float(a),
                             "empirical": float((T <= c).mean())})
        else:
            # Generic fallback: chunked contains_batch per α (still statistic-light methods).
            for a in alpha_grid:
                inside = _chunked_contains(procedure, theta_0, x, a, chunk_size)
                rows.append({**{f"theta_0_{k}": float(v[k]) for k in range(d)},
                             "alpha": float(a), "nominal": float(a),
                             "empirical": float(inside.float().mean().item())})

    cov = pd.DataFrame(rows)
    out = {"coverage": cov,
           "coverage_error_max": float((cov["empirical"] - cov["nominal"]).abs().max())}
    if has_pivot:
        out["joint_chi2_ks_max"] = float(np.max(chi2_ks))
        out["per_coord_pit_ks"] = np.asarray(pit_ks)
        out["per_coord_pit_ks_max"] = float(np.max(pit_ks))
    return out


def _chunked_contains(procedure, theta_0, x, alpha, chunk):
    masks = []
    for i in range(0, x.shape[0], chunk):
        with torch.no_grad():
            masks.append(procedure.contains_batch(theta_0, x[i:i + chunk], alpha).detach())
    return torch.cat(masks, dim=0)
