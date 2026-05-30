"""autoregressive_invert: invert a SingleIndexMonotoneFlow coordinate-by-coordinate.

Each coordinate r_k is strictly monotone in θ_k (sign = flow._s_theta[k]) and depends
only on θ_{≤k}, so we solve θ_k by bisection given θ_{<k} (already solved). Vectorized
over the batch; used to SAMPLE the CD (draw r~N(0,I), invert → θ-samples).
"""
from __future__ import annotations

import torch


def autoregressive_invert(flow, r_target: torch.Tensor, context: torch.Tensor,
                          lo: float = -8.0, hi: float = 8.0, iters: int = 40) -> torch.Tensor:
    """r_target (n,d), context (n,d) → θ (n,d) with flow.forward(θ, context)[:, k] ≈ r_target[:, k]."""
    n, d = r_target.shape
    theta = torch.zeros(n, d, dtype=r_target.dtype, device=r_target.device)
    for k in range(d):
        sign = float(flow._s_theta[k])
        lo_k = torch.full((n,), lo, dtype=r_target.dtype, device=r_target.device)
        hi_k = torch.full((n,), hi, dtype=r_target.dtype, device=r_target.device)
        for _ in range(iters):
            mid = 0.5 * (lo_k + hi_k)
            theta[:, k] = mid
            rk = flow.forward(theta, context=context)[0][:, k]
            below = rk < r_target[:, k]
            if sign > 0:
                lo_k = torch.where(below, mid, lo_k); hi_k = torch.where(below, hi_k, mid)
            else:
                hi_k = torch.where(below, mid, hi_k); lo_k = torch.where(below, lo_k, mid)
        theta[:, k] = 0.5 * (lo_k + hi_k)
    return theta
