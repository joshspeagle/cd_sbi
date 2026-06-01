"""FisherRecovery: efficiency = Fisher-information preservation (theory §11, Cor 12).
Reports det I_h / det I_X at a θ₀, with I_X the true Fisher info (Cov of the autograd
score ∇_θ log p(X|θ)) and I_h = Cov(Ê[U_X|h]), Ê[·|h] a degree-3 POLYNOMIAL LS fit of
the score on the summary. (Linear undershoots: the σ-score is linear in s² but the
feature is log s², so a linear fit caps even a sufficient summary at ~0.80; degree-3
recovers the oracle to ~1.0 — verified: oracle 0.999, σ-dropping 0.10.) Ratio → 1 ⟺
summary sufficient (efficient).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch


def _true_score(simulator, theta0: Sequence[float], x: torch.Tensor) -> torch.Tensor:
    """U_X = ∇_θ log p(X|θ₀), shape (n, d_θ), via autograd of simulator.log_prob."""
    d = len(theta0)
    # build a LEAF tensor with requires_grad (repeat→non-leaf, so .detach() first)
    th = (torch.tensor(theta0, dtype=torch.float32).unsqueeze(0)
          .repeat(x.shape[0], 1).detach().requires_grad_(True))
    lp = simulator.log_prob(x, th)                      # (n,)
    grad = torch.autograd.grad(lp.sum(), th)[0]         # (n, d_θ)
    return grad.reshape(-1, d)


def _cov(a: torch.Tensor) -> np.ndarray:
    a = a - a.mean(dim=0, keepdim=True)
    return (a.T @ a / (a.shape[0] - 1)).detach().cpu().numpy()


def _poly_features(h: np.ndarray, degree: int = 3) -> np.ndarray:
    """[1, h, h², …, h^degree (per-coord), pairwise cross terms] — captures the
    nonlinear sufficiency a linear fit misses."""
    n, d = h.shape
    cols = [np.ones((n, 1)), h]
    for p in range(2, degree + 1):
        cols.append(h ** p)
    for i in range(d):
        for j in range(i + 1, d):
            cols.append((h[:, i] * h[:, j]).reshape(-1, 1))
    return np.concatenate(cols, axis=1)


def fisher_det_ratio(simulator, theta0: Sequence[float],
                     encode_fn: Callable[[object, torch.Tensor], torch.Tensor],
                     n_samples: int = 40000, seed: int = 0) -> float:
    """det I_h / det I_X at θ₀. encode_fn(simulator, x) -> (n, d_θ) summary."""
    rng = np.random.default_rng(seed)
    x = simulator.sample_x_given_theta(np.asarray(theta0), n_samples, rng)
    u = _true_score(simulator, theta0, x)               # (n, d_θ)
    h = encode_fn(simulator, x).detach()                # (n, d_θ)
    I_X = _cov(u)
    # Ê[U_X | h] via degree-3 polynomial LS (linear caps the oracle at ~0.80).
    H = _poly_features(h.cpu().numpy(), degree=3)
    beta, *_ = np.linalg.lstsq(H, u.cpu().numpy(), rcond=None)
    u_hat = torch.from_numpy(H @ beta).float()
    I_h = _cov(u_hat)
    return float(np.linalg.det(I_h) / max(np.linalg.det(I_X), 1e-30))
