"""CriticalValueProcedure.contains_batch agrees with the per-sample
confidence_set + contains() slow path; timing should be much faster on a
realistic batch size."""
from __future__ import annotations

import time

import torch

from cdsbi.confidence_set.procedures import CriticalValueProcedure


def _toy_procedure():
    """Linear test stat in θ, scalar critical value — simple, deterministic.

    T(θ; X) = (θ - X)²  (Wilks-direction: large = far from X).
    c_α(θ) = chi2_quantile_at_α = 2.7055 at α=0.9 for χ²_1.
    So set = {θ : (θ - X)² ≤ 2.7055} = X ± √2.7055 ≈ X ± 1.6449.
    """
    def test_stat_fn(theta, x_obs):
        # theta (B, 1), x_obs (B, 1) → (B,)
        return ((theta - x_obs) ** 2).squeeze(-1)

    def critical_value_fn(theta, alpha):
        # Constant in θ for this toy: chi2 quantile.
        from scipy.stats import chi2
        return torch.full((theta.shape[0],), float(chi2.ppf(alpha, df=1)))

    return CriticalValueProcedure(
        test_stat_fn=test_stat_fn,
        critical_value_fn=critical_value_fn,
        d_theta=1,
        theta_range=(-10.0, 10.0),
    )


def test_contains_batch_agrees_with_slow_path():
    proc = _toy_procedure()
    x_batch = torch.linspace(-3.0, 3.0, 25).view(-1, 1)
    for alpha in [0.5, 0.9, 0.95]:
        # Slow path: per-sample confidence_set
        slow = torch.tensor([
            proc.confidence_set(x_batch[i : i + 1], alpha=alpha).contains(0.5)
            for i in range(x_batch.shape[0])
        ])
        # Fast path
        fast = proc.contains_batch(0.5, x_batch, alpha=alpha)
        assert torch.equal(slow, fast.cpu()), (
            f"alpha={alpha}: slow={slow.tolist()} fast={fast.cpu().tolist()}"
        )


def test_contains_batch_faster_than_slow_path():
    proc = _toy_procedure()
    B = 1000
    x_batch = torch.randn(B, 1)
    alpha = 0.9
    # Warm-up
    _ = proc.contains_batch(0.0, x_batch[:10], alpha)
    _ = proc.confidence_set(x_batch[0:1], alpha)

    t0 = time.time()
    fast = proc.contains_batch(0.0, x_batch, alpha)
    t_fast = time.time() - t0

    t0 = time.time()
    slow = torch.tensor([
        proc.confidence_set(x_batch[i : i + 1], alpha=alpha).contains(0.0)
        for i in range(B)
    ])
    t_slow = time.time() - t0

    assert torch.equal(slow, fast.cpu())
    # The fast path should be at LEAST 10× faster on a realistic batch.
    # In practice on this toy procedure it's typically 100×+.
    assert t_fast * 10 < t_slow, f"fast={t_fast:.4f}s, slow={t_slow:.4f}s"
