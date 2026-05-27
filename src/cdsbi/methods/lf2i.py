"""LF2I shared components — multi-quantile head + stage-2 trainer.

The canonical LF2I baseline lives in lf2i_bff.py (BFF test statistic).
This module exposes the architecture and training pieces both BFF and any
future LF2I variants (e.g. Waldo) reuse:

  - MultiQuantileMLP       (paper's QuantileNN)
  - multi_pinball_loss     (sum of per-α pinball losses)
  - train_multi_quantile_head (recipe-aware stage-2 trainer)

A simplified fixed-reference LR variant (LF2IRunner) previously lived here
as a stepping stone; it was removed once BFF caught up — the LR test
statistic is structurally degenerate at θ_0 = θ_ref and cannot reach
nominal coverage there regardless of training recipe.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MultiQuantileMLP(nn.Module):
    """Shared MLP trunk + per-quantile linear heads (paper's QuantileNN).

    Parameter count is independent of len(alpha_grid) once `depth` and `hidden`
    are fixed. Each forward returns a (B, n_quantiles) matrix of α-quantile
    predictions in the order `alpha_grid` was given.
    """

    def __init__(self, input_dim: int, hidden: int, depth: int, n_quantiles: int):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_quantiles)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        return torch.cat([h(z) for h in self.heads], dim=-1)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    diff = target - pred
    return torch.mean(torch.maximum(alpha * diff, (alpha - 1.0) * diff))


def multi_pinball_loss(
    preds: torch.Tensor, target: torch.Tensor, alphas: List[float]
) -> torch.Tensor:
    """Sum of per-quantile pinball losses over the shared trunk's heads."""
    losses = []
    for k, a in enumerate(alphas):
        losses.append(pinball_loss(preds[:, k], target, a))
    return torch.stack(losses).sum()


def train_multi_quantile_head(
    critical_net: nn.Module,
    theta_cal: torch.Tensor,
    t_cal: torch.Tensor,
    alpha_grid: List[float],
    cfg: dict,
    device: torch.device,
) -> List[float]:
    """Recipe-aware training for the LF2I stage-2 multi-quantile head.

    Stage 2 is inherently finite-sample (t_cal is pre-computed from the
    stage-1 model on a fixed calibration draw), so we don't go through
    train_with_recipe's sampler-and-fresh-batch path. We do mirror the
    optimizer / LR schedule / batching / grad-clip semantics so the
    quantile head benefits from the same training recipe (e.g.
    adamw_cosine_warmup) as the stage-1 backbone.
    """
    from cdsbi.methods.training_utils import build_optimizer, build_scheduler

    n_steps = int(cfg["n_steps"])
    opt = build_optimizer(critical_net.parameters(), cfg)
    scheduler = build_scheduler(opt, cfg, n_steps)
    bs = int(cfg["batch_size"])
    grad_clip = float(cfg.get("grad_clip_norm", 5.0))
    n_data = int(theta_cal.shape[0])

    losses: List[float] = []
    for _ in range(n_steps):
        idx = torch.randint(0, n_data, (bs,), device=device)
        preds = critical_net(theta_cal[idx])
        loss = multi_pinball_loss(preds, t_cal[idx], alpha_grid)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critical_net.parameters(), max_norm=grad_clip)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
    return losses


