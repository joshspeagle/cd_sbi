"""Recipe-aware training building blocks shared across method runners.

Extracts the optimizer factory, LR scheduler factory, and the training loop
mechanics (fresh_batch vs finite-sample, batching mode, grad clipping) that
originally lived inside CDSBIRunner.fit. By centralising them here, the
sbi-wrapped runners (NPE / NLE / NRE / LF2I / LF2I-BFF) can train their own
underlying nets with the same recipe knobs CDSBI uses — closing the
apples-to-oranges seam where sbi.train() ignored optimizer / lr_schedule /
warmup / grad_clip / etc.

The shared loop is plug-in via a method-specific `loss_fn(net, theta_b, x_b)`
closure; see uses in nle.py, npe.py, nre.py, lf2i.py, lf2i_bff.py.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch


def build_optimizer(params, cfg: dict) -> torch.optim.Optimizer:
    """Build a torch optimizer from a recipe config dict."""
    opt_name = cfg.get("optimizer", "adam")
    lr = float(cfg["lr"])
    weight_decay = float(cfg.get("weight_decay", 0.0))
    betas = tuple(cfg.get("betas", [0.9, 0.999]))
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas)
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if opt_name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {opt_name!r}")


def build_scheduler(
    opt: torch.optim.Optimizer, cfg: dict, n_steps: int
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Build an LR scheduler from a recipe config dict."""
    sched_name = cfg.get("lr_schedule", "constant")
    if sched_name == "constant":
        return None
    lr = float(cfg["lr"])
    lr_min_ratio = float(cfg.get("lr_min_ratio", 0.0))
    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=lr * lr_min_ratio,
        )
    if sched_name == "warmup_cosine":
        warmup_steps = int(cfg.get("warmup_steps", 0))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_ratio + (1.0 - lr_min_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    if sched_name == "exponential":
        gamma = float(cfg.get("lr_gamma", 0.999))
        return torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    raise ValueError(f"Unknown lr_schedule: {sched_name!r}")


def train_with_recipe(
    net: torch.nn.Module,
    sampler: Callable,
    cfg: dict,
    device: torch.device,
    loss_fn: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    rngs,
    n_train: Optional[int] = None,
) -> Tuple[list, float]:
    """Generic recipe-aware training loop.

    Args:
        net: nn.Module to train (moved to `device` inside).
        sampler: callable producing (theta, x) tensors, taking (n, rng).
        cfg: training config dict. Required: lr, batch_size, n_steps.
            Optional: optimizer, weight_decay, betas, momentum, lr_schedule,
            warmup_steps, lr_min_ratio, lr_gamma, batching, grad_clip_norm,
            fresh_batch, n_train.
        device: target device.
        loss_fn: closure (net, theta_b, x_b) -> scalar loss tensor. Returns
            the loss to .backward() each step.
        rngs: NumpyRNGTuple from seed_everything; uses rngs.train (sampling)
            and rngs.init (shuffle permutations).
        n_train: pre-sample size if fresh_batch=False; defaults to
            cfg["n_train"].

    Returns:
        (losses, wall_clock_sec): list of per-step loss values + elapsed time.
    """
    import time

    net.to(device)
    n_steps = int(cfg["n_steps"])
    opt = build_optimizer(net.parameters(), cfg)
    scheduler = build_scheduler(opt, cfg, n_steps)

    fresh_batch = bool(cfg.get("fresh_batch", True))
    batching = cfg.get("batching", "random_replacement")
    bs = int(cfg["batch_size"])
    grad_clip = float(cfg.get("grad_clip_norm", 5.0))

    if not fresh_batch:
        n_eff = int(n_train if n_train is not None else cfg["n_train"])
        theta_all, x_all = sampler(n_eff, rngs.train)
        theta_all = theta_all.to(device)
        x_all = x_all.to(device)

    losses: list = []
    t0 = time.time()
    epoch_idx_iter: Optional[tuple] = None

    for step in range(n_steps):
        if fresh_batch:
            theta_b, x_b = sampler(bs, rngs.train)
            theta_b = theta_b.to(device)
            x_b = x_b.to(device)
        else:
            if batching == "random_replacement":
                idx = torch.randint(0, n_eff, (bs,))
            elif batching == "shuffle_epoch":
                if epoch_idx_iter is None or epoch_idx_iter[1] >= n_eff:
                    perm = rngs.init.permutation(n_eff)
                    epoch_idx_iter = (torch.from_numpy(perm).long(), 0)
                start = epoch_idx_iter[1]
                end = min(start + bs, n_eff)
                idx = epoch_idx_iter[0][start:end]
                if end - start < bs:
                    perm2 = rngs.init.permutation(n_eff)
                    extra = torch.from_numpy(perm2[: bs - (end - start)]).long()
                    idx = torch.cat([idx, extra])
                    epoch_idx_iter = (torch.from_numpy(perm2).long(), bs - (end - start))
                else:
                    epoch_idx_iter = (epoch_idx_iter[0], end)
            else:
                raise ValueError(f"Unknown batching: {batching!r}")
            theta_b = theta_all[idx]
            x_b = x_all[idx]

        loss_val = loss_fn(net, theta_b, x_b)
        opt.zero_grad()
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss_val.item())

    return losses, time.time() - t0
