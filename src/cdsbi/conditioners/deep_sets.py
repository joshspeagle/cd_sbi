"""DeepSetsConditioner: permutation-invariant learned summary s_φ(X)=ρ(mean_i φ(X_i)).

Stage-B learned summary for the unknown-(μ,σ²) target. Device-1 (spec §B.3):
log_det_contrib = 0 (the summary is a feature map, not part of the NF-MLE
change-of-variables) + running feature standardization (BatchNorm1d-style,
affine-free) so the network cannot exploit feature scale to cheat the objective.
DeepSets with mean-pooling can represent (X̄, X̄²)→(X̄, s²) exactly, so a
sufficient summary is within its class — whether training finds it is the
Stage-B (M3) question.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _tanh_mlp(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Sequential:
    assert depth >= 1
    layers: list = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class DeepSetsConditioner(nn.Module):
    def __init__(self, n_iid: int, d_out: int = 2, hidden: int = 64, depth: int = 2,
                 momentum: float = 0.1):
        super().__init__()
        self.n_iid = n_iid
        self.d_out = d_out
        self.momentum = momentum
        self.phi = _tanh_mlp(1, hidden, hidden, depth)        # per-element ℝ→ℝ^h
        self.rho = _tanh_mlp(hidden, hidden, d_out, depth)    # pooled ℝ^h→ℝ^{d_out}
        self.register_buffer("running_mean", torch.zeros(d_out))
        self.register_buffer("running_var", torch.ones(d_out))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[-1] == self.n_iid, (
            f"DeepSetsConditioner expected {self.n_iid} iid obs per row, got {x.shape[-1]}"
        )
        n, m = x.shape
        h = self.phi(x.reshape(n * m, 1)).reshape(n, m, -1).mean(dim=1)   # (n, hidden)
        feats = self.rho(h)                                              # (n, d_out)
        if self.training:
            batch_mean = feats.mean(dim=0).detach()
            batch_var = feats.var(dim=0, unbiased=False).detach()
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * batch_var)
            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var
        feats = (feats - mean) / torch.sqrt(var + 1e-5)
        log_det = torch.zeros(n, dtype=x.dtype, device=x.device)
        return feats, log_det

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
