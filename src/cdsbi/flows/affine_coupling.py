"""AffineCouplingBijection: a hand-rolled RealNVP-style invertible map ℝ^d→ℝ^d.

A stack of affine coupling layers with alternating binary masks. Each layer
transforms the un-masked coords as z' = z·exp(s(z_masked)) + t(z_masked), leaving
the masked coords fixed, so the Jacobian is triangular and log|det| = Σ s. The
log-scales are tanh-bounded for stability. Invertible (information-preserving) —
used as the Stage-B Arm I-A summary so the change-of-variables is exact.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _tanh_mlp(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Sequential:
    assert depth >= 1
    layers: list = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class AffineCouplingBijection(nn.Module):
    def __init__(self, d: int, hidden: int = 64, n_layers: int = 6, depth: int = 2,
                 scale_cap: float = 2.0):
        super().__init__()
        if d < 2:
            raise ValueError("AffineCouplingBijection needs d >= 2")
        self.d = d
        self.scale_cap = scale_cap
        masks = []
        for i in range(n_layers):
            m = torch.zeros(d)
            m[i % 2::2] = 1.0
            masks.append(m)
        self.register_buffer("_masks", torch.stack(masks))
        self.scale_nets = nn.ModuleList([_tanh_mlp(d, hidden, d, depth) for _ in range(n_layers)])
        self.shift_nets = nn.ModuleList([_tanh_mlp(d, hidden, d, depth) for _ in range(n_layers)])
        for net in list(self.scale_nets) + list(self.shift_nets):
            nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)

    def forward(self, x: torch.Tensor):
        z = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for i in range(len(self.scale_nets)):
            mask = self._masks[i]
            z_masked = z * mask
            s = self.scale_cap * torch.tanh(self.scale_nets[i](z_masked)) * (1.0 - mask)
            t = self.shift_nets[i](z_masked) * (1.0 - mask)
            z = z_masked + (1.0 - mask) * (z * torch.exp(s) + t)
            log_det = log_det + s.sum(dim=-1)
        return z, log_det

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
