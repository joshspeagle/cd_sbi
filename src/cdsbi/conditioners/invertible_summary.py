"""InvertibleSummaryConditioner: Stage-B Arm I-A learned summary.

Wraps an AffineCouplingBijection s_φ: ℝ^{n_iid}→ℝ^{n_iid}. The first d_theta outputs
S are the inference features (→ the monotone pivot); the rest A are ancillary
(modelled N(0,I) in the exact-density loss). encode(X) returns (S, log|∂s_φ/∂X|) for
the Conditioner protocol + SufficiencyRecovery; transform(X) exposes the full latent
z=(S,A) for the exact-density runner. Invertible ⇒ information cannot be destroyed.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from cdsbi.flows.affine_coupling import AffineCouplingBijection


class InvertibleSummaryConditioner(nn.Module):
    def __init__(self, n_iid: int, d_theta: int = 2, hidden: int = 64,
                 n_layers: int = 6, depth: int = 2):
        super().__init__()
        self.n_iid = n_iid
        self.d_theta = d_theta
        self.bijection = AffineCouplingBijection(n_iid, hidden=hidden, n_layers=n_layers, depth=depth)

    def transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.bijection(x)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, log_det = self.bijection(x)
        return z[:, :self.d_theta], log_det

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
