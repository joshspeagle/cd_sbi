"""Monotone-in-level quantile post-processing (spec §4.3.1, plan step 3).

The existing per-level `MultiQuantileMLP` (methods/lf2i.py) trains independent
heads with no cross-level coupling, so its predicted quantiles can CROSS (the
documented U-shape budget-extreme artifact). The default fix — chosen over
cross-quantile regularization and MCQRNN-style architectures in the plan — is
the Chernozhukov–Fernández-Val–Galichon rearrangement: sort the predictions
along the level axis. Pure post-processing (no retraining), idempotent, and it
preserves the multiset of values per row; CFG (Econometrica 2010) show the
rearranged curve is weakly closer to the truth in Lp.

Assumes the level grid the head was trained on is ASCENDING (e.g. the repo's
alpha_grid of confidence levels [0.5, 0.68, 0.9, 0.95]); the sorted outputs are
then aligned level-to-quantile.
"""
from __future__ import annotations

import torch


def rearrange_quantiles(q: torch.Tensor) -> torch.Tensor:
    """Sort predicted quantiles along the last (level) axis — non-crossing by
    construction. Shape-preserving; idempotent; per-row multiset-preserving."""
    return torch.sort(q, dim=-1).values


class MonotoneQuantileHead(torch.nn.Module):
    """Thin wrapper: any per-level quantile head -> rearranged (non-crossing) output.

    `base` is any module mapping (B, input_dim) -> (B, n_levels) for an ascending
    level grid (e.g. `MultiQuantileMLP`). Training happens on the base as usual;
    wrap for inference (or wrap before training — pinball loss through a sort is
    still valid, but the plan's default is post-hoc).
    """

    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange_quantiles(self.base(x))
