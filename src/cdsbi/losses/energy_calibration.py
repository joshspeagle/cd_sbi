"""EnergyCalibrationLoss: a reparameterization-invariant calibration objective.

For a group of pivot values R={r_j}⊂ℝ^d drawn at a common θ₀ and reference
Z~N(0,I_d), the energy score
    S(R) = (2/(m·M)) Σ_ij‖r_i−z_j‖ − (1/m²) Σ_ii'‖r_i−r_i'‖   (ref-ref term dropped)
is a strictly proper score minimized exactly when R ~ N(0,I_d). It sees only the
*distribution* of r — there is NO Jacobian/density term — so the Stage-B
information-collapse cheat (which inflates a feature-Jacobian) has no channel here.
Loss = mean of S over the B groups in a batch. Requires only R1 (monotone-in-θ) of
the flow — confidence sets still invert the pivot; R2 is unused.
"""
from __future__ import annotations

import torch

from cdsbi.flows.base import Guarantee
from cdsbi.losses.base import Loss, MonotonicityMismatchError


class EnergyCalibrationLoss(Loss):
    required_guarantees = frozenset({Guarantee.R1})

    def __init__(self, n_ref: int = 256):
        self.n_ref = n_ref

    def check_guarantees(self, flow) -> None:
        guarantees = getattr(flow, "monotonicity_guarantees", frozenset())
        if not self.required_guarantees.issubset(guarantees):
            missing = self.required_guarantees - guarantees
            raise MonotonicityMismatchError(
                f"EnergyCalibrationLoss requires {sorted(g.value for g in self.required_guarantees)}; "
                f"flow {type(flow).__name__} provides {sorted(g.value for g in guarantees)}; "
                f"missing {sorted(g.value for g in missing)}."
            )

    def population_lower_bound(self, simulator) -> None:
        return None

    def score(self, r_groups: torch.Tensor, rng_seed: int | None = None,
              generator: torch.Generator | None = None) -> torch.Tensor:
        """r_groups: (B, m, d). Returns the mean energy score (scalar)."""
        B, m, d = r_groups.shape
        if generator is None and rng_seed is not None:
            generator = torch.Generator(device=r_groups.device).manual_seed(int(rng_seed))
        z = torch.randn(self.n_ref, d, device=r_groups.device, dtype=r_groups.dtype,
                        generator=generator)                      # (M, d)
        # cross term: 2 * mean_{i,j} ||r_i - z_j||  per group
        cross = torch.cdist(r_groups, z.unsqueeze(0).expand(B, -1, -1))   # (B, m, M)
        term1 = 2.0 * cross.mean(dim=(1, 2))                              # (B,)
        # within term: mean_{i,i'} ||r_i - r_i'||  per group. The m zero-diagonal
        # self-distances add 0 and are kept (mean over m² not m(m−1)); this only
        # rescales the repulsion by the constant (m−1)/m, which cancels in the
        # gradient direction and does not move the N(0,I) minimizer. cdist's
        # zero-diagonal yields finite grads on modern torch (no eps needed).
        within = torch.cdist(r_groups, r_groups)                          # (B, m, m)
        term2 = within.mean(dim=(1, 2))                                   # (B,)
        return (term1 - term2).mean()
