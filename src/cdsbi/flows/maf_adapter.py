"""MAFAdapter: wraps nflows MaskedAutoregressiveFlow with the Flow protocol.

MAF does not advertise monotonicity in either θ or X — it's a generic
density estimator. Used as the natural backbone for NPE / NLE / LF2I
baselines so their flow capacity is matched to CDSBI's at the same
budget.

Exposes log_prob(x, context) for NPE/NLE wrappers; the Flow.forward
contract is not used for these baselines (they don't produce a pivot).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow as NFlow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation

from cdsbi.flows.base import Guarantee


class MAFAdapter(nn.Module):
    monotonicity_guarantees = frozenset()  # no monotonicity claims

    def __init__(
        self,
        features: int,
        context_features: int,
        hidden: int = 32,
        num_layers: int = 4,
    ):
        super().__init__()
        transforms = []
        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=features))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=hidden,
                    context_features=context_features if context_features > 0 else None,
                    num_blocks=1,
                )
            )
        transform = CompositeTransform(transforms)
        base = StandardNormal(shape=[features])
        self.flow = NFlow(transform=transform, distribution=base)

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.flow.log_prob(inputs=x, context=context)

    def sample(self, n: int, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.flow.sample(num_samples=n, context=context)

    def forward(
        self, theta: torch.Tensor, context: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flow-protocol shim — MAF is not used as a pivot, so this is unused in v0.

        Returns (r, log_det) so the protocol is satisfied; callers
        responsible for using log_prob() instead.
        """
        raise NotImplementedError("MAFAdapter is used via log_prob/sample, not forward().")

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
