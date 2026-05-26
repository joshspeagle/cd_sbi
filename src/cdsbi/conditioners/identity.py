"""Identity conditioner: returns X as the context, zero log-det contribution."""
from __future__ import annotations

from typing import Tuple

import torch


class Identity:
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, torch.zeros(x.shape[0], device=x.device)

    def n_params(self) -> int:
        return 0
