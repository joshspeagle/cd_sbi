"""MLPConditioner: X → T sufficient-statistic reduction with non-zero log|∂T/∂X|.

v3 introduces this as the first conditioner with a non-trivial Jacobian
contribution. For §8.4 the truth is T = Σ X_i (known closed-form sufficient
statistic for the exponential model), so we ship a frozen-sum mode that
hardcodes the reduction and the constant log|∂T/∂X| = ½ log(n_iid). A
trainable-MLP mode for unknown sufficient statistics is a v3.1+ extension.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


class MLPConditioner:
    """Frozen-sum / trainable-MLP sufficient-statistic reducer.

    Currently only `mode='frozen_sum'` is supported; trainable modes land in
    v3.1+ when an unknown-T target arrives.
    """

    def __init__(self, input_dim: int, output_dim: int = 1, mode: str = "frozen_sum"):
        if mode != "frozen_sum":
            raise ValueError(
                f"MLPConditioner mode={mode!r} not supported; only 'frozen_sum' available in v3"
            )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        # log|∂T/∂X| for T = Σ X_i:
        # the map (X_1, …, X_n) → T = Σ X_i has gradient row vector (1, …, 1);
        # interpreted as a directional Jacobian, the magnitude is √n.
        self._log_det_per_row = 0.5 * math.log(float(input_dim))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """X → (T, log|∂T/∂X|).

        Input x: shape (n, input_dim). For frozen-sum mode, T = Σ X_i across
        the last axis, output_dim must equal 1 (sum reduction), and
        log|∂T/∂X| is the constant ½ log(n) per row.
        """
        assert self.mode == "frozen_sum"
        assert self.output_dim == 1, (
            f"frozen_sum mode requires output_dim=1, got {self.output_dim}"
        )
        T = x.sum(dim=-1, keepdim=True)  # (n, 1)
        log_det = torch.full(
            (x.shape[0],), self._log_det_per_row, dtype=x.dtype, device=x.device,
        )
        return T, log_det

    def n_params(self) -> int:
        # frozen-sum mode has no trainable parameters
        return 0
