"""Conditioner protocol: encode X into a context vector + a log-det contribution."""
from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import torch


@runtime_checkable
class Conditioner(Protocol):
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (context_vec, log_det_jac_input_contribution).

        log_det_jac_input_contribution has shape (n,); for X → X (identity) it is zero.
        For an X → T sufficient-statistic reduction (v3+) it is log |∂T/∂X|.
        """
        ...

    def n_params(self) -> int: ...
