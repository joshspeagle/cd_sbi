"""MLPConditioner — frozen-sum sufficient-statistic reduction for §8.4."""
from __future__ import annotations

import math

import torch


def test_mlp_conditioner_frozen_sum_reduction():
    from cdsbi.conditioners.mlp import MLPConditioner
    cond = MLPConditioner(input_dim=5, output_dim=1, mode="frozen_sum")
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [0.1, 0.2, 0.3, 0.4, 0.5]])
    T, log_det = cond.encode(x)
    assert T.shape == (2, 1)
    assert log_det.shape == (2,)
    # Frozen-sum mode: T should equal Σ X_i exactly.
    torch.testing.assert_close(T.squeeze(-1), torch.tensor([15.0, 1.5]))
    # log|∂T/∂X| for T = Σ X_i with n=5: |∂T/∂X| = √5 row-wise (directional gradient norm).
    expected_log_det = 0.5 * math.log(5.0)
    torch.testing.assert_close(log_det, torch.full((2,), expected_log_det))


def test_mlp_conditioner_frozen_sum_no_trainable_params():
    """Frozen-sum mode has zero learnable parameters — the reduction is hardcoded."""
    from cdsbi.conditioners.mlp import MLPConditioner
    cond = MLPConditioner(input_dim=5, output_dim=1, mode="frozen_sum")
    assert cond.n_params() == 0


def test_mlp_conditioner_invalid_mode_raises():
    """Unknown mode raises a clear error (rather than silently producing garbage)."""
    from cdsbi.conditioners.mlp import MLPConditioner
    try:
        MLPConditioner(input_dim=5, output_dim=1, mode="undefined_mode_xyz")
    except ValueError as e:
        assert "mode" in str(e).lower()
        return
    raise AssertionError("MLPConditioner did not raise on unknown mode")
