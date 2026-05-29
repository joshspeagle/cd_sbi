"""SufficientStatConditioner: oracle X → (s², X̄), frozen, constant log-det."""
from __future__ import annotations

import torch


def test_encode_features_and_order():
    from cdsbi.conditioners.sufficient_stat import SufficientStatConditioner
    cond = SufficientStatConditioner(n_iid=10)
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0]])  # mean=2
    feats, log_det = cond.encode(x)
    assert feats.shape == (1, 2)
    import math
    s2 = float(x.var(dim=-1, unbiased=True))
    assert abs(float(feats[0, 0]) - math.log(s2)) < 1e-5     # index 0 = log s²
    assert abs(float(feats[0, 1]) - 2.0) < 1e-6              # index 1 = X̄
    assert log_det.shape == (1,)


def test_zero_params():
    from cdsbi.conditioners.sufficient_stat import SufficientStatConditioner
    assert SufficientStatConditioner(n_iid=10).n_params() == 0
