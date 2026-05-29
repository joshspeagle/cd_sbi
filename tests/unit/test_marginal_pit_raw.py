"""MarginalPIT carries the raw PIT array in meta['pit_u'] for figure sourcing."""
from __future__ import annotations

import numpy as np
import torch


class _FakeTrained:
    def __init__(self):
        from cdsbi.confidence_set.procedures import PivotBasedProcedure
        class P(PivotBasedProcedure):
            def __init__(self): pass
            def pivot(self, theta, x): return theta - x
            def confidence_set(self, *a, **k): raise NotImplementedError
        self.procedure = P()


def test_marginal_pit_meta_carries_raw_u_1d():
    from cdsbi.diagnostics.marginal_pit import MarginalPIT
    rng = np.random.default_rng(0)
    theta = torch.zeros(500, 1)
    x = torch.tensor(rng.normal(0, 1, size=(500, 1)), dtype=torch.float32)
    res = MarginalPIT()(_FakeTrained(), simulator=None, eval_data=(theta, x))
    assert "pit_u" in res.meta
    u = np.asarray(res.meta["pit_u"])
    assert u.shape == (500,)
    assert (u >= 0).all() and (u <= 1).all()
