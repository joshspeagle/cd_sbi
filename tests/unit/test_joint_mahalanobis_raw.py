"""JointMahalanobis carries raw r² arrays per θ_0 in meta['r_sq']."""
from __future__ import annotations

import numpy as np
import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure


class _Sim:
    d_theta = 2
    def sample_x_given_theta(self, theta_0, n, rng):
        mean = np.asarray(theta_0, dtype=np.float32)
        return torch.tensor(rng.normal(mean, 1.0, size=(n, 2)), dtype=torch.float32)


class _Trained:
    def __init__(self):
        class P(PivotBasedProcedure):
            def __init__(self): pass
            def pivot(self, theta, x): return theta - x
            def confidence_set(self, *a, **k): raise NotImplementedError
        self.procedure = P()


def test_joint_mahalanobis_meta_carries_r_sq():
    from cdsbi.diagnostics.joint_mahalanobis import JointMahalanobis
    grid = [[0.0, 0.0], [2.0, -1.0]]
    res = JointMahalanobis(theta_0_grid=grid, n_per_theta=300)(_Trained(), _Sim())
    assert "r_sq" in res.meta
    assert set(res.meta["r_sq"].keys()) == {"[0.0, 0.0]", "[2.0, -1.0]"}
    arr = np.asarray(res.meta["r_sq"]["[0.0, 0.0]"])
    assert arr.shape == (300,)
    assert (arr >= 0).all()
