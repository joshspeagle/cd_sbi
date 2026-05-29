"""JacobianRecovery carries J_emp_mean and J_true matrices in meta."""
from __future__ import annotations

import numpy as np
import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure


class _Sim:
    d_theta = 2
    def __init__(self):
        self.L_inv = np.array([[1.0, 0.0], [-0.4, 0.9]], dtype=np.float32)
    def r_star_jacobian(self):
        return torch.tensor(self.L_inv)
    def sample(self, n, rng):
        theta = torch.tensor(rng.normal(0, 1, size=(n, 2)), dtype=torch.float32)
        x = torch.tensor(rng.normal(0, 1, size=(n, 2)), dtype=torch.float32)
        return theta, x


class _Trained:
    """pivot r = (θ - x) @ L_inv.T: linear, so ∂r/∂θ = L_inv exactly."""
    def __init__(self, L_inv):
        L = torch.tensor(L_inv)
        class P(PivotBasedProcedure):
            def __init__(self): pass
            def pivot(self, theta, x):
                return (theta - x) @ L.T
            def confidence_set(self, *a, **k): raise NotImplementedError
        self.procedure = P()


def test_jacobian_recovery_meta_carries_matrices():
    from cdsbi.diagnostics.jacobian_recovery import JacobianRecovery
    sim = _Sim()
    res = JacobianRecovery(n_points=50)(_Trained(sim.L_inv), sim)
    assert "J_emp_mean" in res.meta and "J_true" in res.meta
    J_emp = np.asarray(res.meta["J_emp_mean"])
    J_true = np.asarray(res.meta["J_true"])
    assert J_emp.shape == (2, 2) and J_true.shape == (2, 2)
    np.testing.assert_allclose(J_true, sim.L_inv, atol=1e-5)
    np.testing.assert_allclose(J_emp, sim.L_inv, atol=1e-4)
