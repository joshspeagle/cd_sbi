import math
import torch
from cdsbi.confidence_set.procedures import PivotBasedProcedure


def _oracle(theta, x):
    """r* = θ - X. Marginal pivot for X ~ N(θ, 1)."""
    return theta - x


def test_contains_batch_matches_single(seed):
    torch.manual_seed(seed)
    proc = PivotBasedProcedure(pivot_fn=_oracle, d_theta=1)
    theta_0 = 0.5
    x_batch = torch.linspace(-3, 3, 50).unsqueeze(-1)
    inside_batch = proc.contains_batch(theta_0, x_batch, alpha=0.9)
    inside_single = []
    for i in range(x_batch.shape[0]):
        cs = proc.confidence_set(x_batch[i:i+1], alpha=0.9)
        inside_single.append(cs.contains(theta_0))
    inside_single_t = torch.tensor(inside_single)
    assert (inside_batch == inside_single_t).all()


def test_confidence_set_batch_matches_single(seed):
    torch.manual_seed(seed)
    proc = PivotBasedProcedure(pivot_fn=_oracle, d_theta=1)
    x_batch = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    left_b, right_b = proc.confidence_set_batch(x_batch, alpha=0.9)
    for i in range(x_batch.shape[0]):
        cs = proc.confidence_set(x_batch[i:i+1], alpha=0.9)
        l_s, r_s = cs.boundary_repr.tolist()
        assert abs(left_b[i].item() - l_s) < 1e-3
        assert abs(right_b[i].item() - r_s) < 1e-3


def test_contains_batch_oracle_chi_sq(seed):
    """For r* = θ - X, x_obs = 0: θ_0=0 is in C_0.95 iff |r|² ≤ χ²_{1, 0.95} = 3.84 ⇒ |θ_0| ≤ 1.96."""
    torch.manual_seed(seed)
    proc = PivotBasedProcedure(pivot_fn=_oracle, d_theta=1)
    x = torch.zeros(10, 1)
    # θ_0 = 0 is always inside
    assert proc.contains_batch(0.0, x, alpha=0.95).all()
    # θ_0 = 3 (well outside ±1.96) is always outside
    assert not proc.contains_batch(3.0, x, alpha=0.95).any()


def test_confidence_set_batch_oracle_chi_sq(seed):
    """For r* = θ - X with X_obs = 0, the 0.95-set is [-1.96, +1.96]."""
    torch.manual_seed(seed)
    proc = PivotBasedProcedure(pivot_fn=_oracle, d_theta=1)
    x_batch = torch.zeros(4, 1)
    left, right = proc.confidence_set_batch(x_batch, alpha=0.95)
    z = math.sqrt(3.8414588)
    for i in range(4):
        assert abs(left[i].item() + z) < 0.01
        assert abs(right[i].item() - z) < 0.01
