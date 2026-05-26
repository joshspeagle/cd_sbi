import math
import torch
from cdsbi.confidence_set.datatypes import ConfidenceSet
from cdsbi.confidence_set.procedures import PivotBasedProcedure


def test_confidence_set_contains_query():
    cs = ConfidenceSet(
        contains=lambda th: bool((-1.0 <= th <= 1.0)),
        boundary_repr=torch.tensor([-1.0, 1.0]),
        alpha=0.95,
    )
    assert cs.contains(0.0)
    assert not cs.contains(2.0)


def test_pivot_based_procedure_1d_chi_sq_inversion(seed):
    """For r(θ, X) = θ − X with X_obs = 0, the 1D α-confidence set is
       {θ : |θ|² ≤ χ²_{1, α}} = [−z_α, z_α] where z_α = sqrt(χ²_{1, α})."""
    torch.manual_seed(seed)
    proc = PivotBasedProcedure(pivot_fn=lambda theta, x: theta - x, d_theta=1)
    x_obs = torch.tensor([[0.0]])
    cs = proc.confidence_set(x_obs, alpha=0.95)
    # χ²_{1, 0.95} = 3.841 ⇒ z = 1.960
    z = math.sqrt(3.8414588)
    assert abs(cs.boundary_repr[0].item() + z) < 0.01
    assert abs(cs.boundary_repr[1].item() - z) < 0.01
