"""SetSize: width is monotone in α; oracle pivot matches the analytic z*."""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import chi2

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.set_size import SetSize
from cdsbi.methods.base import TrainedModel
from cdsbi.reproducibility.seeding import seed_everything
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def _oracle_trained():
    # r*(θ; X) = θ - X is the canonical pivot for LocationNormal1D.
    # Confidence interval at level α: r² ≤ χ²_{1, α} ⇒ |θ - X| ≤ √χ²_{1, α}
    # Width = 2 √χ²_{1, α}, independent of X (and θ_0).
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: th - x, d_theta=1)
    return TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0
    )


def test_set_size_oracle_matches_analytic_width(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    trained = _oracle_trained()
    diag = SetSize(theta_0_grid=[0.0], alpha_grid=[0.5, 0.9, 0.95], n_per_theta=500)
    result = diag(trained, sim, eval_data=None)
    df = result.value
    for _, row in df.iterrows():
        analytic = 2.0 * np.sqrt(chi2.ppf(row["alpha"], df=1))
        assert abs(row["mean_width"] - analytic) < 0.05, (
            f"alpha={row['alpha']}: mean_width={row['mean_width']:.4f}, "
            f"analytic={analytic:.4f}"
        )


def test_set_size_monotone_in_alpha(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    trained = _oracle_trained()
    alphas = [0.5, 0.68, 0.9, 0.95]
    diag = SetSize(theta_0_grid=[0.0], alpha_grid=alphas, n_per_theta=300)
    df = diag(trained, sim, eval_data=None).value.sort_values("alpha").reset_index(drop=True)
    widths = df["mean_width"].to_numpy()
    assert np.all(np.diff(widths) >= 0), f"widths not monotone in alpha: {widths.tolist()}"


def test_set_size_catches_wide_set(seed):
    """A degenerate procedure that always returns the full prior range gets
    the widest possible set; SetSize should expose this."""
    from cdsbi.confidence_set.datatypes import ConfidenceSet

    class WideProcedure:
        d_theta = 1

        def confidence_set(self, x_obs, alpha):
            # Returns the full [-10, 10] range regardless of α
            return ConfidenceSet(
                contains=lambda th: -10.0 <= th <= 10.0,
                boundary_repr=torch.tensor([-10.0, 10.0]),
                alpha=alpha,
            )

    seed_everything(seed)
    sim = LocationNormal1D()
    trained = TrainedModel(
        procedure=WideProcedure(), state_dict={}, final_loss=0.0, n_steps=0,
        wall_clock_sec=0.0,
    )
    diag = SetSize(theta_0_grid=[0.0], alpha_grid=[0.9], n_per_theta=50)
    df = diag(trained, sim, eval_data=None).value
    assert abs(df["mean_width"].iloc[0] - 20.0) < 1e-6


def test_set_size_2d_oracle_matches_analytic_width(seed):
    """d=2 oracle pivot r* = θ - X gives a circular α-set of radius √χ²_{2,α}.
    With the diameter convention (matches d=1's right-left), expect 2√χ²_{2,α}.
    """
    import numpy as np
    from scipy.stats import chi2
    from cdsbi.simulators.location_gauss_2d_iid import LocationGaussian2D_iid
    from cdsbi.confidence_set.procedures import PivotBasedProcedure
    from cdsbi.methods.base import TrainedModel
    from cdsbi.reproducibility.seeding import seed_everything
    seed_everything(seed)
    sim = LocationGaussian2D_iid()
    proc = PivotBasedProcedure(
        pivot_fn=lambda th, x: th - x, d_theta=2, theta_range=(-7.0, 7.0),
    )
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0,
    )
    diag = SetSize(theta_0_grid=[[0.0, 0.0]], alpha_grid=[0.5, 0.9, 0.95], n_per_theta=100)
    df = diag(trained, sim, eval_data=None).value
    for _, row in df.iterrows():
        analytic_diameter = 2.0 * np.sqrt(chi2.ppf(row["alpha"], df=2))
        # Boundary samples + ray bisection tolerance combine to a few-% error.
        assert abs(row["mean_width"] - analytic_diameter) / analytic_diameter < 0.10, (
            f"alpha={row['alpha']}: mean_width={row['mean_width']:.4f}, "
            f"analytic diameter={analytic_diameter:.4f}"
        )
