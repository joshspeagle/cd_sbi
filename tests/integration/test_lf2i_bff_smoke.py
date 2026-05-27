"""LF2I-BFF smoke: trains and produces a finite-width confidence set."""
from __future__ import annotations

import torch

from cdsbi.methods.lf2i_bff import LF2IBFFRunner
from cdsbi.reproducibility.seeding import seed_everything
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def test_lf2i_bff_smoke(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    runner = LF2IBFFRunner(
        classifier_hidden=16,
        classifier_depth=2,
        quantile_hidden=8,
        quantile_depth=2,
        marginal_grid_n=32,
    )
    trained = runner.fit(
        simulator=sim,
        config={
            "lr": 1e-3,
            "batch_size": 32,
            "n_steps": 50,
            "n_train_stat": 300,
            "n_train_quantile": 150,
            "n_epochs_quantile": 50,
            "alpha_grid": [0.5, 0.9],
            "fresh_batch": False,
        },
        seed=seed,
    )
    assert trained.procedure is not None
    cs = trained.procedure.confidence_set(torch.tensor([[0.0]]), alpha=0.9)
    assert cs.boundary_repr.shape == (2,)
    left = float(cs.boundary_repr[0])
    right = float(cs.boundary_repr[1])
    assert right > left, f"empty / inverted set: [{left}, {right}]"


def test_lf2i_bff_n_params_independent_of_alpha_grid_len():
    runner = LF2IBFFRunner(
        classifier_hidden=16, classifier_depth=2, quantile_hidden=8, quantile_depth=2,
    )
    n3 = runner.n_params(d_theta=1, d_x=1, alpha_grid_len=3)["total"]
    n4 = runner.n_params(d_theta=1, d_x=1, alpha_grid_len=4)["total"]
    # Only the per-α heads grow with alpha_grid_len: (H + 1) per quantile.
    assert n4 - n3 == 9  # H=8 → H+1=9


def test_lf2i_bff_test_stat_wilks_direction_at_oracle_classifier():
    """With a perfect classifier on LocationNormal1D and a uniform prior, the
    BFF statistic T_BFF(θ; X) = log marginal − log p(X|θ) should be SMALL
    at θ = data-generating θ and LARGE at far-away θ. Verify the sign."""
    # We don't actually use a perfect classifier — just train briefly to get
    # a meaningful direction; verify T(0; 0) < T(5; 0) (right θ vs wrong θ).
    seed_everything(0)
    sim = LocationNormal1D()
    runner = LF2IBFFRunner(
        classifier_hidden=32, classifier_depth=2, quantile_hidden=8, quantile_depth=2,
        marginal_grid_n=32,
    )
    trained = runner.fit(
        simulator=sim,
        config={
            "lr": 1e-3, "batch_size": 32, "n_steps": 300,
            "n_train_stat": 1000, "n_train_quantile": 200, "n_epochs_quantile": 50,
            "alpha_grid": [0.9], "fresh_batch": False,
        },
        seed=0,
    )
    proc = trained.procedure
    x_obs = torch.tensor([[0.0]])
    t_right = float(proc.test_statistic(torch.tensor([[0.0]]), x_obs).item())
    t_wrong = float(proc.test_statistic(torch.tensor([[5.0]]), x_obs).item())
    assert t_right < t_wrong, (
        f"BFF direction wrong: T(0; X=0)={t_right:.3f}, T(5; X=0)={t_wrong:.3f}"
    )
