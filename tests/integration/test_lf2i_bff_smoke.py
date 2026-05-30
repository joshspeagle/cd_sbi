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


def test_lf2i_bff_marginal_is_mc_from_proposal_and_d_aware():
    """The BFF marginal E_{θ~π}[∏ O(X;θ)] (Dalmasso et al. Eq. 10) must be a
    Monte-Carlo average over draws from the PROPOSAL (the simulator's prior) with a
    d-aware sample count — NOT a fixed box-uniform grid. Regression for the d-blind
    N=64 marginal grid that broke high-d coverage (audit 2026-05-30): the faithful
    estimator samples θ_b ~ π via simulator.sample, and the count floors at 128·d."""
    seed_everything(0)
    sim = LocationNormal1D()
    requested_ns = []
    orig_sample = sim.sample

    def spy(n, rng):
        requested_ns.append(int(n))
        return orig_sample(n, rng)

    sim.sample = spy
    MN = 137  # distinctive prime so the marginal draw is identifiable among sample() calls
    runner = LF2IBFFRunner(
        classifier_hidden=8, classifier_depth=2, quantile_hidden=8, quantile_depth=2,
        marginal_n=MN,
    )
    runner.fit(
        simulator=sim,
        config={
            "lr": 1e-3, "batch_size": 32, "n_steps": 5,
            "n_train_stat": 100, "n_train_quantile": 50,
            "alpha_grid": [0.9], "fresh_batch": False,
        },
        seed=0,
    )
    expected = max(MN, 128 * int(sim.d_theta))   # d-aware floor; d=1 → 137
    assert expected in requested_ns, (
        f"BFF marginal must draw {expected} samples from the proposal; "
        f"simulator.sample() was called with {sorted(set(requested_ns))}"
    )


def test_lf2i_bff_marginal_n_back_compat_alias():
    """`marginal_grid_n` stays accepted as a deprecated alias for `marginal_n`;
    an explicit `marginal_n` wins."""
    assert LF2IBFFRunner(marginal_grid_n=99).marginal_n == 99
    assert LF2IBFFRunner(marginal_n=77, marginal_grid_n=99).marginal_n == 77


def test_lf2i_bff_test_stat_deterministic():
    """The BFF statistic must be DETERMINISTIC given (θ, X) — the marginal MC draws
    are fixed at fit time, not re-sampled per call (else calibration is miscalibrated
    against a noisy statistic)."""
    seed_everything(0)
    sim = LocationNormal1D()
    runner = LF2IBFFRunner(
        classifier_hidden=8, classifier_depth=2, quantile_hidden=8, quantile_depth=2,
        marginal_n=64,
    )
    trained = runner.fit(
        simulator=sim,
        config={
            "lr": 1e-3, "batch_size": 32, "n_steps": 5,
            "n_train_stat": 100, "n_train_quantile": 50,
            "alpha_grid": [0.9], "fresh_batch": False,
        },
        seed=0,
    )
    x = torch.tensor([[0.3]])
    t1 = float(trained.procedure.test_statistic(torch.tensor([[0.0]]), x).item())
    t2 = float(trained.procedure.test_statistic(torch.tensor([[0.0]]), x).item())
    assert t1 == t2, f"BFF statistic not deterministic: {t1} vs {t2}"


def test_lf2i_bff_smoke_d2(seed):
    """BFF runs end-to-end on LocationGaussian2D_iid; produces a finite-size 2D set.

    Only checks: training completes; procedure exposes test_statistic/critical_value;
    `boundary_repr` for the multivariate confidence_set lands in v1 Task 13b — for
    now we exercise contains_batch (which already handles d > 1 via ray-bisection
    inside CriticalValueProcedure once Task 13b lands; for this task we only
    verify the training pipeline survives).
    """
    from cdsbi.simulators.location_gauss_2d_iid import LocationGaussian2D_iid
    seed_everything(seed)
    sim = LocationGaussian2D_iid()
    runner = LF2IBFFRunner(
        classifier_hidden=16, classifier_depth=2,
        quantile_hidden=8, quantile_depth=2,
        marginal_grid_n=32,
    )
    trained = runner.fit(
        simulator=sim,
        config={
            "lr": 1e-3, "batch_size": 32, "n_steps": 50,
            "n_train_stat": 300, "n_train_quantile": 150,
            "alpha_grid": [0.5, 0.9], "fresh_batch": False,
        },
        seed=seed,
    )
    assert trained.procedure is not None
    assert trained.procedure.d_theta == 2
    # Test statistic evaluated at a (B=4) × (d=2) batch returns shape (4,).
    theta_q = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [2.0, -2.0]])
    x_obs = torch.tensor([[0.0, 0.0]])  # (1, d_x); broadcast in test_stat_fn
    t_vals = trained.procedure.test_statistic(theta_q, x_obs)
    assert t_vals.shape == (4,)
    # Closer-to-X θ should give smaller T (BFF is Wilks-direction: large = bad fit).
    assert float(t_vals[0]) < float(t_vals[3]), (
        f"BFF at θ=(0,0) X=(0,0) ({t_vals[0]}) should be < at θ=(2,-2) ({t_vals[3]})"
    )
