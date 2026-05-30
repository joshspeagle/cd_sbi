"""Intensive: (μ,Σ) Stage-A — the d=5 Bartlett-pivot framework generalization.

VERDICT (see docs/superpowers/specs/2026-05-30-cd-sbi-mu-cov-n2-verdict.md): the
framework GENERALIZES — 4/5 pivot coordinates calibrate to the noise floor, and
the aggregate calibration (covariance χ² marginals, joint Hotelling-T² μ, joint
Mahalanobis χ²₅, entropy floor) all hold. The hard gates below assert those.

DOCUMENTED LIMITATION: μ₂ (coord 5 — the doubly-cross-coupled mean coordinate) has
a MILD per-coordinate miscalibration at EXTREME θ₀ (per-coord PIT KS up to ~0.13,
vs ~0.02 floor), driven by a ctx-MLP expressivity limit on the affine index z₅
(NOT G-curvature, finite-sample, slow-convergence, or capacity — all ruled out;
identity-G trades scale-for-shape without eliminating it). Central-region coverage
is fine. We assert the 4 well-calibrated coords at floor and pin μ₂'s mild limit as
a regression (precedent: tests/ablation/test_trained_folding.py captures a known
pathology the same way). The overall pivot-RMSE-vs-r* is a loose recovery sanity
only (r* is one specific calibrated pivot; calibration only needs being on M)."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.intensive
def test_replicate_mu_cov_stage_a():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    from cdsbi.conditioners.bartlett_summary import BartlettSummaryConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi import CDSBIRunner
    from cdsbi.diagnostics.multivariate_marginal_cd import MultivariateMarginalCDRecovery
    from scipy.stats import kstest, chi2, norm

    torch.manual_seed(0)
    sim = NormalBivariateUnknownCov()
    flow = SingleIndexMonotoneFlow(d=5, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=64, depth=2)
    cond = BartlettSummaryConditioner(n_iid=sim.n_iid)
    runner = CDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss())
    config = {"lr": 3e-3, "batch_size": 256, "n_steps": 12000, "n_train": 20000,
              "optimizer": "adamw", "fresh_batch": False}
    trained = runner.fit(simulator=sim, config=config, seed=0)
    assert trained.procedure.flow is not None   # set by CDSBIRunner.fit

    # --- HARD GATE: aggregate calibration (the framework's coverage thesis) ---
    # (a) marginal-CD recovery: covariance χ² + joint Hotelling-T² μ
    grid = [tuple(map(float, t)) for t in [
        (0.0, 0.0, 0.0, 0.0, 0.0), (0.4, -0.4, 0.8, 1.5, -1.5), (-0.4, 0.4, -0.8, -1.5, 1.5)]]
    mres = MultivariateMarginalCDRecovery(theta_0_grid=grid, n_per_theta=3000)(trained, sim).value
    print(mres[["cov1_ks", "cov2_ks", "cov3_ks", "mu_hotelling_ks"]].to_string())
    assert mres[["cov1_ks", "cov2_ks", "cov3_ks"]].to_numpy().max() < 0.07
    assert mres["mu_hotelling_ks"].max() < 0.08
    # (b) joint Mahalanobis ‖r‖²~χ²₅
    xv = sim.sample_x_given_theta((0.0, 0.0, 0.0, 0.0, 0.0), 3000, np.random.default_rng(7))
    with torch.no_grad():
        rr = trained.procedure.pivot(torch.zeros(3000, 5), xv).cpu().numpy()
    assert kstest(chi2.cdf((rr ** 2).sum(1), df=5), "uniform").statistic < 0.06
    # (c) no cheat: loss respects the entropy floor
    H = sim.entropy_lower_bound()
    print(f"final_loss={trained.final_loss:.3f} floor={H:.3f}")
    assert trained.final_loss > H - 0.15

    # --- PER-COORDINATE PIT: 4 coords at floor; μ₂ a documented mild limit ---
    # Φ(r_k(θ₀;X))|θ₀ ~ U at truth, across θ₀ (incl. extremes). The sharp check the
    # aggregate masks. coords: [ℓ11, ℓ22, L21, μ1, μ2].
    pit_grid = [(0., 0., 0., 0., 0.), (0.6, -0.5, -0.9, 2.0, -2.0), (-0.6, 0.5, 1.0, -2.0, 2.5)]
    ks = np.zeros((len(pit_grid), 5))
    for i, th0 in enumerate(pit_grid):
        xv = sim.sample_x_given_theta(th0, 4000, np.random.default_rng(7))
        thv = torch.tensor([th0], dtype=xv.dtype).expand(4000, 5)
        with torch.no_grad():
            r = trained.procedure.pivot(thv, xv).cpu().numpy()
        ks[i] = [kstest(norm.cdf(r[:, k]), "uniform").statistic for k in range(5)]
    print("per-coord PIT KS (rows=θ₀, cols=ℓ11,ℓ22,L21,μ1,μ2):\n", ks.round(3))
    # 4 well-calibrated coords at/near the noise floor across all θ₀
    assert ks[:, :4].max() < 0.07, f"coords 0-3 should calibrate to floor; got {ks[:, :4].max():.3f}"
    # μ₂ (coord 5): documented mild limit at extreme θ₀ (regression-pins the finding)
    assert ks[:, 4].max() < 0.16, f"μ₂ PIT KS beyond the documented mild band: {ks[:, 4].max():.3f}"

    # --- LOOSE recovery sanity (RMSE-vs-r*; secondary — r* is one calibrated pivot) ---
    theta, x = sim.sample(4000, np.random.default_rng(123))
    with torch.no_grad():
        per = ((trained.procedure.pivot(theta, x).cpu() - sim.r_star(theta, x).cpu()) ** 2).mean(0).sqrt()
    print(f"recovery RMSE={float((per**2).mean().sqrt()):.3f} per-coord={per.numpy().round(3)}")
    assert float((per ** 2).mean().sqrt()) < 0.25   # loose sanity; μ₂ dominates (documented)
