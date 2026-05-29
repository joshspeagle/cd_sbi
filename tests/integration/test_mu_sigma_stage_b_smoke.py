"""Stage-B smoke: CDSBI with a LEARNED DeepSets summary trains + recovers a
reparameterization of the sufficient statistic (log s², X̄). Loose bands — the
full Stage-B calibration/floor verdict is M3."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.intensive
def test_stage_b_learned_summary_trains_and_recovers_sufficiency():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi import CDSBIRunner

    torch.manual_seed(0)
    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=32, depth=2)
    cond = DeepSetsConditioner(n_iid=sim.n_iid, d_out=2, hidden=32, depth=2)
    runner = CDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss())
    config = {"lr": 3e-3, "batch_size": 256, "n_steps": 6000, "n_train": 10000,
              "optimizer": "adamw", "lr_schedule": "warmup_cosine",
              "warmup_steps": 300, "lr_min_ratio": 0.01, "fresh_batch": False}
    trained = runner.fit(simulator=sim, config=config, seed=0)

    # (a) trained: finite loss
    assert np.isfinite(trained.final_loss)
    print(f"final_loss = {trained.final_loss:.4f}  (entropy floor H ≈ {sim.entropy_lower_bound():.4f})")

    # (b) sufficiency recovery via SPEARMAN (monotone-invariant); Pearson printed for diagnosis
    rng = np.random.default_rng(123)
    theta, x = sim.sample(4000, rng)
    cond.eval()
    with torch.no_grad():
        feats, _ = cond.encode(x.to(runner.device))
    feats = feats.cpu().numpy()
    xbar, s2 = sim._suff_stats(x)
    suff = np.column_stack([np.log(s2.squeeze(-1).numpy()), xbar.squeeze(-1).numpy()])  # (n,2)
    from scipy.stats import spearmanr, pearsonr
    def best_abscorr(target, F, fn):
        return max(abs(fn(target, F[:, j])[0]) for j in range(F.shape[1]))
    sp_logs2 = best_abscorr(suff[:, 0], feats, spearmanr)
    sp_xbar = best_abscorr(suff[:, 1], feats, spearmanr)
    pe_logs2 = best_abscorr(suff[:, 0], feats, pearsonr)
    pe_xbar = best_abscorr(suff[:, 1], feats, pearsonr)
    print(f"sufficiency (Spearman): log s²={sp_logs2:.3f}  X̄={sp_xbar:.3f}  "
          f"| (Pearson): log s²={pe_logs2:.3f}  X̄={pe_xbar:.3f}")
    assert sp_logs2 > 0.9, f"learned summary lost σ²-information (Spearman {sp_logs2:.2f})"
    assert sp_xbar > 0.9, f"learned summary lost μ-information (Spearman {sp_xbar:.2f})"

    # (c) calibration sanity: joint Mahalanobis PIT at a central θ₀ ~ χ²₂
    from scipy.stats import kstest, chi2
    theta_0 = (0.0, 0.0)
    xv = sim.sample_x_given_theta(theta_0, 3000, np.random.default_rng(7))
    th = torch.tensor([[0.0, 0.0]], dtype=xv.dtype).expand(xv.shape[0], -1)
    with torch.no_grad():
        r = trained.procedure.pivot(th, xv).cpu().numpy()
    pit = chi2.cdf((r ** 2).sum(1), df=2)
    ks = kstest(pit, "uniform").statistic
    print(f"joint Mahalanobis KS (learned summary) = {ks:.3f}")
    assert ks < 0.10, f"learned-summary joint calibration KS {ks:.3f} too high (loose Stage-B sanity)"
