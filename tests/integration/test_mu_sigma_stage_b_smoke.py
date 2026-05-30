"""Regression capture: the naive device-1 learned summary (DeepSets, log_det=0 +
normalization) CHEATS the NF-MLE loss — final loss sinks below the entropy floor
and σ²-information collapses. This documents the Stage-B pathology that motivates
the device bake-off (see specs/2026-05-29-cd-sbi-stage-b-device-bakeoff-design.md
§1). It asserts the cheat is PRESENT; the fixes are the bake-off arms (M3.1–M3.3).
Precedent: tests/ablation/test_trained_folding.py captures the §3.5 folding failure."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.intensive
def test_device1_learned_summary_cheats_below_floor():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi import CDSBIRunner
    from scipy.stats import spearmanr

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

    H = sim.entropy_lower_bound()
    # (1) the cheat: final NF-MLE loss sinks well below the conditional-entropy floor
    assert trained.final_loss < H - 1.0, (
        f"expected device-1 cheat (loss ≪ floor {H:.2f}); got {trained.final_loss:.3f}"
    )

    # (2) the harm: σ²-information collapses (X̄ survives)
    rng = np.random.default_rng(123)
    _, x = sim.sample(4000, rng)
    cond.eval()
    with torch.no_grad():
        feats = cond.encode(x.to(runner.device))[0].cpu().numpy()
    oracle = sim.oracle_summary(x).numpy()
    sp_logs2 = max(abs(spearmanr(oracle[:, 0], feats[:, j]).statistic) for j in range(2))
    sp_xbar = max(abs(spearmanr(oracle[:, 1], feats[:, j]).statistic) for j in range(2))
    print(f"device-1 cheat: final_loss={trained.final_loss:.3f} floor={H:.3f} "
          f"| σ²-Spearman={sp_logs2:.3f} X̄-Spearman={sp_xbar:.3f}")
    assert sp_logs2 < 0.9, f"expected σ²-collapse; got Spearman {sp_logs2:.2f}"
    assert sp_xbar > 0.9, f"expected X̄ to survive; got Spearman {sp_xbar:.2f}"
