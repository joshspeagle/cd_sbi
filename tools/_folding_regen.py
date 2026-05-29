"""Regenerate the E7 catastrophic-folding loss tail from the test recipe.

CDSBIRunner.fit persists only loss_history_tail (last 100 steps); that window
already sits below the entropy floor (the folding signature E7 shows), so we
save it as-is. No runner changes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from cdsbi.flows.joint_umnn import JointUMNNFlow
from cdsbi.conditioners.mlp import MLPConditioner
from cdsbi.losses.nfmle import NFMLELoss
from cdsbi.methods.cd_sbi import CDSBIRunner
from cdsbi.simulators.exp_rate import ExponentialRate

SEED = 2  # the folding-prone seed documented in test_trained_folding.py


def regen_folding(out_dir: str) -> str:
    """Run the R1-only folding recipe; write <out_dir>/folding_tail.parquet."""
    torch.manual_seed(SEED)
    sim = ExponentialRate()
    flow = JointUMNNFlow(hidden=16, theta_ref=sim.theta_range[0])
    conditioner = MLPConditioner(input_dim=sim.n_iid, output_dim=1, mode="frozen_sum")
    runner = CDSBIRunner(flow=flow, conditioner=conditioner, loss=NFMLELoss(),
                         allow_ablation=True)
    config = {"lr": 5e-3, "batch_size": 256, "n_steps": 4000, "n_train": 30000,
              "optimizer": "adamw", "fresh_batch": False}
    trained = runner.fit(simulator=sim, config=config, seed=SEED)
    tail = list(trained.arch_metadata["loss_history_tail"])
    floor = float(sim.entropy_lower_bound())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = config["n_steps"] - len(tail)
    df = pd.DataFrame({"step": range(start, start + len(tail)), "loss": tail,
                       "entropy_floor": floor})
    path = out / "folding_tail.parquet"
    df.to_parquet(path)
    return str(path)
