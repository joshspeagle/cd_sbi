"""TwoStageCDSBIRunner: (1) train a MomentRegressionConditioner by L² regression to
m(θ) and FREEZE it (regression to a function of θ structurally blocks the entropy-
floor cheat — validity-safety; efficiency is asymptotic/BvM, measured by
FisherRecovery, not structural); (2) calibrate the pivot by NF-MLE on the frozen
summary. Inference uses r(θ; frozen-summary(X)) ∈ ℝ^{d_θ}."""
from __future__ import annotations

import time

import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.device import get_device
from cdsbi.flows.base import Guarantee
from cdsbi.losses.base import MonotonicityMismatchError
from cdsbi.losses.nfmle import NFMLELoss
from cdsbi.methods.base import TrainedModel
from cdsbi.reproducibility.seeding import seed_everything


class TwoStageCDSBIRunner:
    def __init__(self, flow, conditioner, loss=None, allow_ablation: bool = False,
                 device: str = "auto"):
        self.flow = flow
        self.conditioner = conditioner       # MomentRegressionConditioner (has regression_target)
        self.loss = loss or NFMLELoss()
        self.allow_ablation = allow_ablation
        self.device = get_device(device)
        if not allow_ablation:
            self.loss.check_guarantees(flow)   # NFMLELoss requires {R1,R2}

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        rngs = seed_everything(seed)
        dev = self.device
        self.flow.to(dev); self.conditioner.to(dev)
        bs = int(config["batch_size"]); lr = float(config["lr"])
        # Stage split: explicit stage1_steps/stage2_steps (tests) override; else split
        # the budget's n_steps by stage1_frac (harness — _recipe_dict passes these
        # through after Task 5's additive edit; default frac 0.4).
        if config.get("stage1_steps") is not None and config.get("stage2_steps") is not None:
            s1, s2 = int(config["stage1_steps"]), int(config["stage2_steps"])
        else:
            n_steps = int(config["n_steps"])
            frac = float(config.get("stage1_frac", 0.4) or 0.4)
            s1 = int(frac * n_steps); s2 = n_steps - s1

        # ---- Stage 1: regression to m(θ), then FREEZE ----
        self.conditioner.train()
        opt1 = torch.optim.Adam(self.conditioner.parameters(), lr=lr)
        for _ in range(s1):
            theta, x = simulator.sample(bs, rngs.train)
            theta, x = theta.to(dev), x.to(dev)
            pred, _ = self.conditioner.encode(x)
            target = self.conditioner.regression_target(theta)
            loss1 = ((pred - target) ** 2).mean()
            opt1.zero_grad(); loss1.backward(); opt1.step()
        for p in self.conditioner.parameters():
            p.requires_grad_(False)
        self.conditioner.eval()

        # ---- Stage 2: NF-MLE pivot on the frozen summary (Task 4) ----
        losses, wall = self._fit_pivot(simulator, config, rngs, dev, s2)

        self.flow.eval()
        flow, conditioner, device = self.flow, self.conditioner, dev

        def pivot_fn(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            theta, x = theta.to(device), x.to(device)
            feats, _ = conditioner.encode(x)
            r, _ = flow.forward(theta, context=feats)
            return r

        def encode_fn(x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                feats, _ = conditioner.encode(x.to(device))
            return feats

        procedure = PivotBasedProcedure(pivot_fn=pivot_fn, d_theta=int(simulator.d_theta),
                                        encode_fn=encode_fn)
        arch_meta = {
            "flow_class": type(self.flow).__name__,
            "loss_class": "NFMLELoss",
            "conditioner_class": type(self.conditioner).__name__,
            "regression_target": getattr(self.conditioner, "target", None),
            "loss_history_tail": losses[-min(100, len(losses)):],
        }
        return TrainedModel(procedure=procedure, state_dict={},
                            final_loss=float(losses[-1]) if losses else float("nan"),
                            n_steps=s1 + s2, wall_clock_sec=wall, arch_metadata=arch_meta)

    def _fit_pivot(self, simulator, config, rngs, dev, n_steps):
        return [], 0.0   # stubbed in this task; implemented in Task 4

    def n_params(self) -> dict:
        backbone = sum(p.numel() for p in self.flow.parameters())
        head = self.conditioner.n_params()
        return {"backbone": backbone, "head": head, "total": backbone + head, "kind": "flow"}
