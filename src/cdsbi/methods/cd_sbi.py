"""CDSBIRunner: trains a monotone pivot flow under NF-MLE."""
from __future__ import annotations

import time
from typing import Optional

import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import Runner, TrainedModel


class CDSBIRunner(Runner):
    def __init__(self, flow, conditioner, loss, allow_ablation: bool = False, device: str = "auto"):
        self.flow = flow
        self.conditioner = conditioner
        self.loss = loss
        self.allow_ablation = allow_ablation
        self.device = get_device(device)
        if not allow_ablation:
            self.loss.check_guarantees(flow)

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        from cdsbi.reproducibility.seeding import seed_everything
        rngs = seed_everything(seed)

        self.flow.to(self.device)
        opt = torch.optim.Adam(self.flow.parameters(), lr=config["lr"])

        # Pre-sample full training set
        theta_all, x_all = simulator.sample(config["n_train"], rngs.train)
        theta_all, x_all = theta_all.to(self.device), x_all.to(self.device)

        bs = config["batch_size"]
        n_steps = config["n_steps"]
        n_train = theta_all.shape[0]
        losses = []
        t0 = time.time()
        for step in range(n_steps):
            idx = torch.randint(0, n_train, (bs,), generator=torch.Generator(device="cpu"))
            theta_b, x_b = theta_all[idx], x_all[idx]
            context, log_det_contrib = self.conditioner.encode(x_b)
            r, log_det_flow = self.flow.forward(theta_b, context=context)
            log_det_total = log_det_flow + log_det_contrib
            loss_val = self.loss(r=r, log_det_jac_input=log_det_total)
            opt.zero_grad()
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=5.0)
            opt.step()
            losses.append(loss_val.item())
        wall = time.time() - t0

        # Build PivotBasedProcedure with a closure over the trained flow + conditioner
        flow = self.flow
        conditioner = self.conditioner
        device = self.device

        def pivot_fn(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            theta = theta.to(device)
            x = x.to(device)
            context, _ = conditioner.encode(x)
            r, _ = flow.forward(theta, context=context)
            return r

        procedure = PivotBasedProcedure(pivot_fn=pivot_fn, d_theta=simulator.d_theta)

        return TrainedModel(
            procedure=procedure,
            state_dict=self.flow.state_dict(),
            final_loss=losses[-1],
            n_steps=n_steps,
            wall_clock_sec=wall,
            arch_metadata={
                "flow_class": type(self.flow).__name__,
                "loss_class": type(self.loss).__name__,
                "loss_history_tail": losses[-min(100, len(losses)) :],
            },
        )

    def n_params(self) -> dict:
        backbone = self.flow.a.n_params() + self.flow.b.n_params()
        head = self.flow.n_params() - backbone  # the α scalars
        return {
            "backbone": backbone,
            "head": head,
            "calibration_stage": 0,
            "total": self.flow.n_params(),
            "kind": "flow",
        }
