"""EnergyCDSBIRunner: Stage-B Arm II-A. End-to-end summary+pivot trained under the
calibration-invariant EnergyCalibrationLoss with grouped-by-θ₀ batches. No Jacobian
term ⇒ no collapse cheat; the summary co-adapts to emit calibratable coordinates.
"""
from __future__ import annotations

import time

import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import TrainedModel
from cdsbi.reproducibility.seeding import seed_everything


class EnergyCDSBIRunner:
    def __init__(self, flow, conditioner, loss, allow_ablation: bool = False, device: str = "auto"):
        self.flow = flow
        self.conditioner = conditioner
        self.loss = loss
        self.allow_ablation = allow_ablation
        self.device = get_device(device)
        if not allow_ablation:
            self.loss.check_guarantees(flow)

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        rngs = seed_everything(seed)
        dev = self.device
        self.flow.to(dev); self.flow.train()
        cond_is_module = isinstance(self.conditioner, torch.nn.Module)
        if cond_is_module:
            self.conditioner.to(dev); self.conditioner.train()

        grp = config["group"]
        B = int(grp["n_theta_per_batch"]); m = int(grp["group_size"])
        n_ref = int(grp.get("n_ref", self.loss.n_ref))
        n_steps = int(config["n_steps"]); lr = float(config["lr"])
        grad_clip = float(config.get("grad_clip_norm", 5.0))

        params = list(self.flow.parameters())
        if cond_is_module:
            params += [p for p in self.conditioner.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)

        d = int(simulator.d_theta)
        losses = []
        t0 = time.time()
        for step in range(n_steps):
            theta0 = simulator._draw_theta(B, rngs.train)               # (B, d) np
            thetas, xs = [], []
            for b in range(B):
                x_b = simulator.sample_x_given_theta(theta0[b], m, rngs.train)  # (m, n_iid)
                xs.append(x_b)
                thetas.append(torch.as_tensor(theta0[b], dtype=x_b.dtype).expand(m, d))
            x = torch.cat(xs, dim=0).to(dev)                            # (B*m, n_iid)
            theta = torch.cat(thetas, dim=0).to(dev)                    # (B*m, d)
            context, _ = self.conditioner.encode(x)
            r, _ = self.flow.forward(theta, context=context)            # (B*m, d)
            r_groups = r.view(B, m, d)
            gen = torch.Generator(device=r.device).manual_seed(seed * 100003 + step)
            loss_val = self.loss.score(r_groups, generator=gen)
            opt.zero_grad(); loss_val.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            opt.step()
            losses.append(loss_val.item())
        wall = time.time() - t0

        self.flow.eval()
        if cond_is_module:
            self.conditioner.eval()
        flow = self.flow; conditioner = self.conditioner; device = dev

        def pivot_fn(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            theta = theta.to(device); x = x.to(device)
            context, _ = conditioner.encode(x)
            r, _ = flow.forward(theta, context=context)
            return r

        def encode_fn(x: torch.Tensor) -> torch.Tensor:
            x = x.to(device)
            with torch.no_grad():
                feats, _ = conditioner.encode(x)
            return feats

        procedure = PivotBasedProcedure(pivot_fn=pivot_fn, d_theta=d, encode_fn=encode_fn)
        arch_meta = {
            "flow_class": type(self.flow).__name__,
            "loss_class": type(self.loss).__name__,
            "loss_history_tail": losses[-min(100, len(losses)):],
            "conditioner_class": type(self.conditioner).__name__,
            "conditioner_params": self.conditioner.n_params(),
            "group": {"n_theta_per_batch": B, "group_size": m, "n_ref": n_ref},
        }
        return TrainedModel(procedure=procedure, state_dict={}, final_loss=float(losses[-1]),
                            n_steps=n_steps, wall_clock_sec=wall, arch_metadata=arch_meta)

    def n_params(self) -> dict:
        backbone = sum(p.numel() for p in self.flow.parameters())
        head = self.conditioner.n_params() if hasattr(self.conditioner, "n_params") else 0
        return {"backbone": backbone, "head": head, "calibration_stage": 0,
                "total": backbone + head, "kind": "flow"}
