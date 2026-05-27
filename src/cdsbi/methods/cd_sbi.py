"""CDSBIRunner: trains a monotone pivot flow under NF-MLE."""
from __future__ import annotations

import math
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
        # Put the flow into train mode so autograd-based Jacobian flows (e.g.
        # JointUMNNFlow, JointUMNN1DFlow) can build the second-order graph via
        # create_graph=self.training during training.
        self.flow.train()

        # --- Optimizer factory ---
        opt_name = config.get("optimizer", "adam")
        lr = config["lr"]
        weight_decay = config.get("weight_decay", 0.0)
        betas = tuple(config.get("betas", [0.9, 0.999]))
        if opt_name == "adam":
            opt = torch.optim.Adam(self.flow.parameters(), lr=lr, betas=betas)
        elif opt_name == "adamw":
            opt = torch.optim.AdamW(self.flow.parameters(), lr=lr, betas=betas,
                                    weight_decay=weight_decay)
        elif opt_name == "sgd":
            momentum = config.get("momentum", 0.9)
            opt = torch.optim.SGD(self.flow.parameters(), lr=lr,
                                  momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name!r}")

        # --- LR scheduler factory ---
        sched_name = config.get("lr_schedule", "constant")
        n_steps = config["n_steps"]
        warmup_steps = int(config.get("warmup_steps", 0))
        lr_min_ratio = float(config.get("lr_min_ratio", 0.0))
        if sched_name == "constant":
            scheduler = None
        elif sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=n_steps, eta_min=lr * lr_min_ratio)
        elif sched_name == "warmup_cosine":
            # linear warmup, then cosine to lr_min_ratio
            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return lr_min_ratio + (1.0 - lr_min_ratio) * cosine
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        elif sched_name == "exponential":
            gamma = float(config.get("lr_gamma", 0.999))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
        else:
            raise ValueError(f"Unknown lr_schedule: {sched_name!r}")

        # --- Batching mode ---
        bs = config["batch_size"]
        fresh_batch = config.get("fresh_batch", True)
        batching = config.get("batching", "random_replacement")

        if not fresh_batch:
            theta_all, x_all = simulator.sample(config["n_train"], rngs.train)
            theta_all = theta_all.to(self.device)
            x_all = x_all.to(self.device)
            n_train = theta_all.shape[0]

        grad_clip = float(config.get("grad_clip_norm", 5.0))
        losses = []
        t0 = time.time()

        epoch_idx_iter: Optional[tuple] = None
        for step in range(n_steps):
            # Get a batch
            if fresh_batch:
                theta_b, x_b = simulator.sample(bs, rngs.train)
                theta_b = theta_b.to(self.device)
                x_b = x_b.to(self.device)
            else:
                if batching == "random_replacement":
                    idx = torch.randint(0, n_train, (bs,))
                elif batching == "shuffle_epoch":
                    if epoch_idx_iter is None or epoch_idx_iter[1] >= n_train:
                        perm = rngs.init.permutation(n_train)
                        epoch_idx_iter = (torch.from_numpy(perm).long(), 0)
                    start = epoch_idx_iter[1]
                    end = min(start + bs, n_train)
                    idx = epoch_idx_iter[0][start:end]
                    if end - start < bs:
                        # wrap into next epoch
                        perm2 = rngs.init.permutation(n_train)
                        extra = torch.from_numpy(perm2[: bs - (end - start)]).long()
                        idx = torch.cat([idx, extra])
                        epoch_idx_iter = (torch.from_numpy(perm2).long(), bs - (end - start))
                    else:
                        epoch_idx_iter = (epoch_idx_iter[0], end)
                else:
                    raise ValueError(f"Unknown batching: {batching!r}")
                theta_b = theta_all[idx]
                x_b = x_all[idx]

            # Forward + loss + step
            context, log_det_contrib = self.conditioner.encode(x_b)
            r, log_det_flow = self.flow.forward(theta_b, context=context)
            log_det_total = log_det_flow + log_det_contrib
            loss_val = self.loss(r=r, log_det_jac_input=log_det_total)
            opt.zero_grad()
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=grad_clip)
            opt.step()
            if scheduler is not None:
                scheduler.step()
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

        # Gather arch metadata including new knobs
        flow_obj = self.flow
        dropout_val = getattr(getattr(flow_obj, "a", None), "mlp", None)
        arch_meta = {
            "flow_class": type(self.flow).__name__,
            "loss_class": type(self.loss).__name__,
            "loss_history_tail": losses[-min(100, len(losses)):],
            "fresh_batch": fresh_batch,
            "optimizer": opt_name,
            "lr_schedule": sched_name,
            "batching": batching,
            "grad_clip_norm": grad_clip,
        }

        return TrainedModel(
            procedure=procedure,
            state_dict=self.flow.state_dict(),
            final_loss=losses[-1],
            n_steps=n_steps,
            wall_clock_sec=wall,
            arch_metadata=arch_meta,
        )

    def n_params(self) -> dict:
        # AdditiveFlow1D exposes .a / .b UMNNBlocks; TriangularAdditiveFlow
        # exposes .a_blocks / .b_blocks ModuleLists of UMNNBlocks. Either way,
        # backbone = sum over the UMNNBlocks; head = α scalars (flow total - backbone).
        if hasattr(self.flow, "a_blocks"):
            backbone = sum(b.n_params() for b in self.flow.a_blocks) + sum(
                b.n_params() for b in self.flow.b_blocks
            )
            head = self.flow.n_params() - backbone  # the α scalars
            return {
                "backbone": backbone,
                "head": head,
                "calibration_stage": 0,
                "total": self.flow.n_params(),
                "kind": "flow",
            }
        if hasattr(self.flow, "a"):
            backbone = self.flow.a.n_params() + self.flow.b.n_params()
            head = self.flow.n_params() - backbone  # the α scalars
            return {
                "backbone": backbone,
                "head": head,
                "calibration_stage": 0,
                "total": self.flow.n_params(),
                "kind": "flow",
            }
        # v3 flows (DoublyMonotoneUMNN, JointUMNNFlow, JointUMNN1DFlow) and any
        # future flow without an explicit backbone/head split: report the
        # whole-flow parameter count.
        flow_total = self.flow.n_params()
        return {
            "backbone": flow_total,
            "head": 0,
            "calibration_stage": 0,
            "total": flow_total,
            "kind": "single_block",
        }
