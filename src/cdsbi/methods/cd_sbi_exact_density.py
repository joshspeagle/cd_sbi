"""ExactDensityCDSBIRunner: Stage-B Arm I-A. Trains an invertible summary + monotone
pivot end-to-end under the exact data NLL −log p(X|θ). The bijection preserves
information and the ancillary A is modelled θ-free N(0,I), so the objective pushes
all θ-information into the pivot features S — and is bounded below by H(X|θ) (no
collapse cheat possible). Inference uses only r(θ;S) ∈ ℝ^{d_theta}.
"""
from __future__ import annotations

import math
import time

import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.device import get_device
from cdsbi.flows.base import Guarantee
from cdsbi.losses.base import MonotonicityMismatchError
from cdsbi.methods.base import TrainedModel
from cdsbi.reproducibility.seeding import seed_everything


class ExactDensityCDSBIRunner:
    def __init__(self, flow, conditioner, loss=None, allow_ablation: bool = False, device: str = "auto"):
        self.flow = flow
        self.conditioner = conditioner          # must expose .transform(X) -> (z, log_det)
        self.loss = loss                        # ignored; the exact-density NLL is computed inline
        self.allow_ablation = allow_ablation
        self.device = get_device(device)
        if not allow_ablation:
            guarantees = getattr(flow, "monotonicity_guarantees", frozenset())
            if Guarantee.R1 not in guarantees:
                raise MonotonicityMismatchError(
                    f"ExactDensityCDSBIRunner requires R1 (monotone-in-θ); flow "
                    f"{type(flow).__name__} provides {sorted(g.value for g in guarantees)}."
                )

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        rngs = seed_everything(seed)
        dev = self.device
        self.flow.to(dev); self.flow.train()
        self.conditioner.to(dev); self.conditioner.train()
        d = int(simulator.d_theta)
        d_x = int(simulator.d_x)
        lr = float(config["lr"]); n_steps = int(config["n_steps"]); bs = int(config["batch_size"])
        grad_clip = float(config.get("grad_clip_norm", 5.0))
        fresh = bool(config.get("fresh_batch", True))

        params = list(self.flow.parameters()) + list(self.conditioner.parameters())
        opt = torch.optim.Adam(params, lr=lr)

        if not fresh:
            theta_all, x_all = simulator.sample(int(config["n_train"]), rngs.train)
            theta_all = theta_all.to(dev); x_all = x_all.to(dev)
            n_train = theta_all.shape[0]

        losses = []; t0 = time.time()
        const = 0.5 * d_x * math.log(2 * math.pi)
        for step in range(n_steps):
            if fresh:
                theta_b, x_b = simulator.sample(bs, rngs.train)
                theta_b = theta_b.to(dev); x_b = x_b.to(dev)
            else:
                idx = torch.randint(0, n_train, (bs,))
                theta_b = theta_all[idx]; x_b = x_all[idx]
            z, bij_logdet = self.conditioner.transform(x_b)
            S = z[:, :d]; A = z[:, d:]
            r, piv_logdet = self.flow.forward(theta_b, context=S)
            nll = (0.5 * (r.pow(2).sum(-1) + A.pow(2).sum(-1)) + const
                   - piv_logdet - bij_logdet)
            loss_val = nll.mean()
            opt.zero_grad(); loss_val.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            opt.step()
            losses.append(loss_val.item())
        wall = time.time() - t0

        self.flow.eval(); self.conditioner.eval()
        flow = self.flow; conditioner = self.conditioner; device = dev

        def pivot_fn(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            theta = theta.to(device); x = x.to(device)
            S, _ = conditioner.encode(x)
            r, _ = flow.forward(theta, context=S)
            return r

        def encode_fn(x: torch.Tensor) -> torch.Tensor:
            x = x.to(device)
            with torch.no_grad():
                S, _ = conditioner.encode(x)
            return S

        procedure = PivotBasedProcedure(pivot_fn=pivot_fn, d_theta=d, encode_fn=encode_fn)
        arch_meta = {
            "flow_class": type(self.flow).__name__,
            "loss_class": "ExactDensityLoss",
            "loss_history_tail": losses[-min(100, len(losses)):],
            "conditioner_class": type(self.conditioner).__name__,
            "conditioner_params": self.conditioner.n_params(),
        }
        return TrainedModel(procedure=procedure, state_dict={}, final_loss=float(losses[-1]),
                            n_steps=n_steps, wall_clock_sec=wall, arch_metadata=arch_meta)

    def n_params(self) -> dict:
        backbone = sum(p.numel() for p in self.flow.parameters())
        head = self.conditioner.n_params()
        return {"backbone": backbone, "head": head, "calibration_stage": 0,
                "total": backbone + head, "kind": "flow"}
