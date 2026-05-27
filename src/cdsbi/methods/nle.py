"""NLERunner: density estimation p̂(X|θ) via MLE with the project's recipe knobs.

Bypasses sbi.SNLE_A.train() so the optimizer / LR schedule / warmup /
grad-clip / fresh_batch / batching knobs in the training config actually
apply (sbi's internal train loop ignores all of them). Inference still
uses our LikelihoodBasedProcedure with a closure over the trained flow.
"""
from __future__ import annotations

from typing import List

import torch

from cdsbi.confidence_set.procedures import LikelihoodBasedProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import Runner, TrainedModel
from cdsbi.methods.training_utils import train_with_recipe
from cdsbi.reproducibility.seeding import seed_everything


class NLERunner(Runner):
    def __init__(self, flow, device: str = "auto"):
        self.flow = flow
        self.device = get_device(device)

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        rngs = seed_everything(seed)

        # NLE: x is the input, θ is the conditioning context.
        # MAFAdapter expects features=d_x and context_features=d_theta at
        # construction time; loss is the standard negative-log-likelihood.
        def nle_loss(net, theta_b, x_b):
            return -net.log_prob(x_b, context=theta_b).mean()

        losses, wall = train_with_recipe(
            self.flow, simulator.sample, config, self.device, nle_loss, rngs,
            n_train=int(config["n_train"]),
        )

        flow = self.flow
        device = self.device

        def log_likelihood_fn(theta: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
            return flow.log_prob(x=x_obs.to(device), context=theta.to(device))

        procedure = LikelihoodBasedProcedure(
            log_likelihood_fn=log_likelihood_fn,
            d_theta=simulator.d_theta,
            theta_range=simulator.theta_range,
        )

        return TrainedModel(
            procedure=procedure,
            state_dict={"flow": flow.state_dict()},
            final_loss=float(losses[-1]),
            n_steps=int(config["n_steps"]),
            wall_clock_sec=wall,
            arch_metadata={
                "method": "NLE",
                "flow_class": "MAFAdapter",
                "loss_history_tail": losses[-min(100, len(losses)):],
                "optimizer": str(config.get("optimizer", "adam")),
                "lr_schedule": str(config.get("lr_schedule", "constant")),
                "fresh_batch": bool(config.get("fresh_batch", True)),
            },
        )

    def n_params(self) -> dict:
        backbone = self.flow.n_params()
        return {
            "backbone": backbone,
            "head": 0,
            "calibration_stage": 0,
            "total": backbone,
            "kind": "flow",
        }
