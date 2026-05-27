"""NPERunner: density estimation p̂(θ|X) via MLE with the project's recipe knobs.

Bypasses sbi.SNPE_C.train() so the optimizer / LR schedule / warmup /
grad-clip / fresh_batch / batching knobs in the training config actually
apply. We keep sbi's DirectPosterior at inference time for its batched
rejection sampling (used by PosteriorBasedProcedure.contains_batch).
"""
from __future__ import annotations

import pickle

import torch
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.neural_nets.estimators import NFlowsFlow
from sbi.utils import BoxUniform

from cdsbi.confidence_set.procedures import PosteriorBasedProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import Runner, TrainedModel
from cdsbi.methods.training_utils import train_with_recipe
from cdsbi.reproducibility.seeding import seed_everything


class NPERunner(Runner):
    def __init__(self, flow, device: str = "auto"):
        self.flow = flow
        self.device = get_device(device)

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        rngs = seed_everything(seed)
        a, b = simulator.theta_range

        # NPE: θ is the input, X is the conditioning context.
        # MAFAdapter is constructed with features=d_theta, context_features=d_x.
        def npe_loss(net, theta_b, x_b):
            return -net.log_prob(theta_b, context=x_b).mean()

        losses, wall = train_with_recipe(
            self.flow, simulator.sample, config, self.device, npe_loss, rngs,
            n_train=int(config["n_train"]),
        )

        # Wrap the trained nflows.Flow in sbi's NFlowsFlow estimator so we can
        # use DirectPosterior's batched rejection sampling.
        density_estimator = NFlowsFlow(
            net=self.flow.flow,
            input_shape=torch.Size([simulator.d_theta]),
            condition_shape=torch.Size([simulator.d_x]),
        )
        density_estimator.to(self.device)

        prior = BoxUniform(
            low=torch.tensor([a], device=self.device),
            high=torch.tensor([b], device=self.device),
        )
        posterior = DirectPosterior(
            posterior_estimator=density_estimator,
            prior=prior,
            device=str(self.device),
        )
        device = self.device

        def sample_fn(x_obs: torch.Tensor, n: int) -> torch.Tensor:
            return posterior.sample(
                (n,), x=x_obs.squeeze(0).to(device), show_progress_bars=False,
            )

        def sample_batched_fn(x_obs_batch: torch.Tensor, n: int) -> torch.Tensor:
            return posterior.sample_batched(
                (n,), x=x_obs_batch.to(device), show_progress_bars=False,
            )

        procedure = PosteriorBasedProcedure(
            sample_fn=sample_fn,
            d_theta=simulator.d_theta,
            sample_batched_fn=sample_batched_fn,
        )

        return TrainedModel(
            procedure=procedure,
            state_dict={"pickle": pickle.dumps(posterior)},
            final_loss=float(losses[-1]),
            n_steps=int(config["n_steps"]),
            wall_clock_sec=wall,
            arch_metadata={
                "method": "NPE",
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
