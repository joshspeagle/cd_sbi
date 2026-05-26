"""NRERunner: wraps sbi.inference.SNRE_B with num_rounds=1 and a controlled MLP classifier.

API note (sbi 0.26.x): the classifier factory must return a RatioEstimator instance
(a ConditionalEstimator subclass), not a bare nn.Module.  We wrap our MLP in
sbi's RatioEstimator, which handles the theta/x concatenation internally via
combine_theta_and_x and exposes unnormalized_log_ratio(theta, x).

The trained ratio_estimator returned by inferer.train() is a RatioEstimator;
log_ratio_fn therefore calls ratio_estimator.unnormalized_log_ratio(theta, x)
with separate tensors rather than pre-concatenating them.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
from sbi.inference import SNRE_B
from sbi.neural_nets.ratio_estimators import RatioEstimator
from sbi.utils import BoxUniform

from cdsbi.confidence_set.procedures import RatioBasedProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import Runner, TrainedModel
from cdsbi.reproducibility.seeding import seed_everything


def build_classifier_mlp(input_dim: int, hidden: int, depth: int) -> nn.Module:
    """MLP classifier head: [input_dim -> hidden -> hidden -> ... -> 1]."""
    layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, 1))
    return nn.Sequential(*layers)


def _classifier_builder(hidden: int, depth: int):
    """Factory satisfying sbi's ConditionalEstimatorBuilder protocol.

    sbi calls build(batch_theta, batch_x) on the first training batch to infer
    shapes, then expects a RatioEstimator back.  We construct the MLP with the
    correct input_dim (theta_dim + x_dim) and wrap it in RatioEstimator, which
    handles concatenation internally.
    """
    def build(batch_theta: torch.Tensor, batch_x: torch.Tensor) -> RatioEstimator:
        theta_dim = batch_theta.shape[-1]
        x_dim = batch_x.shape[-1]
        input_dim = theta_dim + x_dim
        net = build_classifier_mlp(input_dim, hidden, depth)
        return RatioEstimator(
            net=net,
            theta_shape=torch.Size([theta_dim]),
            x_shape=torch.Size([x_dim]),
        )
    return build


class NRERunner(Runner):
    def __init__(self, classifier_hidden: int = 32, classifier_depth: int = 3, device: str = "auto"):
        self.classifier_hidden = classifier_hidden
        self.classifier_depth = classifier_depth
        self.device = get_device(device)

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        seed_everything(seed)
        a, b = simulator.theta_range
        prior = BoxUniform(
            low=torch.tensor([a], device=self.device),
            high=torch.tensor([b], device=self.device),
        )

        inferer = SNRE_B(
            prior=prior,
            classifier=_classifier_builder(self.classifier_hidden, self.classifier_depth),
            device=str(self.device),
            show_progress_bars=False,
        )

        rngs = seed_everything(seed)
        theta, x = simulator.sample(config["n_train"], rngs.train)
        theta, x = theta.to(self.device), x.to(self.device)
        inferer.append_simulations(theta, x)

        t0 = time.time()
        ratio_estimator = inferer.train(max_num_epochs=config["n_epochs"], show_train_summary=False)
        wall = time.time() - t0

        # ratio_estimator is a RatioEstimator; its forward / unnormalized_log_ratio
        # takes separate theta and x tensors (not pre-concatenated).
        # Shape contract: theta (n_theta, d_theta), x_obs (1, d_x) broadcast to (n_theta, d_x).
        def log_ratio_fn(theta: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
            n_th = theta.shape[0]
            x_rep = x_obs.expand(n_th, -1)
            return ratio_estimator.unnormalized_log_ratio(theta, x_rep).squeeze(-1)

        procedure = RatioBasedProcedure(
            log_ratio_fn=log_ratio_fn,
            d_theta=simulator.d_theta,
            theta_range=simulator.theta_range,
        )

        n_class = sum(p.numel() for p in ratio_estimator.parameters())
        return TrainedModel(
            procedure=procedure,
            state_dict={"classifier": ratio_estimator.state_dict()},
            final_loss=0.0,
            n_steps=config["n_epochs"],
            wall_clock_sec=wall,
            arch_metadata={"method": "NRE_B", "classifier_params_actual": n_class},
        )

    def n_params(self) -> dict:
        # Build a dummy classifier to count with the correct input_dim for 1D problems
        # (theta_dim=1, x_dim=1 -> input_dim=2)
        dummy = build_classifier_mlp(input_dim=2, hidden=self.classifier_hidden, depth=self.classifier_depth)
        head = sum(p.numel() for p in dummy.parameters())
        return {
            "backbone": 0,
            "head": head,
            "calibration_stage": 0,
            "total": head,
            "kind": "classifier",
        }
