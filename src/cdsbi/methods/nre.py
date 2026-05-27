"""NRERunner: binary-classifier-based likelihood ratio with the project's recipe knobs.

Bypasses sbi.SNRE_B.train() so the optimizer / LR schedule / warmup /
grad-clip / fresh_batch / batching knobs in the training config actually
apply. The classifier is just an MLP on concat[θ, X]; we train it via BCE
on joint (label 1) vs prior-shuffled (label 0) pairs and expose the logit
as the unnormalized log-ratio at inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cdsbi.confidence_set.procedures import RatioBasedProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import Runner, TrainedModel
from cdsbi.methods.training_utils import train_with_recipe
from cdsbi.reproducibility.seeding import seed_everything


def build_classifier_mlp(input_dim: int, hidden: int, depth: int) -> nn.Module:
    """MLP classifier head: [input_dim -> hidden -> hidden -> ... -> 1]."""
    layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, 1))
    return nn.Sequential(*layers)


def _nre_bce_loss(net, theta_b: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
    """BCE on joint (Y=1) vs shuffled (Y=0) pairs.

    For each batch element, the joint pair (θ_i, X_i) is labelled 1; the
    shuffled pair (θ_{π(i)}, X_i) — independently sampled θ given X — is
    labelled 0. Trained classifier f(θ, X) approximates the posterior
    odds-ratio at the joint, which is the log likelihood ratio at the
    prior level.
    """
    B = theta_b.shape[0]
    perm = torch.randperm(B, device=theta_b.device)
    theta_shuffled = theta_b[perm]
    inputs_joint = torch.cat([theta_b, x_b], dim=-1)
    inputs_marg = torch.cat([theta_shuffled, x_b], dim=-1)
    logits_joint = net(inputs_joint).squeeze(-1)
    logits_marg = net(inputs_marg).squeeze(-1)
    loss_joint = F.binary_cross_entropy_with_logits(
        logits_joint, torch.ones_like(logits_joint)
    )
    loss_marg = F.binary_cross_entropy_with_logits(
        logits_marg, torch.zeros_like(logits_marg)
    )
    return 0.5 * (loss_joint + loss_marg)


class NRERunner(Runner):
    def __init__(self, classifier_hidden: int = 32, classifier_depth: int = 3, device: str = "auto"):
        self.classifier_hidden = classifier_hidden
        self.classifier_depth = classifier_depth
        self.device = get_device(device)
        self._classifier: nn.Module = None  # built at fit time when d_theta/d_x are known

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        rngs = seed_everything(seed)

        input_dim = simulator.d_theta + simulator.d_x
        classifier = build_classifier_mlp(
            input_dim=input_dim,
            hidden=self.classifier_hidden,
            depth=self.classifier_depth,
        )
        self._classifier = classifier

        losses, wall = train_with_recipe(
            classifier, simulator.sample, config, self.device, _nre_bce_loss, rngs,
            n_train=int(config["n_train"]),
        )

        device = self.device

        def log_ratio_fn(theta: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
            theta = theta.to(device)
            x_obs = x_obs.to(device)
            n_th = theta.shape[0]
            x_rep = x_obs.expand(n_th, -1) if x_obs.shape[0] == 1 else x_obs
            return classifier(torch.cat([theta, x_rep], dim=-1)).squeeze(-1)

        procedure = RatioBasedProcedure(
            log_ratio_fn=log_ratio_fn,
            d_theta=simulator.d_theta,
            theta_range=simulator.theta_range,
        )

        n_class = sum(p.numel() for p in classifier.parameters())
        return TrainedModel(
            procedure=procedure,
            state_dict={"classifier": classifier.state_dict()},
            final_loss=float(losses[-1]),
            n_steps=int(config["n_steps"]),
            wall_clock_sec=wall,
            arch_metadata={
                "method": "NRE_custom",
                "classifier_params_actual": n_class,
                "loss_history_tail": losses[-min(100, len(losses)):],
                "optimizer": str(config.get("optimizer", "adam")),
                "lr_schedule": str(config.get("lr_schedule", "constant")),
                "fresh_batch": bool(config.get("fresh_batch", True)),
            },
        )

    def n_params(self, d_theta: int = 1, d_x: int = 1) -> dict:
        dummy = build_classifier_mlp(input_dim=d_theta + d_x, hidden=self.classifier_hidden, depth=self.classifier_depth)
        head = sum(p.numel() for p in dummy.parameters())
        return {
            "backbone": 0,
            "head": head,
            "calibration_stage": 0,
            "total": head,
            "kind": "classifier",
        }
