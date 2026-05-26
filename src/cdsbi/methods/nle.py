"""NLERunner: wraps sbi.inference.SNLE_A with num_rounds=1 and our MAFAdapter.

API note (sbi 0.26.x): the density_estimator factory must return a
ConditionalDensityEstimator instance, not a bare nn.Module.  We reuse
_density_estimator_builder from npe.py — both SNPE_C and SNLE_A share
the same factory contract.

Note: for NLE the roles of theta/x are flipped relative to NPE — the
likelihood is p(x|theta), so x is the "input" and theta is the
"condition" in sbi's vocabulary.
"""
from __future__ import annotations

import time

import torch
from sbi.inference import SNLE_A
from sbi.neural_nets.estimators import NFlowsFlow
from sbi.utils import BoxUniform

from cdsbi.confidence_set.procedures import LikelihoodBasedProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import Runner, TrainedModel
from cdsbi.reproducibility.seeding import seed_everything


def _likelihood_estimator_builder(maf_adapter, features: int, context_features: int):
    """Factory for SNLE_A: input=x (features), condition=theta (context_features)."""
    def build(batch_theta: torch.Tensor, batch_x: torch.Tensor) -> NFlowsFlow:
        return NFlowsFlow(
            net=maf_adapter.flow,
            input_shape=torch.Size([features]),
            condition_shape=torch.Size([context_features]),
        )
    return build


class NLERunner(Runner):
    def __init__(self, flow, device: str = "auto"):
        self.flow = flow
        self.device = get_device(device)

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        seed_everything(seed)
        a, b = simulator.theta_range
        prior = BoxUniform(
            low=torch.tensor([a], device=self.device),
            high=torch.tensor([b], device=self.device),
        )

        # For NLE: input=x, condition=theta → features=d_x, context_features=d_theta
        inferer = SNLE_A(
            prior=prior,
            density_estimator=_likelihood_estimator_builder(
                self.flow.to(self.device),
                features=simulator.d_x,
                context_features=simulator.d_theta,
            ),
            device=str(self.device),
            show_progress_bars=False,
        )

        rngs = seed_everything(seed)
        theta, x = simulator.sample(config["n_train"], rngs.train)
        theta, x = theta.to(self.device), x.to(self.device)
        inferer.append_simulations(theta, x)

        t0 = time.time()
        inferer.train(max_num_epochs=config["n_epochs"], show_train_summary=False)
        wall = time.time() - t0

        flow = self.flow

        def log_likelihood_fn(theta: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
            return flow.log_prob(x=x_obs, context=theta)

        procedure = LikelihoodBasedProcedure(
            log_likelihood_fn=log_likelihood_fn,
            d_theta=simulator.d_theta,
            theta_range=simulator.theta_range,
        )

        return TrainedModel(
            procedure=procedure,
            state_dict={"flow": flow.state_dict()},
            final_loss=0.0,
            n_steps=config["n_epochs"],
            wall_clock_sec=wall,
            arch_metadata={"flow_class": "MAFAdapter", "method": "NLE_A"},
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
