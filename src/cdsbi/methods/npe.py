"""NPERunner: wraps sbi.inference.SNPE_C with num_rounds=1 (amortized) and our MAFAdapter.

API note (sbi 0.26.x): the density_estimator factory must return a
ConditionalDensityEstimator instance, not a bare nn.Module.  We use
sbi's NFlowsFlow wrapper around the MAFAdapter's internal nflows.Flow.
"""
from __future__ import annotations

import pickle
import time

import torch
from sbi.inference import SNPE_C
from sbi.neural_nets.estimators import NFlowsFlow
from sbi.utils import BoxUniform

from cdsbi.confidence_set.procedures import PosteriorBasedProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import Runner, TrainedModel
from cdsbi.reproducibility.seeding import seed_everything


def _density_estimator_builder(maf_adapter, features: int, context_features: int):
    """Factory that sbi's SNPE_C accepts as `density_estimator`.

    sbi calls build(batch_theta, batch_x) on the first training round and expects
    a ConditionalDensityEstimator back.  We wrap the MAFAdapter's internal
    nflows.Flow in NFlowsFlow which satisfies that interface.
    """
    def build(batch_theta: torch.Tensor, batch_x: torch.Tensor) -> NFlowsFlow:
        return NFlowsFlow(
            net=maf_adapter.flow,
            input_shape=torch.Size([features]),
            condition_shape=torch.Size([context_features]),
        )
    return build


class NPERunner(Runner):
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

        inferer = SNPE_C(
            prior=prior,
            density_estimator=_density_estimator_builder(
                self.flow.to(self.device),
                features=simulator.d_theta,
                context_features=simulator.d_x,
            ),
            device=str(self.device),
            show_progress_bars=False,
        )

        rngs = seed_everything(seed)
        theta, x = simulator.sample(config["n_train"], rngs.train)
        theta, x = theta.to(self.device), x.to(self.device)
        inferer.append_simulations(theta, x)

        t0 = time.time()
        density_estimator = inferer.train(
            max_num_epochs=config["n_epochs"], show_train_summary=False
        )
        wall = time.time() - t0

        posterior = inferer.build_posterior(density_estimator)

        def sample_fn(x_obs: torch.Tensor, n: int) -> torch.Tensor:
            return posterior.sample((n,), x=x_obs.squeeze(0), show_progress_bars=False)

        procedure = PosteriorBasedProcedure(
            sample_fn=sample_fn, d_theta=simulator.d_theta
        )

        training_loss = inferer.summary.get("training_loss", [])
        final_loss = float(training_loss[-1]) if training_loss else 0.0

        return TrainedModel(
            procedure=procedure,
            state_dict={"pickle": pickle.dumps(posterior)},
            final_loss=final_loss,
            n_steps=config["n_epochs"],
            wall_clock_sec=wall,
            arch_metadata={"flow_class": "MAFAdapter", "method": "NPE_C"},
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
