"""LF2IRunner — two-stage: (1) NLE-style flow gives test statistic,
(2) pinball-loss MLP gives critical-value function c_α(θ).
"""
from __future__ import annotations

import time
from typing import List

import torch
import torch.nn as nn
from sbi.inference import SNLE_A
from sbi.utils import BoxUniform

from cdsbi.confidence_set.procedures import CriticalValueProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import Runner, TrainedModel
from cdsbi.methods.nre import build_classifier_mlp
from cdsbi.reproducibility.seeding import seed_everything

# _likelihood_estimator_builder(maf_adapter, features, context_features) -> factory
from cdsbi.methods.nle import _likelihood_estimator_builder


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    diff = target - pred
    return torch.mean(torch.maximum(alpha * diff, (alpha - 1.0) * diff))


class LF2IRunner(Runner):
    def __init__(
        self,
        stat_flow,
        quantile_hidden: int = 32,
        quantile_depth: int = 3,
        theta_ref: float = None,  # default: prior midpoint
        device: str = "auto",
    ):
        self.stat_flow = stat_flow
        self.quantile_hidden = quantile_hidden
        self.quantile_depth = quantile_depth
        self.theta_ref = theta_ref
        self.device = get_device(device)

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        rngs = seed_everything(seed)
        a, b = simulator.theta_range
        theta_ref = self.theta_ref if self.theta_ref is not None else 0.5 * (a + b)

        # === Stage 1: train NLE-style flow ===
        # _likelihood_estimator_builder signature: (maf_adapter, features, context_features)
        # For NLE: features=d_x (input), context_features=d_theta (condition).
        prior = BoxUniform(
            low=torch.tensor([a], device=self.device),
            high=torch.tensor([b], device=self.device),
        )
        # NOTE: don't pre-move stat_flow / training tensors — see npe.py for the
        # sbi 0.26 CPU-probe rationale. sbi will move the net to self._device.
        inferer = SNLE_A(
            prior=prior,
            density_estimator=_likelihood_estimator_builder(
                self.stat_flow,
                features=simulator.d_x,
                context_features=simulator.d_theta,
            ),
            device=str(self.device),
            show_progress_bars=False,
        )
        theta, x = simulator.sample(config["n_train_stat"], rngs.train)
        inferer.append_simulations(theta, x)

        t0 = time.time()
        inferer.train(max_num_epochs=config["n_epochs_stat"], show_train_summary=False)
        flow = self.stat_flow

        theta_ref_t = torch.tensor([[theta_ref]], device=self.device, dtype=torch.float32)

        def test_stat_fn(theta: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
            theta = theta.to(self.device)
            x_obs = x_obs.to(self.device)
            theta_ref_b = theta_ref_t.expand(theta.shape[0], -1)
            ll_th = flow.log_prob(x=x_obs.expand(theta.shape[0], -1), context=theta)
            ll_ref = flow.log_prob(x=x_obs.expand(theta.shape[0], -1), context=theta_ref_b)
            return ll_th - ll_ref

        # === Stage 2: pinball-loss MLP for c_α(θ) ===
        theta_cal, x_cal = simulator.sample(config["n_train_quantile"], rngs.eval)
        theta_cal, x_cal = theta_cal.to(self.device), x_cal.to(self.device)
        with torch.no_grad():
            t_cal = torch.stack([
                test_stat_fn(theta_cal[i : i + 1], x_cal[i : i + 1]).squeeze()
                for i in range(theta_cal.shape[0])
            ])

        alpha_grid: List[float] = config["alpha_grid"]
        critical_nets = {}
        for alpha in alpha_grid:
            net = build_classifier_mlp(
                input_dim=simulator.d_theta,
                hidden=self.quantile_hidden,
                depth=self.quantile_depth,
            ).to(self.device)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            for _ in range(config["n_epochs_quantile"]):
                pred = net(theta_cal).squeeze(-1)
                loss = pinball_loss(pred, t_cal, alpha)
                opt.zero_grad()
                loss.backward()
                opt.step()
            critical_nets[alpha] = net
        wall = time.time() - t0

        def critical_value_fn(theta: torch.Tensor, alpha: float) -> torch.Tensor:
            net = critical_nets[alpha]
            return net(theta.to(self.device)).squeeze(-1)

        procedure = CriticalValueProcedure(
            test_stat_fn=test_stat_fn,
            critical_value_fn=critical_value_fn,
            d_theta=simulator.d_theta,
            theta_range=simulator.theta_range,
        )

        return TrainedModel(
            procedure=procedure,
            state_dict={
                "stat_flow": flow.state_dict(),
                "critical_nets": {a: net.state_dict() for a, net in critical_nets.items()},
            },
            final_loss=0.0,
            n_steps=config["n_epochs_stat"] + config["n_epochs_quantile"],
            wall_clock_sec=wall,
            arch_metadata={"method": "LF2I", "theta_ref": theta_ref, "alpha_grid": alpha_grid},
        )

    def n_params(self, alpha_grid_len: int = 4) -> dict:
        backbone = self.stat_flow.n_params()
        dummy = build_classifier_mlp(
            input_dim=1, hidden=self.quantile_hidden, depth=self.quantile_depth
        )
        per_net = sum(p.numel() for p in dummy.parameters())
        calibration_stage = alpha_grid_len * per_net
        return {
            "backbone": backbone,
            "head": 0,
            "calibration_stage": calibration_stage,
            "total": backbone + calibration_stage,
            "kind": "two_stage",
        }
