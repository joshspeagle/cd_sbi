"""LF2IRunner — two-stage: (1) NLE-style flow gives test statistic,
(2) shared-trunk multi-quantile MLP gives critical-value function c_α(θ).
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
from cdsbi.reproducibility.seeding import seed_everything

# _likelihood_estimator_builder(maf_adapter, features, context_features) -> factory
from cdsbi.methods.nle import _likelihood_estimator_builder


class MultiQuantileMLP(nn.Module):
    """Shared MLP trunk + per-quantile linear heads (paper's QuantileNN).

    Parameter count is independent of len(alpha_grid) once `depth` and `hidden`
    are fixed. Each forward returns a (B, n_quantiles) matrix of α-quantile
    predictions in the order `alpha_grid` was given.
    """

    def __init__(self, input_dim: int, hidden: int, depth: int, n_quantiles: int):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_quantiles)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        return torch.cat([h(z) for h in self.heads], dim=-1)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    diff = target - pred
    return torch.mean(torch.maximum(alpha * diff, (alpha - 1.0) * diff))


def multi_pinball_loss(
    preds: torch.Tensor, target: torch.Tensor, alphas: List[float]
) -> torch.Tensor:
    """Sum of per-quantile pinball losses over the shared trunk's heads."""
    losses = []
    for k, a in enumerate(alphas):
        losses.append(pinball_loss(preds[:, k], target, a))
    return torch.stack(losses).sum()


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

        # NOTE: sign convention. The inversion in CriticalValueProcedure is
        # {θ : T(θ; X_obs) ≤ c_α(θ)}, and we train c_α via pinball loss at
        # level α (the α-quantile of T | θ). For that pair to produce a set
        # with both nominal coverage AND useful power, T must be "large =
        # evidence against θ" (Wilks-direction). We therefore return
        # T = ll_ref − ll_θ: small when θ fits, large when it doesn't.
        # The naive ll_θ − ll_ref direction (used pre-2026-05-26) gave
        # nominal coverage by quantile algebra but wide sets, because at a
        # wrong θ* T(θ*; X_obs) was systematically below c_α(θ*).
        # This is a simplified fixed-reference LR test statistic, not the
        # paper's canonical LF2I form (ACORE / BFF / Waldo). A proper BFF
        # implementation lands in a follow-up commit.
        def test_stat_fn(theta: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
            theta = theta.to(self.device)
            x_obs = x_obs.to(self.device)
            theta_ref_b = theta_ref_t.expand(theta.shape[0], -1)
            ll_th = flow.log_prob(x=x_obs.expand(theta.shape[0], -1), context=theta)
            ll_ref = flow.log_prob(x=x_obs.expand(theta.shape[0], -1), context=theta_ref_b)
            return ll_ref - ll_th

        # === Stage 2: pinball-loss MLP for c_α(θ) ===
        theta_cal, x_cal = simulator.sample(config["n_train_quantile"], rngs.eval)
        theta_cal, x_cal = theta_cal.to(self.device), x_cal.to(self.device)
        with torch.no_grad():
            t_cal = torch.stack([
                test_stat_fn(theta_cal[i : i + 1], x_cal[i : i + 1]).squeeze()
                for i in range(theta_cal.shape[0])
            ])

        alpha_grid: List[float] = config["alpha_grid"]
        # Single shared-trunk multi-quantile MLP. Parameter count is
        # independent of len(alpha_grid), matching the paper's QuantileNN.
        critical_net = MultiQuantileMLP(
            input_dim=simulator.d_theta,
            hidden=self.quantile_hidden,
            depth=self.quantile_depth,
            n_quantiles=len(alpha_grid),
        ).to(self.device)
        opt = torch.optim.Adam(critical_net.parameters(), lr=1e-3)
        for _ in range(config["n_epochs_quantile"]):
            preds = critical_net(theta_cal)  # (B, n_quantiles)
            loss = multi_pinball_loss(preds, t_cal, alpha_grid)
            opt.zero_grad()
            loss.backward()
            opt.step()
        wall = time.time() - t0

        alpha_to_head_idx = {a: k for k, a in enumerate(alpha_grid)}

        def critical_value_fn(theta: torch.Tensor, alpha: float) -> torch.Tensor:
            k = alpha_to_head_idx[alpha]
            return critical_net(theta.to(self.device))[:, k]

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
                "critical_net": critical_net.state_dict(),
            },
            final_loss=0.0,
            n_steps=config["n_epochs_stat"] + config["n_epochs_quantile"],
            wall_clock_sec=wall,
            arch_metadata={
                "method": "LF2I",
                "test_statistic": "fixed_reference_LR_wilks_direction",
                "theta_ref": theta_ref,
                "alpha_grid": alpha_grid,
                "critical_net_class": "MultiQuantileMLP",
            },
        )

    def n_params(self, alpha_grid_len: int = 4) -> dict:
        backbone = self.stat_flow.n_params()
        dummy = MultiQuantileMLP(
            input_dim=1,
            hidden=self.quantile_hidden,
            depth=self.quantile_depth,
            n_quantiles=alpha_grid_len,
        )
        calibration_stage = sum(p.numel() for p in dummy.parameters())
        return {
            "backbone": backbone,
            "head": 0,
            "calibration_stage": calibration_stage,
            "total": backbone + calibration_stage,
            "kind": "two_stage",
        }
