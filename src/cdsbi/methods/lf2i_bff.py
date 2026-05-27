"""LF2IBFFRunner — LF2I with BFF (Bayes-Frequentist Factor) test statistic.

This is the paper's canonical LF2I variant for problems with a box-uniform
proposal: train a classifier to distinguish joint (θ, X∼F_θ) from marginal
(θ, X∼G) samples (G = ∫F_θ π(θ)dθ), then form

    T_BFF(θ; X) = log[(1/N) Σ_b O(X; θ_b)] − log O(X; θ_0)
                = logsumexp_b log_ratio(θ_b, X) − log N − log_ratio(θ_0, X)

where O(X; θ) is the classifier odds and the average is taken over an
integration grid drawn from the prior. The sign convention matches our
existing CriticalValueProcedure: large T = evidence against θ
(Wilks-direction). Inversion as before: {θ : T(θ; X) ≤ c_α(θ)}.

References:
    Dalmasso et al., "Likelihood-Free Frequentist Inference", EJS 2024
    §3.2.2 (BFF definition), §3.3.1 (Algorithm 1 for c_α), §3.3.3 (Eq. 12).
"""
from __future__ import annotations

import math
import time
from typing import List

import torch

from cdsbi.confidence_set.procedures import CriticalValueProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import Runner, TrainedModel
from cdsbi.methods.lf2i import MultiQuantileMLP, multi_pinball_loss
from cdsbi.methods.nre import _nre_bce_loss, build_classifier_mlp
from cdsbi.methods.training_utils import train_with_recipe
from cdsbi.reproducibility.seeding import seed_everything


class LF2IBFFRunner(Runner):
    def __init__(
        self,
        classifier_hidden: int = 32,
        classifier_depth: int = 3,
        quantile_hidden: int = 12,
        quantile_depth: int = 2,
        marginal_grid_n: int = 64,
        device: str = "auto",
    ):
        self.classifier_hidden = classifier_hidden
        self.classifier_depth = classifier_depth
        self.quantile_hidden = quantile_hidden
        self.quantile_depth = quantile_depth
        self.marginal_grid_n = marginal_grid_n
        self.device = get_device(device)

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        rngs = seed_everything(seed)
        a, b = simulator.theta_range

        # === Stage 1: train NRE-style classifier with our recipe knobs ===
        classifier = build_classifier_mlp(
            input_dim=simulator.d_theta + simulator.d_x,
            hidden=self.classifier_hidden,
            depth=self.classifier_depth,
        )

        t0 = time.time()
        stage1_losses, _ = train_with_recipe(
            classifier, simulator.sample, config, self.device, _nre_bce_loss, rngs,
            n_train=int(config["n_train_stat"]),
        )

        # === Stage 2: build BFF test statistic ===
        # Integration grid over the prior (uniform over [a, b] in our case).
        theta_grid = torch.linspace(a, b, self.marginal_grid_n, device=self.device).view(-1, 1)
        N_grid = theta_grid.shape[0]
        log_N = math.log(N_grid)
        device = self.device

        def test_stat_fn(theta: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
            theta = theta.to(device)
            x_obs = x_obs.to(device)
            # Support either {one θ, many X} (contains_batch) or {many θ, one X}
            # (grid search) by broadcasting the singleton dimension.
            if theta.shape[0] != x_obs.shape[0]:
                if x_obs.shape[0] == 1:
                    x_obs = x_obs.expand(theta.shape[0], -1)
                elif theta.shape[0] == 1:
                    theta = theta.expand(x_obs.shape[0], -1)
                else:
                    raise ValueError(
                        f"test_stat_fn: shape mismatch theta {theta.shape}, x_obs {x_obs.shape}"
                    )
            B = theta.shape[0]
            # log O(X_i; θ_i) — the numerator (size B).
            log_r_at = classifier(torch.cat([theta, x_obs], dim=-1)).squeeze(-1)
            # logsumexp_b log O(X_i; θ_b) − log N — the marginal denominator.
            theta_grid_exp = theta_grid.unsqueeze(0).expand(B, -1, -1).reshape(-1, theta.shape[-1])
            x_obs_exp = x_obs.unsqueeze(1).expand(-1, N_grid, -1).reshape(-1, x_obs.shape[-1])
            log_r_grid = classifier(
                torch.cat([theta_grid_exp, x_obs_exp], dim=-1)
            ).squeeze(-1).view(B, N_grid)
            log_marginal = torch.logsumexp(log_r_grid, dim=1) - log_N
            return log_marginal - log_r_at  # Wilks direction: large = bad fit at θ

        # === Stage 3: shared-trunk multi-quantile head ===
        theta_cal, x_cal = simulator.sample(config["n_train_quantile"], rngs.eval)
        theta_cal, x_cal = theta_cal.to(self.device), x_cal.to(self.device)
        with torch.no_grad():
            t_cal = torch.stack([
                test_stat_fn(theta_cal[i : i + 1], x_cal[i : i + 1]).squeeze()
                for i in range(theta_cal.shape[0])
            ])

        alpha_grid: List[float] = config["alpha_grid"]
        critical_net = MultiQuantileMLP(
            input_dim=simulator.d_theta,
            hidden=self.quantile_hidden,
            depth=self.quantile_depth,
            n_quantiles=len(alpha_grid),
        ).to(self.device)
        opt = torch.optim.Adam(critical_net.parameters(), lr=1e-3)
        for _ in range(config["n_epochs_quantile"]):
            preds = critical_net(theta_cal)
            loss = multi_pinball_loss(preds, t_cal, alpha_grid)
            opt.zero_grad()
            loss.backward()
            opt.step()
        wall = time.time() - t0

        alpha_to_head_idx = {a_: k for k, a_ in enumerate(alpha_grid)}

        def critical_value_fn(theta: torch.Tensor, alpha: float) -> torch.Tensor:
            k = alpha_to_head_idx[alpha]
            return critical_net(theta.to(self.device))[:, k]

        procedure = CriticalValueProcedure(
            test_stat_fn=test_stat_fn,
            critical_value_fn=critical_value_fn,
            d_theta=simulator.d_theta,
            theta_range=simulator.theta_range,
        )

        n_class = sum(p.numel() for p in classifier.parameters())
        return TrainedModel(
            procedure=procedure,
            state_dict={
                "classifier": classifier.state_dict(),
                "critical_net": critical_net.state_dict(),
            },
            final_loss=float(stage1_losses[-1]),
            n_steps=int(config["n_steps"]) + int(config["n_epochs_quantile"]),
            wall_clock_sec=wall,
            arch_metadata={
                "method": "LF2I_BFF",
                "test_statistic": "bff_wilks_direction",
                "marginal_grid_n": self.marginal_grid_n,
                "alpha_grid": alpha_grid,
                "critical_net_class": "MultiQuantileMLP",
                "classifier_params_actual": n_class,
                "stage1_loss_tail": stage1_losses[-min(100, len(stage1_losses)):],
                "optimizer": str(config.get("optimizer", "adam")),
                "lr_schedule": str(config.get("lr_schedule", "constant")),
                "fresh_batch": bool(config.get("fresh_batch", True)),
            },
        )

    def n_params(self, d_theta: int = 1, d_x: int = 1, alpha_grid_len: int = 4) -> dict:
        backbone = sum(p.numel() for p in build_classifier_mlp(
            input_dim=d_theta + d_x,
            hidden=self.classifier_hidden,
            depth=self.classifier_depth,
        ).parameters())
        head = MultiQuantileMLP(
            input_dim=d_theta,
            hidden=self.quantile_hidden,
            depth=self.quantile_depth,
            n_quantiles=alpha_grid_len,
        )
        calibration_stage = sum(p.numel() for p in head.parameters())
        return {
            "backbone": backbone,
            "head": 0,
            "calibration_stage": calibration_stage,
            "total": backbone + calibration_stage,
            "kind": "two_stage_bff",
        }
