"""LF2IBFFRunner — LF2I with BFF (Bayes-Frequentist Factor) test statistic.

This is the paper's canonical LF2I variant for problems with a box-uniform
proposal: train a classifier to distinguish joint (θ, X∼F_θ) from marginal
(θ, X∼G) samples (G = ∫F_θ π(θ)dθ), then form

    T_BFF(θ; X) = log[(1/N) Σ_b O(X; θ_b)] − log O(X; θ_0)
                = logsumexp_b log_ratio(θ_b, X) − log N − log_ratio(θ_0, X)

where O(X; θ) is the classifier odds and the average is a Monte-Carlo estimate
of the proposal expectation E_{θ~π}[O(X;θ)] (Eq. 10): the θ_b are drawn from the
proposal π via simulator.sample (the same π the classifier was trained on), with
a d-aware sample count `max(marginal_n, 128·d)`. This is grid-free and scales as
O(N) in any dimension — NOT a tensor-product grid (exponential in d) nor a
box-uniform lattice (which would extrapolate the classifier outside a non-box
prior). The sign convention matches our existing CriticalValueProcedure: large
T = evidence against θ (Wilks-direction). Inversion as before: {θ : T(θ;X) ≤ c_α(θ)}.

References:
    Dalmasso et al., "Likelihood-Free Frequentist Inference", EJS 2024
    §3.2.2 (BFF definition), §3.3.1 (Algorithm 1 for c_α), §3.3.3 (Eq. 12).
"""
from __future__ import annotations

import math
import time
from typing import List

import torch

from cdsbi.confidence_set.procedures import CriticalValueProcedure, _BatchCache
from cdsbi.device import get_device
from cdsbi.methods.base import Runner, TrainedModel
from cdsbi.methods.lf2i import MultiQuantileMLP, train_multi_quantile_head
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
        marginal_n: int = None,
        marginal_grid_n: int = 64,
        device: str = "auto",
    ):
        self.classifier_hidden = classifier_hidden
        self.classifier_depth = classifier_depth
        self.quantile_hidden = quantile_hidden
        self.quantile_depth = quantile_depth
        # `marginal_n` is the Monte-Carlo sample count for the BFF marginal
        # E_{θ~π}[∏ O(X;θ)] (an MC budget, NOT a per-axis grid resolution).
        # `marginal_grid_n` is a deprecated alias kept for back-compat; an explicit
        # `marginal_n` wins.
        self.marginal_n = int(marginal_n if marginal_n is not None else marginal_grid_n)
        self.marginal_grid_n = self.marginal_n  # back-compat attribute
        self.device = get_device(device)

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        rngs = seed_everything(seed)

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
        # The BFF averaged term is an EXPECTATION under the proposal π
        # (Dalmasso et al. 2024, EJS, Eq. 10): E_{θ~π}[∏ᵢ O(Xᵢ; θ)]. The faithful,
        # dimension-scalable estimator is a Monte-Carlo average over draws θ_b ~ π —
        # NOT a tensor-product grid (exponential in d) nor a box-uniform lattice
        # (which extrapolates the classifier outside its training prior when the
        # prior is not box-uniform). We draw θ_b from the proposal via
        # simulator.sample (the same π the classifier was trained on), once, fixed
        # for the run so the statistic is deterministic given (θ, X). `marginal_n`
        # is the MC budget; a 128·d floor guards against under-sampling in higher d
        # (the prior d-blind N=64 grid collapsed coverage at d≥5 — audit 2026-05-30).
        # Uses rngs.eval (advancing its state once before the calibration draw).
        d = int(simulator.d_theta)
        n_marginal = max(self.marginal_n, 128 * d)
        theta_grid, _ = simulator.sample(n_marginal, rngs.eval)
        theta_grid = theta_grid.to(self.device)
        N_grid = theta_grid.shape[0]
        log_N = math.log(N_grid)
        device = self.device

        # The BFF marginal depends ONLY on x — never on the tested θ. Pilot
        # profiling (2026-06-10) showed set construction probing hundreds of θ
        # per fixed x_obs and recomputing the N_grid-row marginal on EVERY
        # probe: ~99.6% of the statistic's cost was x-only recomputation
        # (~25 min/run in SetSize). Cache it per live x tensor (anchor-
        # validated _BatchCache — the id-recycling-safe pattern) and count
        # evaluations so the cost stays observable.
        marginal_cache = _BatchCache()
        marginal_evals = {"n": 0}

        def _log_marginal(x_in: torch.Tensor) -> torch.Tensor:
            def _compute():
                marginal_evals["n"] += 1
                xb = x_in.to(device)
                B = xb.shape[0]
                tg = theta_grid.unsqueeze(0).expand(B, -1, -1).reshape(-1, theta_grid.shape[-1])
                xe = xb.unsqueeze(1).expand(-1, N_grid, -1).reshape(-1, xb.shape[-1])
                lr = classifier(torch.cat([tg, xe], dim=-1)).squeeze(-1).view(B, N_grid)
                return torch.logsumexp(lr, dim=1) - log_N

            return marginal_cache.get_or_compute(
                ("bff_marginal", id(x_in), int(x_in.shape[0])), _compute, anchor=x_in,
            )

        def test_stat_fn(theta: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
            x_in = x_obs  # cache anchor: the caller's stable tensor object
            theta = theta.to(device)
            x_obs = x_obs.to(device)
            # Support either {one θ, many X} (contains_batch) or {many θ, one X}
            # (grid search) by broadcasting the singleton dimension.
            if theta.shape[0] != x_obs.shape[0]:
                if x_obs.shape[0] == 1:
                    # one X, many θ: the marginal is a single scalar for this x —
                    # computed once (cached), reused across every θ probe.
                    log_marginal = _log_marginal(x_in)            # (1,)
                    log_r_at = classifier(
                        torch.cat([theta, x_obs.expand(theta.shape[0], -1)], dim=-1)
                    ).squeeze(-1)
                    return log_marginal.expand_as(log_r_at) - log_r_at
                elif theta.shape[0] == 1:
                    theta = theta.expand(x_obs.shape[0], -1)
                else:
                    raise ValueError(
                        f"test_stat_fn: shape mismatch theta {theta.shape}, x_obs {x_obs.shape}"
                    )
            # log O(X_i; θ_i) — the numerator (size B).
            log_r_at = classifier(torch.cat([theta, x_obs], dim=-1)).squeeze(-1)
            log_marginal = _log_marginal(x_in)
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
        stage2_losses = train_multi_quantile_head(
            critical_net, theta_cal, t_cal, alpha_grid, config, self.device,
        )
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
        procedure.bff_marginal_evals = marginal_evals
        return TrainedModel(
            procedure=procedure,
            state_dict={
                "classifier": classifier.state_dict(),
                "critical_net": critical_net.state_dict(),
            },
            final_loss=float(stage1_losses[-1]),
            n_steps=int(config["n_steps"]) * 2,  # stage-1 + stage-2 share the recipe's n_steps
            wall_clock_sec=wall,
            arch_metadata={
                "method": "LF2I_BFF",
                "test_statistic": "bff_wilks_direction",
                "marginal_n": self.marginal_n,
                "marginal_n_effective": N_grid,
                "marginal_estimator": "mc_from_proposal",
                "alpha_grid": alpha_grid,
                "critical_net_class": "MultiQuantileMLP",
                "classifier_params_actual": n_class,
                "stage1_loss_tail": stage1_losses[-min(100, len(stage1_losses)):],
                "stage2_loss_tail": stage2_losses[-min(100, len(stage2_losses)):],
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
