"""ScoreCDRunner — Score-CD: a neural likelihood read out as a frequentist
confidence distribution via the score (Rao) pivot.

Stage 1 trains an NLE conditional density q_φ(X|θ) (identical to NLERunner) — no
bottleneck, information-complete, cannot collapse. Inference reads a d_θ-dimensional
frequentist CD from the SCORE U(θ;X)=∇_θ log q_φ(X|θ), which at the true θ has mean
zero and covariance the Fisher information I(θ) when q_φ = p (the surrogate inherits
this only as q_φ → p), and is ≈N(0,I(θ)) by the CLT over the n_iid replicates. Two readouts (both as a CriticalValueProcedure, so they
slot into the coverage/set-size diagnostics exactly like LF2I-BFF):

  variant="rao": stat = U(θ;X)ᵀ Î(θ)⁻¹ U(θ;X) ~ χ²_{d_θ} (asymptotic), critical value
                 the constant χ²_{d_θ,α}. Î(θ) is MC-estimated from the simulator at the
                 queried θ (cached), so there is no per-row Fisher blow-up.
  variant="cal": stat = ‖U(θ;X)‖² with a LEARNED critical value c_α(θ) (quantile
                 regression — LF2I-style, grid-free). Robust to finite-sample looseness.

The score needs autograd through q_φ w.r.t. θ, so the statistic enables grad internally
(it is evaluated under the diagnostics' torch.no_grad()).

Reference: Rao score test; LF2I (Dalmasso et al. 2024) for the calibrated readout.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
from scipy.stats import chi2

from cdsbi.confidence_set.procedures import CriticalValueProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import Runner, TrainedModel
from cdsbi.methods.lf2i import MultiQuantileMLP, train_multi_quantile_head
from cdsbi.methods.training_utils import train_with_recipe
from cdsbi.reproducibility.seeding import seed_everything


class ScoreCDRunner(Runner):
    def __init__(self, flow, variant: str = "rao", fisher_n: int = 4000,
                 quantile_hidden: int = 64, quantile_depth: int = 2, device: str = "auto"):
        if variant not in ("rao", "cal"):
            raise ValueError(f"variant must be 'rao' or 'cal', got {variant!r}")
        self.flow = flow
        self.variant = variant
        self.fisher_n = fisher_n
        self.quantile_hidden = quantile_hidden
        self.quantile_depth = quantile_depth
        self.device = get_device(device)

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        rngs = seed_everything(seed)
        dev = self.device
        d = int(simulator.d_theta)

        # === Stage 1: NLE density model q_φ(X|θ) (same objective as NLERunner) ===
        def nle_loss(net, theta_b, x_b):
            return -net.log_prob(x_b, context=theta_b).mean()

        losses, wall = train_with_recipe(
            self.flow, simulator.sample, config, dev, nle_loss, rngs,
            n_train=int(config["n_train"]),
        )
        self.flow.eval()
        flow = self.flow

        def score(theta_rows: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            """U(θ;X) = ∇_θ log q_φ(X|θ), shape (n, d). Enables grad internally so it
            works under the diagnostics' no_grad; returns a detached tensor."""
            with torch.enable_grad():
                th = theta_rows.detach().clone().to(dev).requires_grad_(True)
                lp = flow.log_prob(x.to(dev), context=th).sum()
                g, = torch.autograd.grad(lp, th)
            # Guard against non-finite scores from an under-trained / unstable flow
            # (a huge/NaN gradient on an extreme X would poison the Fisher matmul).
            return torch.nan_to_num(g.detach(), nan=0.0, posinf=0.0, neginf=0.0)

        if self.variant == "rao":
            fisher_n = self.fisher_n
            fisher_cache: dict = {}
            fisher_rng = np.random.default_rng(seed + 777)

            def fisher_inv_sqrtdet(theta_row):
                key = tuple(round(float(v), 6) for v in theta_row)
                if key not in fisher_cache:
                    xf = simulator.sample_x_given_theta(tuple(theta_row), fisher_n, fisher_rng)
                    thr = torch.tensor([list(theta_row)], dtype=xf.dtype).expand(fisher_n, d)
                    U = score(thr, xf).cpu().numpy()
                    Iinv = np.linalg.inv(U.T @ U / fisher_n + 1e-6 * np.eye(d))
                    fisher_cache[key] = torch.tensor(Iinv, dtype=torch.float32, device=dev)
                return fisher_cache[key]

            def test_stat_fn(theta: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
                theta = theta.to(dev); x_obs = x_obs.to(dev)
                if theta.shape[0] != x_obs.shape[0]:
                    if x_obs.shape[0] == 1:
                        x_obs = x_obs.expand(theta.shape[0], -1)
                    elif theta.shape[0] == 1:
                        theta = theta.expand(x_obs.shape[0], -1)
                Iinv = fisher_inv_sqrtdet(theta[0].detach().cpu().numpy())  # θ fixed across batch
                U = score(theta, x_obs)                                    # (n,d)
                return torch.einsum('ni,ij,nj->n', U, Iinv, U)             # Rao stat

            def critical_value_fn(theta: torch.Tensor, alpha: float) -> torch.Tensor:
                return torch.tensor(float(chi2.ppf(alpha, df=d)), device=dev)

            arch = {"method": "ScoreCD", "variant": "rao", "fisher_n": fisher_n,
                    "flow_class": type(flow).__name__}

        else:  # variant == "cal"
            theta_cal, x_cal = simulator.sample(int(config["n_train_quantile"]), rngs.eval)
            stat_cal = (score(theta_cal, x_cal) ** 2).sum(-1)              # ‖U‖²
            alpha_grid: List[float] = list(config["alpha_grid"])
            qnet = MultiQuantileMLP(input_dim=d, hidden=self.quantile_hidden,
                                    depth=self.quantile_depth, n_quantiles=len(alpha_grid)).to(dev)
            train_multi_quantile_head(qnet, theta_cal.to(dev), stat_cal.to(dev),
                                      alpha_grid, config, dev)
            qnet.eval()
            a2idx = {a: k for k, a in enumerate(alpha_grid)}

            def test_stat_fn(theta: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
                theta = theta.to(dev); x_obs = x_obs.to(dev)
                if theta.shape[0] != x_obs.shape[0]:
                    if x_obs.shape[0] == 1:
                        x_obs = x_obs.expand(theta.shape[0], -1)
                    elif theta.shape[0] == 1:
                        theta = theta.expand(x_obs.shape[0], -1)
                return (score(theta, x_obs) ** 2).sum(-1)

            def critical_value_fn(theta: torch.Tensor, alpha: float) -> torch.Tensor:
                with torch.no_grad():
                    return qnet(theta.to(dev))[:, a2idx[alpha]]

            arch = {"method": "ScoreCD", "variant": "cal", "alpha_grid": alpha_grid,
                    "flow_class": type(flow).__name__}

        procedure = CriticalValueProcedure(
            test_stat_fn=test_stat_fn, critical_value_fn=critical_value_fn,
            d_theta=d, theta_range=simulator.theta_range,
        )
        arch["loss_history_tail"] = losses[-min(100, len(losses)):]
        return TrainedModel(procedure=procedure, state_dict={"flow": flow.state_dict()},
                            final_loss=float(losses[-1]), n_steps=int(config["n_steps"]),
                            wall_clock_sec=wall, arch_metadata=arch)

    def n_params(self) -> dict:
        backbone = sum(p.numel() for p in self.flow.parameters())
        return {"backbone": backbone, "head": 0, "calibration_stage": 0,
                "total": backbone, "kind": "flow"}
