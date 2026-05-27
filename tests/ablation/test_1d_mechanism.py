"""1D direct-construction mechanism test:

Train R1-only JointUMNN1DFlow on LocationNormal1D. Verify two things:
(1) final loss falls below the closed-form entropy ½ log(2πe) of the
1D normal, AND (2) ∂r/∂x is negative somewhere in the support (the
literal folding).
"""
from __future__ import annotations

import math

import pytest
import torch

from cdsbi.flows.joint_umnn_1d import JointUMNN1DFlow
from cdsbi.conditioners.identity import Identity
from cdsbi.losses.nfmle import NFMLELoss
from cdsbi.methods.cd_sbi import CDSBIRunner
from cdsbi.simulators.location_normal_1d import LocationNormal1D


@pytest.mark.ablation
def test_1d_r1_only_flow_folds_in_x():
    # Seed before flow init so the model's weights are deterministic across
    # runs (CDSBIRunner.fit re-seeds via seed_everything, but that runs
    # AFTER flow construction so it doesn't cover model init RNG).
    torch.manual_seed(0)
    sim = LocationNormal1D()
    H_floor = 0.5 * math.log(2 * math.pi * math.e)  # 1.4189...
    # Set theta_ref to the lower endpoint of the proposal support [-7, 7]
    flow = JointUMNN1DFlow(hidden=32, theta_ref=sim.theta_range[0])
    conditioner = Identity()
    loss = NFMLELoss()
    runner = CDSBIRunner(
        flow=flow, conditioner=conditioner, loss=loss, allow_ablation=True,
    )
    # The 1D direct-construction setting is closer to the entropy floor than
    # the ExpRate case (because there's no sufficient-statistic reduction to
    # exploit), so we need deeper convergence (8000 steps) + warmup-cosine
    # LR + larger hidden to clearly cross below the floor.
    config = {
        "lr": 5e-3,
        "batch_size": 512,
        "n_steps": 8000,
        "n_train": 30000,
        "optimizer": "adamw",
        "fresh_batch": False,
        "lr_schedule": "warmup_cosine",
        "warmup_steps": 200,
        "lr_min_ratio": 0.01,
    }
    trained = runner.fit(simulator=sim, config=config, seed=0)
    # Check 1: loss below the entropy lower bound for N(θ, 1). The §3.5
    # mechanism creates loss oscillations, so use the min over the
    # last-100-step tail as the robust signature — captures whether
    # the flow ever crossed the floor in the converged regime.
    tail_min = float(min(trained.arch_metadata["loss_history_tail"]))
    assert tail_min < H_floor - 0.02, (
        f"tail_min={tail_min:.3f} (final={float(trained.final_loss):.3f}) did NOT fall below "
        f"H_floor={H_floor:.3f}; 1D R2-ablation mechanism failed to reproduce"
    )
    # Check 2: the trained flow's ∂r/∂x is negative somewhere on the support
    # (this is the literal folding — non-monotone in X). Place test tensors
    # on the same device as the trained flow (CDSBIRunner moves it to GPU
    # when available).
    device = next(flow.parameters()).device
    theta_test = torch.full((200, 1), 0.0, device=device)
    x_test = torch.linspace(-3.0, 3.0, 200, device=device).view(-1, 1).requires_grad_(True)
    r, _ = flow(theta_test, context=x_test)
    grad_x = torch.autograd.grad(r.sum(), x_test, retain_graph=False)[0]
    assert (grad_x < 0).any(), (
        f"trained R1-only flow did NOT fold in X — min(∂r/∂X)={float(grad_x.min()):.4f} >= 0; "
        f"expected at least one point with negative ∂r/∂X"
    )
