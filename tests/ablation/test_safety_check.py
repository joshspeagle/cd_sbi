"""Safety check: CDSBIRunner refuses an R1-only flow unless allow_ablation=True."""
from __future__ import annotations

import pytest

from cdsbi.flows.joint_umnn import JointUMNNFlow
from cdsbi.conditioners.mlp import MLPConditioner
from cdsbi.losses.nfmle import NFMLELoss
from cdsbi.losses.base import MonotonicityMismatchError
from cdsbi.methods.cd_sbi import CDSBIRunner


@pytest.mark.ablation
def test_runner_refuses_r1_only_flow_by_default():
    """allow_ablation defaults to False; constructing CDSBIRunner with a
    flow that lacks R2 must raise MonotonicityMismatchError."""
    flow = JointUMNNFlow(hidden=8)
    conditioner = MLPConditioner(input_dim=5, output_dim=1, mode="frozen_sum")
    loss = NFMLELoss()
    with pytest.raises(MonotonicityMismatchError):
        CDSBIRunner(flow=flow, conditioner=conditioner, loss=loss)


@pytest.mark.ablation
def test_runner_allows_r1_only_flow_with_explicit_allow_ablation():
    """allow_ablation=True bypasses the check — the ablation experiment needs this."""
    flow = JointUMNNFlow(hidden=8)
    conditioner = MLPConditioner(input_dim=5, output_dim=1, mode="frozen_sum")
    loss = NFMLELoss()
    runner = CDSBIRunner(
        flow=flow, conditioner=conditioner, loss=loss, allow_ablation=True,
    )
    assert runner.allow_ablation is True
