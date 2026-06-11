"""Hardening items 3 + 4: Score-CD-cal budget accounting and the flow-dispatch trap.

Item 3: the "cal" variant trains a MultiQuantileMLP critical-value head; it must
count against the parameter budget exactly as LF2I-BFF's identical head does
(previously reported as calibration_stage: 0 while LF2I-BFF's was counted — an
asymmetric, reviewer-visible accounting gap).

Item 4: when method.flow != the Hydra /flow group, _build_flow builds the METHOD
label and silently ignores the experiment-level override (the cb1e08e trap, which
produced wrong-architecture §8.4 runs while the index claimed otherwise). The
fall-through must now warn loudly; the index additionally records the BUILT class.
"""
import logging

import torch
from omegaconf import OmegaConf

from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.methods.lf2i import MultiQuantileMLP
from cdsbi.methods.score_cd import ScoreCDRunner


def _tiny_flow():
    return MAFAdapter(features=2, context_features=2, hidden=8, num_layers=2)


def test_cal_counts_quantile_head():
    runner = ScoreCDRunner(_tiny_flow(), variant="cal", quantile_hidden=16, quantile_depth=2)
    np = runner.n_params(d_theta=2, alpha_grid_len=4)
    direct = sum(p.numel() for p in MultiQuantileMLP(
        input_dim=2, hidden=16, depth=2, n_quantiles=4).parameters())
    assert np["calibration_stage"] == direct > 0
    assert np["total"] == np["backbone"] + direct
    assert np["kind"] == "two_stage_score"


def test_rao_has_no_calibration_stage():
    runner = ScoreCDRunner(_tiny_flow(), variant="rao")
    np = runner.n_params(d_theta=2, alpha_grid_len=4)
    assert np["calibration_stage"] == 0
    assert np["total"] == np["backbone"]
    assert np["kind"] == "flow"


def test_cal_head_at_budget_width_is_small_fraction():
    """With the budget-tuned width (quantile_hidden=16, as the method yamls
    interpolate from ${budget.quantile_hidden}), the head must be a small
    fraction of a medium backbone — i.e. counting it keeps the ±15% band
    realistic rather than blowing the budget."""
    flow = MAFAdapter(features=2, context_features=2, hidden=32, num_layers=2)
    runner = ScoreCDRunner(flow, variant="cal", quantile_hidden=16, quantile_depth=2)
    np = runner.n_params(d_theta=2, alpha_grid_len=4)
    assert np["calibration_stage"] / np["backbone"] < 0.15


class _StubSim:
    d_theta = 1
    d_x = 1
    theta_range = (-7.0, 7.0)


def _trap_cfg(method_flow: str, hydra_flow: str):
    return OmegaConf.create({
        "method": {"name": "cd_sbi", "flow": method_flow},
        "flow": {"name": hydra_flow, "_target_": "unused", "depth": 2},
        "budget": {"cdsbi_flow_hidden": 8, "doubly_monotone_hidden": 8},
    })


def test_dispatch_trap_warns_loudly(caplog):
    """method.flow='additive_umnn' with /flow group 'doubly_monotone': the
    experiment override is ignored (additive_umnn is built) — must WARN."""
    from cdsbi.experiments.run import _build_flow

    cfg = _trap_cfg("additive_umnn", "doubly_monotone")
    with caplog.at_level(logging.WARNING, logger="cdsbi.experiments.run"):
        flow = _build_flow(cfg, _StubSim())
    assert type(flow).__name__ == "AdditiveFlow1D"  # the method label won
    assert any("FLOW DISPATCH TRAP" in r.message for r in caplog.records)


def test_no_warning_when_labels_agree(caplog):
    from cdsbi.experiments.run import _build_flow

    cfg = OmegaConf.create({
        "method": {"name": "cd_sbi", "flow": "additive_umnn"},
        "flow": {"name": "additive_umnn",
                 "_target_": "cdsbi.flows.additive.AdditiveFlow1D",
                 "hidden": 8, "depth": 2},
        "budget": {"cdsbi_flow_hidden": 8},
    })
    with caplog.at_level(logging.WARNING, logger="cdsbi.experiments.run"):
        _build_flow(cfg, _StubSim())
    assert not any("FLOW DISPATCH TRAP" in r.message for r in caplog.records)


def test_no_warning_for_maf_methods(caplog):
    """NPE/NLE/Score-CD carry method.flow='maf' and ignore the group by design
    — the trap warning must not fire for them."""
    from cdsbi.experiments.run import _build_flow

    cfg = OmegaConf.create({
        "method": {"name": "nle", "flow": "maf"},
        "flow": {"name": "additive_umnn", "_target_": "unused", "depth": 2},
        "budget": {"maf_hidden": 8},
    })
    with caplog.at_level(logging.WARNING, logger="cdsbi.experiments.run"):
        flow = _build_flow(cfg, _StubSim())
    assert type(flow).__name__ == "MAFAdapter"
    assert not any("FLOW DISPATCH TRAP" in r.message for r in caplog.records)
