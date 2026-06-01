import math
import numpy as np
import torch
from cdsbi.diagnostics.fisher_recovery import fisher_det_ratio
from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar


def _sufficient_h(sim, x):
    return sim.oracle_summary(x)


def _lossy_h(sim, x):
    xbar = x.mean(dim=-1, keepdim=True)
    return torch.cat([xbar, xbar], dim=-1)


def test_sufficient_summary_ratio_near_one():
    sim = NormalUnknownMeanVar()
    theta0 = (math.log(1.0), 0.0)
    ratio = fisher_det_ratio(sim, theta0, _sufficient_h, n_samples=40000, seed=0)
    assert ratio > 0.9, f"sufficient summary should give det ratio ≈ 1, got {ratio:.3f}"


def test_lossy_summary_ratio_below_one():
    sim = NormalUnknownMeanVar()
    theta0 = (math.log(1.0), 0.0)
    ratio = fisher_det_ratio(sim, theta0, _lossy_h, n_samples=40000, seed=0)
    assert ratio < 0.3, f"σ-dropping summary should lose Fisher info, got {ratio:.3f}"
