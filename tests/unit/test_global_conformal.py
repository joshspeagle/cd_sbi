"""CPF plan step 4 — global split-conformal (cpf/conformal.py).

Conventions under test are PINNED by the 2026-06-10 audit: score s = F_hat(T) in
[0,1] (large = extreme); threshold = ceil(coverage*(n+1))-th smallest score; set
keeps {s <= q_hat}. An alpha-lower-quantile slip here would yield ~(1-coverage)
coverage — the classic conformal tail bug these tests exist to catch.
"""
import math

import pytest
import torch

from cdsbi.methods.cpf.conformal import GlobalConformal, conformal_quantile


# --- pure math (worked examples; atol exact) ---------------------------------

def test_worked_example_n9_cov09():
    # n=9, coverage=0.9 -> ceil(0.9*10)=9 -> the 9th smallest (= the max element).
    scores = torch.linspace(0.1, 0.9, 9)
    assert conformal_quantile(scores, 0.9) == scores.max().item()
    assert conformal_quantile(scores, 0.9) == pytest.approx(0.9, abs=1e-6)


def test_worked_example_n9_cov05():
    # n=9, coverage=0.5 -> ceil(0.5*10)=5 -> the 5th smallest.
    scores = torch.linspace(0.1, 0.9, 9)
    assert conformal_quantile(scores, 0.5) == pytest.approx(0.5, abs=1e-7)


def test_index_exceeds_n_gives_inf():
    # n=3, coverage=0.9 -> ceil(3.6)=4 > 3 -> +inf (set = everything).
    assert conformal_quantile(torch.tensor([0.1, 0.2, 0.3]), 0.9) == float("inf")


def test_order_invariance_and_shape():
    a = torch.tensor([0.9, 0.1, 0.5, 0.3, 0.7])
    b = torch.sort(a).values.reshape(5, 1)
    assert conformal_quantile(a, 0.5) == conformal_quantile(b, 0.5)


def test_threshold_monotone_in_coverage():
    gc = GlobalConformal(torch.rand(101, generator=torch.Generator().manual_seed(0)))
    qs = [gc.threshold(c) for c in (0.5, 0.68, 0.9, 0.95)]
    assert all(qs[i] <= qs[i + 1] for i in range(len(qs) - 1))


def test_nested_sets_on_fixed_field():
    # One global threshold per coverage on a FIXED field -> nesting by
    # construction (spec §4.3.2 premise, post-conformal).
    gen = torch.Generator().manual_seed(1)
    gc = GlobalConformal(torch.rand(199, generator=gen))
    field = torch.rand(500, generator=gen)  # s(theta) on a theta-grid
    inside_50 = field <= gc.threshold(0.5)
    inside_90 = field <= gc.threshold(0.9)
    assert bool((inside_50 & ~inside_90).sum() == 0)  # C_0.5 subset of C_0.9


# --- statistical (seeded; two-sided band) -------------------------------------

def test_coverage_band_two_sided_uniform_scores():
    """Exchangeable continuous scores: P(s_test <= q_hat) in
    [coverage, coverage + 1/(n+1)] — assert BOTH bounds (uniform scores sit at
    the lower edge; a one-sided test would not catch an anti-conservative bug).
    Exact coverage here = ceil(0.9*50)/50 = 0.90."""
    gen = torch.Generator().manual_seed(42)
    n_cal, reps, coverage = 49, 4000, 0.9
    hits = 0
    for _ in range(reps):
        s = torch.rand(n_cal + 1, generator=gen)
        q = conformal_quantile(s[:n_cal], coverage)
        hits += int(s[n_cal] <= q)
    emp = hits / reps
    mc_tol = 3 * math.sqrt(coverage * (1 - coverage) / reps)  # ~0.014
    assert emp >= coverage - mc_tol
    assert emp <= coverage + 1.0 / (n_cal + 1) + mc_tol


def test_miscalibrated_fhat_repaired_but_wider():
    """The audit-mandated wrong-F-hat probe: MONOTONE-but-miscalibrated (NOT a
    constant — a constant passes for the wrong reason: all scores equal -> set =
    everything). Setup: T|theta ~ Exp(rate=theta), theta in {0.5, 4.0} (50/50);
    the miscalibrated F-hat ignores theta (rate 1.0). Assertions:
      (i)  marginal coverage is REPAIRED (>= coverage - MC tol): scores remain
           iid hence exchangeable, so the conformal floor holds regardless;
      (ii) the acceptance region is STRICTLY WIDER than under the true per-theta
           F (the width cost of miscalibration — the spec's headline finding).
    Analytic widths at coverage 0.9: mis ~ 6.4 vs true ~ 5.2 (ratio ~1.24)."""
    gen = torch.Generator().manual_seed(7)
    coverage, n_cal, n_test = 0.9, 2000, 4000
    rates = torch.tensor([0.5, 4.0])

    def draw(n):
        th = rates[torch.randint(0, 2, (n,), generator=gen)]
        t = -torch.log(1 - torch.rand(n, generator=gen)) / th  # Exp(rate=th)
        return th, t

    f_mis = lambda t: 1 - torch.exp(-1.0 * t)          # ignores theta
    f_true = lambda t, th: 1 - torch.exp(-th * t)      # exact per-theta CDF

    th_c, t_c = draw(n_cal)
    q_mis = conformal_quantile(f_mis(t_c), coverage)
    q_true = conformal_quantile(f_true(t_c, th_c), coverage)

    # (i) repair: fresh pairs, marginal coverage of {f_mis(T) <= q_mis}.
    th_t, t_t = draw(n_test)
    emp = (f_mis(t_t) <= q_mis).float().mean().item()
    mc_tol = 3 * math.sqrt(coverage * (1 - coverage) / n_test)
    assert emp >= coverage - mc_tol

    # (ii) width cost: acceptance region in t per theta is {t <= F^{-1}(q)}.
    width_mis = sum(-math.log(1 - q_mis) / 1.0 for _ in rates)          # same t-hat both theta
    width_true = sum(-math.log(1 - q_true) / float(r) for r in rates)   # exact per-theta
    assert width_mis > 1.1 * width_true
