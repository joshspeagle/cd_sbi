# CD-SBI (μ, σ²) — M2: Stage-B Learned Summary + fit() Extension

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the oracle `SufficientStatConditioner` with a *learned* permutation-invariant summary `DeepSetsConditioner` (`s_φ(X): ℝ¹⁰ → ℝ²`), extend `CDSBIRunner.fit` to train conditioner parameters jointly with the flow, and confirm via a training smoke that the learned-summary pivot trains and the summary recovers a reparameterization of the sufficient statistic. This is the **infrastructure** for Stage B; the cheat/regularity *investigation and verdict* (sufficiency recovery + floor integrity + full calibration across seeds, escalation to an invertible summary if the cheat appears) is **M3**.

**Architecture:** `DeepSetsConditioner` is an `nn.Module` implementing the `Conditioner` protocol: `s_φ(X) = ρ( mean_i φ(X_i) )` with `φ: ℝ→ℝ^h`, `ρ: ℝ^h→ℝ²`, mean-pooled over the exchangeable `n_iid` axis. It bakes in **device-1** from the spec (§B.3): `log_det_contrib = 0` (the summary is a feature map, not part of the change-of-variables) + **running feature standardization** (removes the scale/collapse cheat). The existing `SingleIndexMonotoneFlow` is reused unchanged with its fixed per-coord signs — the learned summary has the freedom to orient `∂feat/∂X` so the fixed-sign flow can represent the truth (whether training finds the right scale/location feature *ordering* is the M3 question). `CDSBIRunner.fit` is extended to optimize, grad-clip, device-move, and train/eval the conditioner's parameters when it is an `nn.Module` (no-op for `Identity`/frozen/`SufficientStatConditioner`).

**Tech Stack:** PyTorch (`nn.Module`, mean-pool, BatchNorm-style running stats), Hydra, pytest, numpy/scipy.

**Spec:** `docs/superpowers/specs/2026-05-29-cd-sbi-unknown-mean-variance-design.md` §B.1 (DeepSets), §B.2 (fit change), §B.3 device-1. §B.3 escalation + §B.4 full verdict = M3.

---

## Conventions / decisions locked for M2

- **DeepSets, mean-pool**, fixed `n_iid=10`, `d_out=2`. (Transformer/variable-`n_iid` is explicitly out of scope — the *next* milestone this unlocks.)
- **Device-1 baked in:** `log_det_contrib = 0`; features running-standardized (BatchNorm1d-style, affine-free) so the network cannot exploit feature scale. The two together prevent both the density-inflation cheat (no log-det term to inflate) and scale drift (the flow sees well-scaled features).
- **Flow unchanged.** Reuse `SingleIndexMonotoneFlow` with `theta_signs=(+1,+1)`, `feat_signs=(−1,−1)` injected from the simulator. Rationale: `∂r/∂X = (∂r/∂feat)·(∂feat/∂X)`; the learned `s_φ` chooses `∂feat/∂X`'s sign, so fixed flow signs are not an obstruction to representing the truth. The feature *ordering* (coord-0 ↔ scale, coord-1 ↔ location) is a soft prior the network must satisfy — an **M3 calibration risk**, surfaced by the M2 smoke, not resolved here.
- **fit() change is the only edit to existing training code** (spec §B.2): include `conditioner.parameters()` in the optimizer + grad-clip, and `.to(device)` / `.train()` / `.eval()` the conditioner, **guarded on `isinstance(conditioner, nn.Module)`**.
- **M2 deliverable = it trains + the summary recovers a smooth reparam of `(log s², X̄)`** (correlation check). Full floor-integrity / cross-seed calibration / cheat verdict = M3.

---

## File structure

```
NEW
  src/cdsbi/conditioners/deep_sets.py             # DeepSetsConditioner (nn.Module)
  tests/unit/test_deep_sets_conditioner.py
  configs/conditioner/deep_sets.yaml
  configs/experiment/mu_sigma_stage_b.yaml        # deep_sets conditioner variant
  tests/integration/test_mu_sigma_stage_b_smoke.py

MODIFY
  src/cdsbi/methods/cd_sbi.py                      # fit(): train conditioner params (nn.Module-guarded)
  tests/unit/test_cd_sbi_runner.py                 # (or nearest existing) fit-trains-conditioner test
```

No change to `SingleIndexMonotoneFlow`, the simulator, the diagnostics, or `run.py`. `run.py`'s conditioner dispatch (≈ line 201) already builds any non-identity `cfg.conditioner` generically from its config keys — so `deep_sets` works with `n_iid` hardcoded in the YAML (like `sufficient_stat`); Task 3 only verifies this.

---

## Task 1: `DeepSetsConditioner`

**Files:**
- Create: `src/cdsbi/conditioners/deep_sets.py`
- Test: `tests/unit/test_deep_sets_conditioner.py`

First READ `src/cdsbi/conditioners/base.py` (the `Conditioner` protocol), `src/cdsbi/conditioners/sufficient_stat.py` (the encode contract + `_SUFFICIENT_STAT_LOG_DET_CONST`), and `src/cdsbi/flows/single_index_monotone.py`'s `_tanh_mlp` helper (match the MLP idiom).

- [ ] **Step 1: Write failing tests**

```python
"""DeepSetsConditioner: permutation-invariant learned summary, device-1 (log_det=0)."""
import torch


def _cond(n_iid=10, hidden=32):
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    return DeepSetsConditioner(n_iid=n_iid, d_out=2, hidden=hidden)


def test_encode_shapes_and_zero_logdet():
    cond = _cond()
    x = torch.randn(16, 10)
    feats, log_det = cond.encode(x)
    assert feats.shape == (16, 2)
    assert log_det.shape == (16,)
    assert torch.allclose(log_det, torch.zeros(16))      # device-1: summary log-det is 0


def test_permutation_invariant_over_iid_axis():
    torch.manual_seed(0)
    cond = _cond().eval()                                 # eval → fixed running stats
    x = torch.randn(8, 10)
    perm = torch.randperm(10)
    f1, _ = cond.encode(x)
    f2, _ = cond.encode(x[:, perm])
    assert torch.allclose(f1, f2, atol=1e-5)              # exchangeable over n_iid


def test_has_trainable_params_and_is_module():
    cond = _cond()
    assert isinstance(cond, torch.nn.Module)
    assert cond.n_params() > 0


def test_running_standardization_eval_uses_buffers():
    cond = _cond()
    x = torch.randn(64, 10)
    cond.train(); cond.encode(x)                          # updates running stats
    cond.eval()
    f1, _ = cond.encode(x); f2, _ = cond.encode(x)
    assert torch.allclose(f1, f2)                         # eval is deterministic
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_deep_sets_conditioner.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
"""DeepSetsConditioner: permutation-invariant learned summary s_φ(X)=ρ(mean_i φ(X_i)).

Stage-B learned summary for the unknown-(μ,σ²) target. Device-1 (spec §B.3):
log_det_contrib = 0 (the summary is a feature map, not part of the NF-MLE
change-of-variables) + running feature standardization (BatchNorm1d-style,
affine-free) so the network cannot exploit feature scale to cheat the objective.
DeepSets with mean-pooling can represent (X̄, X̄²)→(X̄, s²) exactly, so a
sufficient summary is within its class — whether training finds it is the
Stage-B (M3) question.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _tanh_mlp(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Sequential:
    assert depth >= 1
    layers: list = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class DeepSetsConditioner(nn.Module):
    def __init__(self, n_iid: int, d_out: int = 2, hidden: int = 64, depth: int = 2,
                 momentum: float = 0.1):
        super().__init__()
        self.n_iid = n_iid
        self.d_out = d_out
        self.momentum = momentum
        self.phi = _tanh_mlp(1, hidden, hidden, depth)        # per-element ℝ→ℝ^h
        self.rho = _tanh_mlp(hidden, hidden, d_out, depth)    # pooled ℝ^h→ℝ^{d_out}
        # running standardization of the d_out features (affine-free; removes the
        # scale/collapse cheat). Updated from detached batch stats in train mode.
        self.register_buffer("running_mean", torch.zeros(d_out))
        self.register_buffer("running_var", torch.ones(d_out))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[-1] == self.n_iid, (
            f"DeepSetsConditioner expected {self.n_iid} iid obs per row, got {x.shape[-1]}"
        )
        n, m = x.shape
        h = self.phi(x.reshape(n * m, 1)).reshape(n, m, -1).mean(dim=1)   # (n, hidden)
        feats = self.rho(h)                                              # (n, d_out)
        if self.training:
            batch_mean = feats.mean(dim=0).detach()
            batch_var = feats.var(dim=0, unbiased=False).detach()
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * batch_var)
            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var
        feats = (feats - mean) / torch.sqrt(var + 1e-5)
        log_det = torch.zeros(n, dtype=x.dtype, device=x.device)
        return feats, log_det

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_deep_sets_conditioner.py -v` → 4 pass.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/conditioners/deep_sets.py tests/unit/test_deep_sets_conditioner.py
git commit -m "feat(cond): DeepSetsConditioner — permutation-invariant learned summary (device-1)"
```

---

## Task 2: `fit()` trains conditioner parameters

**Files:**
- Modify: `src/cdsbi/methods/cd_sbi.py`
- Test: `tests/unit/test_cd_sbi_runner.py` (create if absent; otherwise the nearest runner test file — check `tests/unit/` for an existing CDSBIRunner test and append)

READ `src/cdsbi/methods/cd_sbi.py` `fit()` first. The current code: builds `opt` over `self.flow.parameters()` only (the `if opt_name ==` block), grad-clips `self.flow.parameters()`, and `.to`/`.train`/`.eval`s only `self.flow`. Extend each to include the conditioner when it is an `nn.Module`.

- [ ] **Step 1: Write the failing test**

```python
"""CDSBIRunner.fit trains conditioner parameters when the conditioner is an nn.Module."""
import copy
import torch


def test_fit_updates_deep_sets_conditioner_params():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi import CDSBIRunner

    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=16)
    cond = DeepSetsConditioner(n_iid=sim.n_iid, d_out=2, hidden=16)
    before = copy.deepcopy({k: v.clone() for k, v in cond.state_dict().items()
                            if v.dtype.is_floating_point})
    runner = CDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss())
    runner.fit(simulator=sim, config={"lr": 1e-3, "batch_size": 128, "n_steps": 30,
                                      "n_train": 2000, "optimizer": "adam",
                                      "fresh_batch": False}, seed=0)
    after = cond.state_dict()
    # at least one phi/rho weight tensor must have changed
    changed = any(not torch.allclose(before[k], after[k])
                  for k in before if k.startswith(("phi", "rho")))
    assert changed, "conditioner parameters did not update during fit()"


def test_fit_still_works_with_frozen_conditioner():
    # SufficientStatConditioner is a plain object (not nn.Module) — fit must no-op it.
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.sufficient_stat import SufficientStatConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi import CDSBIRunner
    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=16)
    runner = CDSBIRunner(flow=flow, conditioner=SufficientStatConditioner(n_iid=sim.n_iid),
                         loss=NFMLELoss())
    trained = runner.fit(simulator=sim, config={"lr": 1e-3, "batch_size": 128,
                         "n_steps": 10, "n_train": 1000, "optimizer": "adam",
                         "fresh_batch": False}, seed=0)
    assert trained.final_loss == trained.final_loss   # ran without error (no NaN)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_cd_sbi_runner.py::test_fit_updates_deep_sets_conditioner_params -v` → FAIL (params unchanged: conditioner not in optimizer).

- [ ] **Step 3: Implement the fit() extension**

In `cd_sbi.py` `fit()`:

(a) After `self.flow.to(self.device)` / `self.flow.train()`, add conditioner device+train (guarded):
```python
        cond_is_module = isinstance(self.conditioner, torch.nn.Module)
        if cond_is_module:
            self.conditioner.to(self.device)
            self.conditioner.train()
```

(b) Build the trainable-parameter list once and use it for BOTH the optimizer and grad-clip. Replace `self.flow.parameters()` in the optimizer constructors and the `clip_grad_norm_` call with `trainable_params`:
```python
        trainable_params = list(self.flow.parameters())
        if cond_is_module:
            trainable_params += list(self.conditioner.parameters())
```
Then each optimizer uses `trainable_params` (e.g. `torch.optim.Adam(trainable_params, lr=lr, betas=betas)`), and the clip becomes:
```python
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
```

(c) After the training loop, before building `pivot_fn`, switch the conditioner to eval (guarded):
```python
        if cond_is_module:
            self.conditioner.eval()
```

(d) Confirm `arch_metadata` records the conditioner — add `"conditioner_class": type(self.conditioner).__name__` and `"conditioner_params": self.conditioner.n_params()` to `arch_meta` if not already present (check; helps M3 analysis). Keep changes minimal.

Leave the `pivot_fn` closure as-is — it already calls `conditioner.encode(x)`, which now uses the trained params + eval-mode running stats.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_cd_sbi_runner.py -v` → both pass. Then a regression check that Stage-A (oracle) still trains: `pytest tests/integration/test_mu_sigma_smoke.py -q -m intensive` should still pass (the frozen `SufficientStatConditioner` is not an `nn.Module`, so the guard makes fit() behave exactly as before).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/methods/cd_sbi.py tests/unit/test_cd_sbi_runner.py
git commit -m "feat(fit): train conditioner params jointly when conditioner is an nn.Module"
```

---

## Task 3: configs + wiring check

**Files:**
- Create: `configs/conditioner/deep_sets.yaml`, `configs/experiment/mu_sigma_stage_b.yaml`
- Verify: `src/cdsbi/experiments/run.py` builds the learned conditioner (no edit expected)

- [ ] **Step 1: Write `configs/conditioner/deep_sets.yaml`**

```yaml
name: deep_sets
_target_: cdsbi.conditioners.deep_sets.DeepSetsConditioner
# run.py builds conditioners GENERICALLY from this config (pops _target_/name,
# instantiates with the rest) — it does NOT inject n_iid. So hardcode it here,
# exactly as configs/conditioner/sufficient_stat.yaml does (n_iid: 10).
n_iid: 10
d_out: 2
hidden: ${budget.cdsbi_flow_hidden}
depth: 2
```

- [ ] **Step 2: Write `configs/experiment/mu_sigma_stage_b.yaml`** (mirrors `mu_sigma_replication` but swaps the conditioner)

```yaml
# @package _global_
defaults:
  - override /target: normal_mu_sigma
  - override /flow: single_index_monotone
  - override /conditioner: deep_sets
  - override /method: cd_sbi
  - override /budget: medium

# self-contained flow dispatch (cb1e08e guard), as in mu_sigma_replication
method:
  flow: single_index_monotone

experiment:
  name: mu_sigma_stage_b
  n_eval: 6000
  eval_thetas_interior:
    - [-0.69, -2.0]
    - [-0.69,  0.0]
    - [-0.69,  2.0]
    - [ 0.0,  -2.0]
    - [ 0.0,   0.0]
    - [ 0.0,   2.0]
    - [ 0.69, -2.0]
    - [ 0.69,  0.0]
    - [ 0.69,  2.0]
  eval_thetas_edge: []
  alpha_grid: [0.5, 0.68, 0.9, 0.95]
  n_eval_per_theta: 2000
  joint_mahalanobis_n_per_theta: 2000
```

- [ ] **Step 3: Verify the generic conditioner build works (NO run.py edit expected)**

`run.py`'s conditioner dispatch (≈ line 201–206) builds any non-identity conditioner **generically**: it pops `_target_`/`name` from `cfg.conditioner` and instantiates with the remaining keys — so `n_iid` (hardcoded in the YAML, Step 1) flows straight to the ctor. No `n_iid` injection exists or is needed. Verify the config composes and the conditioner instantiates:
```bash
python -c "
from hydra import initialize, compose
from cdsbi.experiments.run import _instantiate
from omegaconf import OmegaConf
with initialize(version_base=None, config_path='configs'):
    cfg = compose(config_name='config', overrides=['experiment=mu_sigma_stage_b'])
cc = OmegaConf.to_container(cfg.conditioner, resolve=True)
tgt = cc.pop('_target_'); cc.pop('name', None)
cond = _instantiate(tgt, **cc)
print('conditioner:', type(cond).__name__, '| n_params:', cond.n_params(),
      '| n_eval grid:', len(cfg.experiment.eval_thetas_interior))
"
```
Expected: `DeepSetsConditioner`, `n_params > 0`, grid `9`. (If for some reason the build path does NOT cover `deep_sets`, that is a surprise — STOP and report; do not edit run.py speculatively.)

- [ ] **Step 4: Commit**

```bash
git add configs/conditioner/deep_sets.yaml configs/experiment/mu_sigma_stage_b.yaml
git commit -m "config: deep_sets conditioner + mu_sigma_stage_b experiment"
```

---

## Task 4: Stage-B training smoke

**Files:**
- Create: `tests/integration/test_mu_sigma_stage_b_smoke.py`

A single-run smoke: train end-to-end with the learned summary and confirm (a) it trains (loss finite, decreasing vs init), (b) the learned summary recovers a smooth reparameterization of the oracle sufficient statistic `(log s², X̄)` — i.e. the 2 learned features are jointly highly correlated with `(log s², X̄)` (canonical-correlation / per-target best |corr|), and (c) a basic calibration sanity (joint Mahalanobis PIT not wildly off). The full cross-seed calibration + floor-integrity verdict is M3 — keep tolerances loose here (this is "does the learned path work at all", not the benchmark).

- [ ] **Step 1: Write the smoke**

```python
"""Stage-B smoke: CDSBI with a LEARNED DeepSets summary trains + recovers a
reparameterization of the sufficient statistic (log s², X̄). Loose bands — the
full Stage-B calibration/floor verdict is M3."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.intensive
def test_stage_b_learned_summary_trains_and_recovers_sufficiency():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi import CDSBIRunner

    torch.manual_seed(0)
    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=32, depth=2)
    cond = DeepSetsConditioner(n_iid=sim.n_iid, d_out=2, hidden=32, depth=2)
    runner = CDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss())
    config = {"lr": 3e-3, "batch_size": 256, "n_steps": 6000, "n_train": 10000,
              "optimizer": "adamw", "lr_schedule": "warmup_cosine",
              "warmup_steps": 300, "lr_min_ratio": 0.01, "fresh_batch": False}
    trained = runner.fit(simulator=sim, config=config, seed=0)

    # (a) trained: finite loss
    assert np.isfinite(trained.final_loss)

    # (b) sufficiency recovery: learned features vs oracle (log s², X̄)
    rng = np.random.default_rng(123)
    theta, x = sim.sample(4000, rng)
    cond.eval()
    with torch.no_grad():
        feats, _ = cond.encode(x.to(runner.device))
    feats = feats.cpu().numpy()
    xbar, s2 = sim._suff_stats(x)
    suff = np.column_stack([np.log(s2.squeeze(-1).numpy()), xbar.squeeze(-1).numpy()])  # (n,2)
    # each oracle coordinate should be well-predicted by the 2 learned features:
    # max |corr| of each suff coord with the learned feature set (simple, robust).
    def best_abscorr(target, F):
        return max(abs(np.corrcoef(target, F[:, j])[0, 1]) for j in range(F.shape[1]))
    corr_logs2 = best_abscorr(suff[:, 0], feats)
    corr_xbar = best_abscorr(suff[:, 1], feats)
    print(f"sufficiency recovery: |corr(log s², ·)|={corr_logs2:.3f}  |corr(X̄, ·)|={corr_xbar:.3f}")
    assert corr_logs2 > 0.9, f"learned summary lost σ²-information (corr {corr_logs2:.2f})"
    assert corr_xbar > 0.9, f"learned summary lost μ-information (corr {corr_xbar:.2f})"

    # (c) calibration sanity: joint Mahalanobis PIT at a central θ₀ ~ χ²₂
    from scipy.stats import kstest, chi2
    theta_0 = (0.0, 0.0)
    xv = sim.sample_x_given_theta(theta_0, 3000, np.random.default_rng(7))
    th = torch.tensor([[0.0, 0.0]], dtype=xv.dtype).expand(xv.shape[0], -1)
    with torch.no_grad():
        r = trained.procedure.pivot(th, xv).cpu().numpy()
    pit = chi2.cdf((r ** 2).sum(1), df=2)
    ks = kstest(pit, "uniform").statistic
    print(f"joint Mahalanobis KS (learned summary) = {ks:.3f}")
    assert ks < 0.10, f"learned-summary joint calibration KS {ks:.3f} too high (loose Stage-B sanity)"
```

- [ ] **Step 2: Run it (intensive; trains one model, minutes on GPU)**

Run: `pytest tests/integration/test_mu_sigma_stage_b_smoke.py -v -s -m intensive`

Interpretation:
- **Pass** → Stage-B infrastructure works; the learned summary recovers sufficiency and roughly calibrates. M3 then does the cross-seed verdict + cheat instrumentation.
- **Sufficiency-recovery fail** (corr < 0.9) → the learned summary is not finding `(log s², X̄)`. Report the corr values + final loss. This is a *finding*, not a tolerance to loosen — likely the feature-ordering / fixed-sign interaction flagged in the spec (M3 territory surfacing early). STOP and report for the controller.
- **Calibration fail but sufficiency OK** → the summary is sufficient but the pivot mis-orients; report KS + corr. Also a finding (the M3 ordering question). STOP and report.
- **Loss dips far below the entropy floor** (`NormalUnknownMeanVar().entropy_lower_bound()` ≈ 0.92) → the cheat (device-1 insufficient); report `final_loss` vs floor. STOP — escalation to the invertible summary (spec device-2) is M3.

Do NOT loosen tolerances or change device-1; report findings.

- [ ] **Step 3: Confirm fast suite green**

Run: `pytest -q` (the smoke is intensive → deselected).

- [ ] **Step 4: Commit** (commit the test even if step 2 surfaces a finding — mark DONE_WITH_CONCERNS and report)

```bash
git add tests/integration/test_mu_sigma_stage_b_smoke.py
git commit -m "test(intensive): Stage-B learned-summary smoke — trains + recovers sufficiency"
```

---

## Self-review

- **Spec coverage (§B.1/§B.2/§B.3-device-1):** `DeepSetsConditioner` mean-pool permutation-invariant summary (Task 1) ✓; `fit()` trains conditioner params, nn.Module-guarded (Task 2) ✓; device-1 (log_det=0 + running standardization) baked into the conditioner (Task 1) ✓; configs + wiring (Task 3) ✓; training smoke validating it trains + recovers sufficiency (Task 4) ✓. **M3 (deferred):** cross-seed floor-integrity verdict, full calibration (PivotRMSE/MarginalPIT/Coverage/MarginalCDRecovery on the learned summary), cheat instrumentation, escalation to device-2 (invertible summary) if the cheat appears.
- **Placeholder scan:** every step has runnable code + commands + expected output; no TBD.
- **Type consistency:** `DeepSetsConditioner.encode(x) -> (feats (n,2), log_det (n,))` matches the `Conditioner` protocol and the flow's `forward(theta, context=feats)`; `n_params()` present; the `fit()` `trainable_params` list is used identically in optimizer + grad-clip; configs reference `cdsbi.conditioners.deep_sets.DeepSetsConditioner` with the ctor args the class defines (`n_iid`, `d_out`, `hidden`, `depth`).
- **Risk — feature ordering vs fixed flow signs.** The flow's `feat_signs=(−1,−1)` + autoregressive ordering assume coord-0 ↔ scale, coord-1 ↔ location. The learned summary must produce features in that orientation; the network's freedom over `∂feat/∂X` makes the truth *representable*, but training may land in a mis-ordered optimum. Task 4's sufficiency + calibration checks surface this; resolving it (e.g. permutation-robust verification, or learnable flow signs) is M3. Flagged, not pre-solved.
- **Risk — running-standardization stability.** Detached batch stats shift the feature definition during training (BatchNorm-style); standard and stable in practice, but if the smoke shows loss oscillation, the momentum (0.1) or a warmup on the stats is the knob — noted for M3, not expected to bite at n_steps=6000.
- **Known follow-on:** transformer/variable-`n_iid` summary encoders remain out of scope (the next milestone this unlocks).
