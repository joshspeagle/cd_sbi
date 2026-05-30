# CD-SBI M3.1 — Stage-B Arm I-B (two-stage freeze)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first Stage-B fix-arm: pretrain a DeepSets summary to predict θ directly (a bounded objective that recovers a sufficient statistic and canonicalizes feature order + orientation), freeze it, then train the unchanged fixed-sign monotone pivot on the frozen features — and show on the M3.0 shared harness that it calibrates as well as the Stage-A oracle WITHOUT the device-1 cheat.

**Architecture:** A `pretrain_summary` utility trains `s_φ:ℝ¹⁰→ℝ²` to minimize `E‖s_φ(X) − θ‖²` (the summary's two outputs ARE the θ-estimates). For the Gaussian target this recovers `(log σ̂, μ̂)`-type features, and because the closed-form pivot's signs relative to `(σ̂, μ̂)` are exactly `(+1,+1)/(−1,−1)` (and `r*_σ` is single-index in `log(s²/σ²)`), the frozen features land in the order and orientation the fixed-sign `SingleIndexMonotoneFlow` expects — resolving the frozen-summary ordering/orientation risk by construction. A `TwoStageCDSBIRunner` orchestrates: pretrain+freeze the conditioner, then run the standard pivot `fit()`. Because the summary is frozen, the NF-MLE loss in stage 2 cannot collapse it (loss → the fixed `H(feat|θ)`). The summary uses `standardize=False` (the freeze, not normalization, is what prevents the cheat here; predicting θ needs un-standardized outputs).

**Tech Stack:** PyTorch (`nn.Module`, Adam), Hydra, pytest, numpy/scipy. Reuses `DeepSetsConditioner`, `SingleIndexMonotoneFlow`, `CDSBIRunner`, and the M3.0 harness diagnostics.

**Spec:** `docs/superpowers/specs/2026-05-29-cd-sbi-stage-b-device-bakeoff-design.md` — Arm I-B (§3) + the shared harness (§2) + success criteria (§5).

---

## Conventions / decisions locked

- **Stage-1 objective = predict θ directly** (MSE), summary `d_out = d_theta = 2`. The summary's outputs are the θ-estimates; no separate read-out head. This canonicalizes feature **order** (output k ↔ θ_k: coord-0 ↔ log σ, coord-1 ↔ μ) and **orientation** (the pivot's fixed signs `(+1,+1)/(−1,−1)` match `r*` relative to `(σ̂, μ̂)`; verified: `r*_σ` ↓ in observed spread `σ̂≈feat_0` → `s_f0=−1` ✓; `r*_μ` ↓ in `X̄≈feat_1` → `s_f1=−1` ✓). InfoNCE is the documented fallback only if predict-θ under-recovers (it should not for this target).
- **`standardize=False`** for the I-B summary: the FREEZE prevents the cheat (frozen feat ⇒ NF-MLE → fixed `H(feat|θ)`), and predict-θ MSE needs raw (un-standardized) outputs. (Running standardization stays the default for the end-to-end device-1 path.)
- **Freeze = `requires_grad_(False)` + `.eval()`** on the pretrained summary. Stage-2 `fit()` must NOT re-enable training/running-stat updates on it → a small `fit()` robustness change: train the conditioner only when it has trainable params.
- **Orchestration via `TwoStageCDSBIRunner(CDSBIRunner)`** whose `fit()` pretrains+freezes `self.conditioner` (reading `config["pretrain"]`) then calls `super().fit()`. Wired through a `cd_sbi_two_stage` method config (`name: cd_sbi`, custom `runner_class`).
- **Judged on the M3.0 harness:** `SufficiencyRecovery` (Spearman > 0.9 on BOTH coords — incl. σ², the device-1 casualty), `FloorIntegrity` (no cheat: `final_loss > H − 0.10`), plus `Coverage`/`MarginalCDRecovery`/`JointMahalanobis` matching the Stage-A control bands.

---

## File structure

```
NEW
  src/cdsbi/methods/pretrain_summary.py            # pretrain_summary(simulator, summary, config) -> frozen summary
  src/cdsbi/methods/cd_sbi_two_stage.py            # TwoStageCDSBIRunner(CDSBIRunner)
  configs/method/cd_sbi_two_stage.yaml
  configs/experiment/mu_sigma_stage_b_freeze.yaml
  tests/unit/test_pretrain_summary.py
  tests/unit/test_two_stage_runner.py
  tests/intensive/test_replicate_mu_sigma_stage_b_freeze.py

MODIFY
  src/cdsbi/conditioners/deep_sets.py              # add standardize: bool flag
  src/cdsbi/methods/cd_sbi.py                      # fit(): train conditioner only if it has trainable params
  src/cdsbi/experiments/run.py                      # _fit_config: pass training.pretrain through for cd_sbi
  tests/unit/test_deep_sets_conditioner.py          # standardize=False test
```

No change to `SingleIndexMonotoneFlow`, the simulator, the loss, or the M3.0 diagnostics.

---

## Task 1: `DeepSetsConditioner` `standardize` flag

**Files:**
- Modify: `src/cdsbi/conditioners/deep_sets.py`
- Test: `tests/unit/test_deep_sets_conditioner.py` (append)

READ `src/cdsbi/conditioners/deep_sets.py` (the `encode` running-standardization block).

- [ ] **Step 1: Write the failing test (append)**

```python
def test_standardize_false_passes_raw_rho_output():
    import torch
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    torch.manual_seed(0)
    cond = DeepSetsConditioner(n_iid=10, d_out=2, hidden=16, standardize=False).eval()
    x = torch.randn(64, 10)
    feats, log_det = cond.encode(x)
    assert feats.shape == (64, 2) and torch.allclose(log_det, torch.zeros(64))
    # with standardization OFF, feats are NOT forced to ~unit-variance/zero-mean:
    # running buffers stay at init (mean 0, var 1) and are not applied.
    # equivalence check: raw rho output == encode output
    n, m = x.shape
    h = cond.phi(x.reshape(n * m, 1)).reshape(n, m, -1).mean(dim=1)
    raw = cond.rho(h)
    assert torch.allclose(feats, raw, atol=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_deep_sets_conditioner.py::test_standardize_false_passes_raw_rho_output -v`
Expected: FAIL (`standardize` is not a ctor arg).

- [ ] **Step 3: Implement**

Add `standardize: bool = True` to `__init__` (store `self.standardize = standardize`). In `encode`, guard the standardization block:
```python
        feats = self.rho(h)                                              # (n, d_out)
        if self.standardize:
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
```
(Keep the `running_mean`/`running_var` buffers registered unconditionally so `state_dict` is stable across the flag.)

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_deep_sets_conditioner.py -v`
Expected: all pass (existing 4 + 1 new).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/conditioners/deep_sets.py tests/unit/test_deep_sets_conditioner.py
git commit -m "feat(cond): DeepSetsConditioner standardize flag (off for the frozen predict-θ arm)"
```

---

## Task 2: `fit()` trains the conditioner only when it has trainable params

**Files:**
- Modify: `src/cdsbi/methods/cd_sbi.py`
- Test: `tests/unit/test_two_stage_runner.py` (create — a fit-respects-frozen test)

So a frozen (`requires_grad=False`, eval) conditioner stays frozen + eval through stage-2 `fit()` (no `.train()` re-enabling DeepSets running-stat updates, no params added to the optimizer). READ `cd_sbi.py` `fit()`: the `cond_is_module` block (`.to`/`.train()`), the `trainable_params` build, the grad-clip, and the post-loop `.eval()` (all added in M2).

- [ ] **Step 1: Write the failing test**

```python
"""fit() leaves a frozen conditioner frozen + in eval mode."""
import torch


def test_fit_does_not_train_frozen_conditioner():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi import CDSBIRunner

    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=16)
    cond = DeepSetsConditioner(n_iid=sim.n_iid, d_out=2, hidden=16, standardize=False)
    for p in cond.parameters():
        p.requires_grad_(False)
    before = {k: v.clone() for k, v in cond.state_dict().items() if v.dtype.is_floating_point}
    CDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss()).fit(
        simulator=sim, config={"lr": 1e-3, "batch_size": 64, "n_steps": 20,
                               "n_train": 1000, "optimizer": "adam", "fresh_batch": False}, seed=0)
    after = cond.state_dict()
    # frozen params unchanged AND the conditioner ends in eval mode
    assert all(torch.allclose(before[k], after[k]) for k in before
               if k.startswith(("phi", "rho")))
    assert not cond.training
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_two_stage_runner.py::test_fit_does_not_train_frozen_conditioner -v`
Expected: FAIL — `.train()` is currently called for any nn.Module conditioner (so `cond.training` is True), and any non-frozen behavior differs. (If it happens to pass because grads are zero, the `assert not cond.training` still fails since M2's code calls `.train()`.)

- [ ] **Step 3: Implement the robustness change**

In `fit()`, replace the M2 `cond_is_module` gating with a *trainable*-aware gate:
```python
        cond_is_module = isinstance(self.conditioner, torch.nn.Module)
        cond_trainable = cond_is_module and any(
            p.requires_grad for p in self.conditioner.parameters()
        )
        if cond_is_module:
            self.conditioner.to(self.device)
            self.conditioner.train() if cond_trainable else self.conditioner.eval()
```
Use `cond_trainable` (not `cond_is_module`) when building `trainable_params`:
```python
        trainable_params = list(self.flow.parameters())
        if cond_trainable:
            trainable_params += [p for p in self.conditioner.parameters() if p.requires_grad]
```
And the post-loop eval switch stays guarded on `cond_is_module` (a frozen module is already eval, re-calling `.eval()` is a harmless no-op):
```python
        if cond_is_module:
            self.conditioner.eval()
```
(The grad-clip already uses `trainable_params` — unchanged.)

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_two_stage_runner.py::test_fit_does_not_train_frozen_conditioner -v`
Then regression: `pytest tests/unit/test_cd_sbi_runner.py -v` (the M2 trainable-conditioner test must still pass — a non-frozen DeepSets is still trained).
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/methods/cd_sbi.py tests/unit/test_two_stage_runner.py
git commit -m "feat(fit): train conditioner only when it has trainable params (frozen-summary safe)"
```

---

## Task 3: `pretrain_summary` utility

**Files:**
- Create: `src/cdsbi/methods/pretrain_summary.py`
- Test: `tests/unit/test_pretrain_summary.py`

Trains the summary to predict θ (MSE), then freezes it. Returns the (same, now frozen) summary.

- [ ] **Step 1: Write the failing test**

```python
"""pretrain_summary: summary learns to predict θ, then is frozen."""
import numpy as np
import torch


def test_pretrain_summary_predicts_theta_and_freezes():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    from cdsbi.methods.pretrain_summary import pretrain_summary

    torch.manual_seed(0)
    sim = NormalUnknownMeanVar()
    summary = DeepSetsConditioner(n_iid=sim.n_iid, d_out=sim.d_theta, hidden=32, standardize=False)
    frozen = pretrain_summary(sim, summary, config={"lr": 3e-3, "n_steps": 1500,
                                                    "batch_size": 256}, seed=0, device="cpu")
    # frozen: no grad + eval
    assert not frozen.training
    assert all(not p.requires_grad for p in frozen.parameters())
    # predicts θ: encode(X) ≈ θ in MSE (the summary outputs are θ-estimates)
    rng = np.random.default_rng(1)
    theta, x = sim.sample(2000, rng)
    with torch.no_grad():
        pred, _ = frozen.encode(x)
    mse = float(((pred - theta) ** 2).mean())
    print(f"predict-θ MSE = {mse:.4f}")
    assert mse < 0.5, f"summary did not learn to predict θ (MSE {mse:.3f})"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_pretrain_summary.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `src/cdsbi/methods/pretrain_summary.py`**

```python
"""pretrain_summary: Stage-1 of the two-stage (freeze) Stage-B arm.

Trains a summary network s_φ: X → ℝ^{d_theta} to predict θ by MSE, then freezes it
(requires_grad=False + eval). Predicting θ directly gives features in the order and
orientation the fixed-sign monotone pivot expects (output k ↔ θ_k), so the frozen
features drop straight into the Stage-A pivot. The MSE objective is bounded below
by 0 and rewards retaining θ-information, so the summary cannot collapse (unlike the
end-to-end device-1 NF-MLE path).
"""
from __future__ import annotations

import torch

from cdsbi.device import get_device
from cdsbi.reproducibility.seeding import seed_everything


def pretrain_summary(simulator, summary, config: dict, seed: int, device: str = "auto"):
    dev = get_device(device)
    rngs = seed_everything(seed)
    summary.to(dev)
    summary.train()
    lr = config["lr"]
    n_steps = int(config["n_steps"])
    bs = int(config.get("batch_size", 256))
    opt = torch.optim.Adam([p for p in summary.parameters() if p.requires_grad], lr=lr)
    for _ in range(n_steps):
        theta, x = simulator.sample(bs, rngs.train)
        theta = theta.to(dev); x = x.to(dev)
        pred, _ = summary.encode(x)                 # (bs, d_theta)
        loss = ((pred - theta) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # freeze
    summary.eval()
    for p in summary.parameters():
        p.requires_grad_(False)
    return summary
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_pretrain_summary.py -v -s`
Expected: PASS; printed predict-θ MSE well under 0.5 (the summary recovers `(log σ̂, μ̂)`). If MSE is high (> 0.5), STOP and report — predict-θ under-recovers and the spec's InfoNCE fallback would be needed (do not loosen the threshold).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/methods/pretrain_summary.py tests/unit/test_pretrain_summary.py
git commit -m "feat(method): pretrain_summary — predict-θ Stage-1 for the two-stage freeze arm"
```

---

## Task 4: `TwoStageCDSBIRunner`

**Files:**
- Create: `src/cdsbi/methods/cd_sbi_two_stage.py`
- Test: `tests/unit/test_two_stage_runner.py` (append)

Orchestrates: pretrain+freeze the conditioner (Stage 1), then the standard pivot `fit()` (Stage 2). Reads stage-1 config from `config["pretrain"]`.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_two_stage_runner_pretrains_then_freezes_then_fits():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi_two_stage import TwoStageCDSBIRunner

    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=16)
    cond = DeepSetsConditioner(n_iid=sim.n_iid, d_out=sim.d_theta, hidden=16, standardize=False)
    runner = TwoStageCDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss(), device="cpu")
    trained = runner.fit(simulator=sim, config={
        "lr": 3e-3, "batch_size": 128, "n_steps": 30, "n_train": 1000,
        "optimizer": "adam", "fresh_batch": False,
        "pretrain": {"lr": 3e-3, "n_steps": 200, "batch_size": 128},
    }, seed=0)
    # conditioner frozen after fit; procedure exposes the (frozen) encode_fn
    assert all(not p.requires_grad for p in cond.parameters())
    assert trained.procedure.encode_fn is not None
    assert trained.arch_metadata.get("two_stage") is True
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_two_stage_runner.py::test_two_stage_runner_pretrains_then_freezes_then_fits -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `src/cdsbi/methods/cd_sbi_two_stage.py`**

```python
"""TwoStageCDSBIRunner: Stage-B Arm I-B (two-stage freeze).

Stage 1: pretrain the conditioner (a learned summary) to predict θ, then freeze.
Stage 2: the standard CDSBIRunner pivot fit() on the frozen features. Because the
summary is frozen, the NF-MLE loss cannot collapse it — the loss converges to the
fixed conditional-entropy floor H(feat|θ) (no device-1 cheat).
"""
from __future__ import annotations

from cdsbi.methods.cd_sbi import CDSBIRunner
from cdsbi.methods.pretrain_summary import pretrain_summary


class TwoStageCDSBIRunner(CDSBIRunner):
    def fit(self, simulator, config: dict, seed: int):
        pre_cfg = config.get("pretrain")
        if pre_cfg is None:
            raise ValueError("TwoStageCDSBIRunner requires config['pretrain'] (Stage-1 settings)")
        # Stage 1: pretrain + freeze the summary (predict-θ).
        self.conditioner = pretrain_summary(
            simulator, self.conditioner, config=pre_cfg, seed=seed,
            device=str(self.device),
        )
        # Stage 2: standard pivot fit on the frozen features.
        trained = super().fit(simulator=simulator, config=config, seed=seed)
        trained.arch_metadata["two_stage"] = True
        trained.arch_metadata["pretrain_steps"] = int(pre_cfg["n_steps"])
        return trained
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_two_stage_runner.py -v`
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/methods/cd_sbi_two_stage.py tests/unit/test_two_stage_runner.py
git commit -m "feat(method): TwoStageCDSBIRunner — pretrain+freeze summary, then pivot (Arm I-B)"
```

---

## Task 5: configs (method + experiment) + `pretrain` passthrough in run.py

**Files:**
- Create: `configs/method/cd_sbi_two_stage.yaml`, `configs/experiment/mu_sigma_stage_b_freeze.yaml`
- Modify: `src/cdsbi/experiments/run.py` (`_fit_config`: pass the `pretrain` block through for cd_sbi)

READ `configs/method/cd_sbi.yaml` and `configs/experiment/mu_sigma_stage_b.yaml` to mirror their shape.

**CRITICAL:** `run.py` builds the `fit()` config via `_fit_config` → `_recipe_dict`, which copies an EXPLICIT whitelist of training keys (`lr`, `batch_size`, …) and **drops everything else** — so a `training.pretrain` block would NOT reach `TwoStageCDSBIRunner.fit` without a passthrough. Step 0 below adds it.

- [ ] **Step 0: Pass `pretrain` through `_fit_config` (run.py)**

In `src/cdsbi/experiments/run.py`, change the `cd_sbi` branch of `_fit_config` to attach an optional `pretrain` block:
```python
def _fit_config(cfg: DictConfig, method_name: str) -> dict:
    t = cfg.training
    if method_name == "cd_sbi":
        d = _recipe_dict(t)
        pre = OmegaConf.select(t, "pretrain", default=None)
        if pre is not None:
            d["pretrain"] = OmegaConf.to_container(pre, resolve=True)
        return d
    if method_name in ("npe", "nle", "nre"):
        return _recipe_dict(t)
    ...
```
(Leave the `npe/nle/nre`, `lf2i_bff`, and `raise` branches unchanged. The `pretrain` key only appears when an experiment defines `training.pretrain` — single-stage cd_sbi runs are unaffected.)

- [ ] **Step 1: Write `configs/method/cd_sbi_two_stage.yaml`**

```yaml
name: cd_sbi            # same method family; runner_class selects the two-stage variant
runner_class: cdsbi.methods.cd_sbi_two_stage.TwoStageCDSBIRunner
flow: single_index_monotone
allow_ablation: false
```

- [ ] **Step 2: Write `configs/experiment/mu_sigma_stage_b_freeze.yaml`**

```yaml
# @package _global_
defaults:
  - override /target: normal_mu_sigma
  - override /flow: single_index_monotone
  - override /conditioner: deep_sets
  - override /method: cd_sbi_two_stage
  - override /budget: medium

# self-contained flow dispatch (cb1e08e guard)
method:
  flow: single_index_monotone

# I-B summary: d_out=d_theta (predict θ), standardization OFF (freeze prevents the cheat)
conditioner:
  d_out: 2
  standardize: false

# Stage-1 pretraining settings consumed by TwoStageCDSBIRunner.fit
training:
  pretrain:
    lr: 3e-3
    n_steps: 3000
    batch_size: 256

experiment:
  name: mu_sigma_stage_b_freeze
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

- [ ] **Step 3: Verify composition + that `config["pretrain"]` reaches fit**

The runner reads `config["pretrain"]` from the training config dict. Confirm `cfg.training.pretrain` composes and that `cfg.conditioner` carries `standardize: false` + `d_out: 2`:
```bash
python -c "
from hydra import initialize, compose
from omegaconf import OmegaConf
with initialize(version_base=None, config_path='configs'):
    cfg = compose(config_name='config', overrides=['experiment=mu_sigma_stage_b_freeze'])
print('method runner:', cfg.method.runner_class)
print('conditioner:', cfg.conditioner.name, '| standardize:', cfg.conditioner.standardize, '| d_out:', cfg.conditioner.d_out)
print('pretrain:', OmegaConf.select(cfg, 'training.pretrain'))
print('grid:', len(cfg.experiment.eval_thetas_interior))
"
```
Expected: runner `...TwoStageCDSBIRunner`; conditioner `deep_sets`, standardize `False`, d_out `2`; pretrain block present; grid `9`.

Then confirm the `pretrain` block actually reaches the fit-config dict through `_fit_config` (the Step-0 passthrough):
```bash
python -c "
from hydra import initialize, compose
from cdsbi.experiments.run import _fit_config
with initialize(version_base=None, config_path='configs'):
    cfg = compose(config_name='config', overrides=['experiment=mu_sigma_stage_b_freeze'])
fc = _fit_config(cfg, 'cd_sbi')
print('pretrain in fit_config:', fc.get('pretrain'))
assert fc.get('pretrain') and int(fc['pretrain']['n_steps']) == 3000
print('OK')
"
```
Expected: prints the `pretrain` dict (`{lr, n_steps: 3000, batch_size}`) and `OK`. (This is the check that the Step-0 passthrough works — without it, `_fit_config` would drop `pretrain` and the runner would raise.)

- [ ] **Step 4: Commit**

```bash
git add configs/method/cd_sbi_two_stage.yaml configs/experiment/mu_sigma_stage_b_freeze.yaml \
        src/cdsbi/experiments/run.py
git commit -m "config: cd_sbi_two_stage method + mu_sigma_stage_b_freeze experiment + pretrain passthrough (Arm I-B)"
```

---

## Task 6: intensive replication on the harness

**Files:**
- Create: `tests/intensive/test_replicate_mu_sigma_stage_b_freeze.py`

The verdict for Arm I-B: a single full run (or small sweep) that the frozen-summary pivot (a) recovers sufficiency on BOTH coords (σ² no longer collapses), (b) does NOT cheat the floor, (c) calibrates near the Stage-A control.

- [ ] **Step 1: Write the test**

```python
"""Intensive: Arm I-B (two-stage freeze) — sufficiency recovered (incl. σ²), no
floor cheat, calibration near the Stage-A oracle control."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.intensive
def test_arm_ib_two_stage_freeze_calibrates_without_cheat():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi_two_stage import TwoStageCDSBIRunner
    from cdsbi.diagnostics.sufficiency_recovery import SufficiencyRecovery
    from cdsbi.diagnostics.floor_integrity import FloorIntegrity
    from scipy.stats import kstest, chi2

    torch.manual_seed(0)
    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=32, depth=2)
    cond = DeepSetsConditioner(n_iid=sim.n_iid, d_out=sim.d_theta, hidden=32, depth=2,
                               standardize=False)
    runner = TwoStageCDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss())
    config = {"lr": 3e-3, "batch_size": 256, "n_steps": 6000, "n_train": 10000,
              "optimizer": "adamw", "lr_schedule": "warmup_cosine",
              "warmup_steps": 300, "lr_min_ratio": 0.01, "fresh_batch": False,
              "pretrain": {"lr": 3e-3, "n_steps": 3000, "batch_size": 256}}
    trained = runner.fit(simulator=sim, config=config, seed=0)

    # (a) sufficiency recovered on BOTH coords (σ² is the device-1 casualty)
    sr = SufficiencyRecovery(n_eval=4000)(trained, sim).value
    print(f"I-B sufficiency: log s²={sr['spearman_log_s2'].iloc[0]:.3f} "
          f"X̄={sr['spearman_xbar'].iloc[0]:.3f}")
    assert sr["spearman_log_s2"].iloc[0] > 0.9, "σ²-information not recovered"
    assert sr["spearman_xbar"].iloc[0] > 0.9, "μ-information not recovered"

    # (b) no floor cheat
    fi = FloorIntegrity()(trained, sim).value
    print(f"I-B floor: final_loss={fi['final_loss'].iloc[0]:.3f} "
          f"floor={fi['entropy_floor'].iloc[0]:.3f} margin={fi['floor_margin'].iloc[0]:.3f}")
    assert not bool(fi["cheats"].iloc[0]), "device cheated below the entropy floor"

    # (c) calibration near the Stage-A control: joint Mahalanobis PIT ~ χ²₂
    theta_0 = (0.0, 0.0)
    xv = sim.sample_x_given_theta(theta_0, 3000, np.random.default_rng(7))
    th = torch.tensor([[0.0, 0.0]], dtype=xv.dtype).expand(xv.shape[0], -1)
    with torch.no_grad():
        r = trained.procedure.pivot(th, xv).cpu().numpy()
    ks = kstest(chi2.cdf((r ** 2).sum(1), df=2), "uniform").statistic
    print(f"I-B joint Mahalanobis KS = {ks:.3f}")
    assert ks < 0.06, f"learned-summary (I-B) joint calibration KS {ks:.3f} too high"
```

- [ ] **Step 2: Run it (intensive; pretrain + pivot, minutes on GPU)**

Run: `pytest tests/intensive/test_replicate_mu_sigma_stage_b_freeze.py -v -s -m intensive`
Report the printed sufficiency / floor / KS values. Interpretation (do NOT loosen tolerances; report findings):
- **All pass** → Arm I-B works: a learned summary, frozen after predict-θ pretraining, calibrates without the cheat. Record the numbers for the M3.4 verdict table.
- **σ²-Spearman < 0.9** → predict-θ stage-1 under-recovered σ²; report MSE + Spearman — the spec's InfoNCE fallback is the escalation (M3.1 follow-up). STOP.
- **Calibration KS ≥ 0.06 but sufficiency OK** → the frozen features are sufficient but mis-oriented/ordered for the fixed-sign flow (the risk the predict-θ design is meant to prevent — if it bites, report and the fix is the sign-free flow variant). STOP, report.
- **`cheats` True** → unexpected (the frozen summary should not be collapsible); report final_loss vs floor. STOP.

- [ ] **Step 3: Confirm fast suite green**

Run: `pytest -q`
Expected: all fast tests pass; this intensive test deselected.

- [ ] **Step 4: Commit** (commit the test regardless; mark DONE_WITH_CONCERNS if step 2 surfaced a finding)

```bash
git add tests/intensive/test_replicate_mu_sigma_stage_b_freeze.py
git commit -m "test(intensive): Arm I-B (two-stage freeze) — sufficiency + no-cheat + calibration"
```

---

## Self-review

- **Spec coverage (Arm I-B):** Stage-1 predict-θ pretrain (Task 3) ✓; freeze (Tasks 2–3) ✓; Stage-2 pivot on frozen features via `TwoStageCDSBIRunner` (Task 4) ✓; `standardize=False` for the freeze arm (Task 1) ✓; configs (Task 5) ✓; harness verdict — `SufficiencyRecovery` + `FloorIntegrity` + calibration (Task 6) ✓. The spec's success criteria (Spearman > 0.9 both coords; no cheat; calibration near control) are exactly Task 6's assertions.
- **Placeholder scan:** every step has runnable code + commands + expected output; no TBD.
- **Type consistency:** `DeepSetsConditioner(..., standardize=...)` (Task 1) is used in Tasks 3/4/6 and the config (Task 5); `pretrain_summary(simulator, summary, config, seed, device)` (Task 3) is called by `TwoStageCDSBIRunner.fit` (Task 4) with `config=pre_cfg`; `TwoStageCDSBIRunner(flow, conditioner, loss, device)` matches the `CDSBIRunner.__init__` signature (it inherits it) and `_build_method`'s instantiate args (flow/conditioner/loss/allow_ablation/device); `config["pretrain"]` written in the config (Task 5) is read in Task 4. `d_out=sim.d_theta` (=2) is consistent with predicting θ.
- **Risk — predict-θ may under-identify σ at the prior center.** `log σ` is identified by the data spread `s²` across `n_iid=10`; the MSE regression should recover it, but if the prior-mean region is σ-insensitive the σ-coordinate MSE could lag. Task 3's MSE check and Task 6's σ²-Spearman gate both catch this early; the documented escalation is InfoNCE (a later follow-up), NOT loosening thresholds.
- **Risk — orientation/order.** Addressed by construction (predict-θ → output k ↔ θ_k, signs matching the fixed-sign flow). Task 6's calibration KS is the backstop if the assumption fails.
- **Known follow-on:** Arm II-A (M3.2), Arm I-A (M3.3), cross-arm verdict (M3.4).
