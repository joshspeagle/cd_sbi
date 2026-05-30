# CD-SBI M3.1′ — Stage-B Arm II-A (calibration-invariant energy objective)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the **primary** Stage-B arm: train a learned DeepSets summary end-to-end with the monotone pivot under a **calibration-invariant energy objective** (distance of `{r(θ₀;X)}` to `N(0,I₂)`, no Jacobian) and show on the M3.0 harness that co-adaptation makes the summary emit **calibratable, un-warped** coordinates — calibrating σ *and* μ as well as the Stage-A oracle, with no cheat.

**Architecture:** A new `EnergyCalibrationLoss` scores a *group* of pivot values `{r_j}` (drawn at a common θ₀) against `N(0,I₂)` via the (strictly proper) energy score — a reparameterization-invariant distance with **no density/Jacobian term to inflate**, so the M2 collapse channel is absent. A new `EnergyCDSBIRunner` trains via **grouped-by-θ₀ batches**: draw `B` proposal θ₀, `m` datasets per θ₀, forward through summary→flow to get `r` shaped `(B, m, d)`, score each group vs `N(0,I₂)`, average. The summary and the monotone pivot **co-adapt**: gradient pressure forces the summary to emit coordinates the pivot can calibrate (the §1.5 cure). The flow keeps R1 (monotone-in-θ, needed for confidence sets); R2 is not used by the energy loss.

**Tech Stack:** PyTorch (`torch.cdist`, Adam), Hydra, pytest, numpy/scipy. Reuses `DeepSetsConditioner`, `SingleIndexMonotoneFlow`, the M3.0 harness, and `CDSBIRunner` plumbing (procedure + `encode_fn`).

**Spec:** `docs/superpowers/specs/2026-05-29-cd-sbi-stage-b-device-bakeoff-design.md` — Arm II-A (§3, PRIMARY) + §1.5 (co-adaptation) + §2 harness + §5 criteria.

**Scope note (deviation from spec §4):** the spec folds the `FloorIntegrity` empirical-`H(feat|θ)` fix into M3.1′, but II-A uses the energy loss → `FloorIntegrity` *correctly no-ops* (non-NF-MLE loss), so the fix is not exercised here. It is **deferred to M3.2′ (Arm I-A)**, the NF-MLE-on-learned-features arm that actually needs an honest floor. M3.1′ relies on `SufficiencyRecovery` + the calibration diagnostics for its verdict.

---

## Conventions / decisions locked

- **Energy score (strictly proper), dropping the model-independent ref-ref term.** For a group `R={r_1..r_m}⊂ℝ^d` at one θ₀ and reference `Z={z_1..z_M}~N(0,I_d)`:
  `S(R) = (2/(m·M)) Σ_{i,j}‖r_i−z_j‖ − (1/m²) Σ_{i,i'}‖r_i−r_i'‖`.
  `S` is minimized exactly when `R ~ N(0,I_d)`; it has **no Jacobian/density term** → no collapse channel. Loss = mean of `S` over the `B` groups in a batch.
- **Grouped training:** `B` = θ₀ per batch (default 32), `m` = datasets per θ₀ (default 64), `M` = `N(0,I_d)` reference draws per step (default 256, resampled each step). θ₀ drawn from the prior via `simulator._draw_theta`; datasets via `simulator.sample_x_given_theta`.
- **Summary:** `DeepSetsConditioner(d_out=2)`, **end-to-end trainable**. `standardize=False` (no cheat channel ⇒ normalization unneeded; the flow's affine combiner + co-adaptation handle scale). The flow co-adapts, so the summary is free to emit affine-friendly coordinates.
- **Flow:** unchanged `SingleIndexMonotoneFlow` (R1 needed for the pivot inversion; R2 unused by the energy loss but harmless). Signs injected from the simulator as usual.
- **Loss guarantees:** `EnergyCalibrationLoss.required_guarantees = {R1}` (monotone-in-θ for confidence sets). The single-index flow provides {R1,R2} ⊇ {R1} → `check_guarantees` passes.
- **Method dispatch:** a `cd_sbi_energy` family selected by `cfg.method.loss == "energy"` + `runner_class = EnergyCDSBIRunner`. `FloorIntegrity` keys on `loss_class == "NFMLELoss"`, so it no-ops for `EnergyCalibrationLoss` — correct.

---

## File structure

```
NEW
  src/cdsbi/losses/energy_calibration.py           # EnergyCalibrationLoss (group energy score)
  src/cdsbi/methods/cd_sbi_energy.py               # EnergyCDSBIRunner (grouped-by-θ₀ fit)
  configs/method/cd_sbi_energy.yaml
  configs/experiment/mu_sigma_stage_b_energy.yaml
  tests/unit/test_energy_calibration_loss.py
  tests/unit/test_energy_runner.py
  tests/intensive/test_replicate_mu_sigma_stage_b_energy.py

MODIFY
  src/cdsbi/experiments/run.py                      # _build_method: energy-loss selector; _fit_config: group passthrough
```

No change to `DeepSetsConditioner`, `SingleIndexMonotoneFlow`, the simulator, or the M3.0 diagnostics.

---

## Task 1: `EnergyCalibrationLoss`

**Files:**
- Create: `src/cdsbi/losses/energy_calibration.py`
- Test: `tests/unit/test_energy_calibration_loss.py`

READ `src/cdsbi/losses/base.py` (the `Loss` base + `MonotonicityMismatchError`) and `src/cdsbi/losses/nfmle.py` (the `required_guarantees` + `check_guarantees` pattern) and `src/cdsbi/flows/base.py` (`Guarantee`).

- [ ] **Step 1: Write the failing tests**

```python
"""EnergyCalibrationLoss: group energy score vs N(0,I); minimized at calibration."""
import torch
from cdsbi.flows.base import Guarantee


def _loss():
    from cdsbi.losses.energy_calibration import EnergyCalibrationLoss
    return EnergyCalibrationLoss(n_ref=512)


def test_required_guarantees_is_r1_only():
    assert _loss().required_guarantees == frozenset({Guarantee.R1})


def test_score_lower_for_calibrated_group():
    torch.manual_seed(0)
    loss = _loss()
    # one group, m=400 samples. Calibrated R ~ N(0,I₂) should score BELOW a
    # mis-scaled group (N(0, 2.5²·I)) and a shifted group (N([2,2], I)).
    calibrated = torch.randn(1, 400, 2)
    misscaled = 2.5 * torch.randn(1, 400, 2)
    shifted = torch.randn(1, 400, 2) + torch.tensor([2.0, 2.0])
    s_cal = loss.score(calibrated, rng_seed=1)
    s_mis = loss.score(misscaled, rng_seed=1)
    s_shift = loss.score(shifted, rng_seed=1)
    assert s_cal < s_mis, f"calibrated {s_cal:.3f} should beat misscaled {s_mis:.3f}"
    assert s_cal < s_shift, f"calibrated {s_cal:.3f} should beat shifted {s_shift:.3f}"


def test_score_is_differentiable_wrt_r():
    loss = _loss()
    r = torch.randn(2, 64, 2, requires_grad=True)
    s = loss.score(r, rng_seed=0)
    s.backward()
    assert r.grad is not None and torch.isfinite(r.grad).all()


def test_score_averages_over_groups():
    loss = _loss()
    r = torch.randn(4, 50, 2)
    s = loss.score(r, rng_seed=0)
    assert s.ndim == 0  # scalar (mean over the 4 groups)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_energy_calibration_loss.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `src/cdsbi/losses/energy_calibration.py`**

```python
"""EnergyCalibrationLoss: a reparameterization-invariant calibration objective.

For a group of pivot values R={r_j}⊂ℝ^d drawn at a common θ₀ and reference
Z~N(0,I_d), the energy score
    S(R) = (2/(m·M)) Σ_ij‖r_i−z_j‖ − (1/m²) Σ_ii'‖r_i−r_i'‖   (ref-ref term dropped)
is a strictly proper score minimized exactly when R ~ N(0,I_d). It sees only the
*distribution* of r — there is NO Jacobian/density term — so the Stage-B
information-collapse cheat (which inflates a feature-Jacobian) has no channel here.
Loss = mean of S over the B groups in a batch. Requires only R1 (monotone-in-θ) of
the flow — confidence sets still invert the pivot; R2 is unused.
"""
from __future__ import annotations

import torch

from cdsbi.flows.base import Guarantee
from cdsbi.losses.base import Loss, MonotonicityMismatchError


class EnergyCalibrationLoss(Loss):
    required_guarantees = frozenset({Guarantee.R1})

    def __init__(self, n_ref: int = 256):
        self.n_ref = n_ref

    def check_guarantees(self, flow) -> None:
        guarantees = getattr(flow, "monotonicity_guarantees", frozenset())
        if not self.required_guarantees.issubset(guarantees):
            missing = self.required_guarantees - guarantees
            raise MonotonicityMismatchError(
                f"EnergyCalibrationLoss requires {sorted(g.value for g in self.required_guarantees)}; "
                f"flow {type(flow).__name__} provides {sorted(g.value for g in guarantees)}; "
                f"missing {sorted(g.value for g in missing)}."
            )

    def score(self, r_groups: torch.Tensor, rng_seed: int | None = None,
              generator: torch.Generator | None = None) -> torch.Tensor:
        """r_groups: (B, m, d). Returns the mean energy score (scalar)."""
        B, m, d = r_groups.shape
        if generator is None and rng_seed is not None:
            generator = torch.Generator(device=r_groups.device).manual_seed(int(rng_seed))
        z = torch.randn(self.n_ref, d, device=r_groups.device, dtype=r_groups.dtype,
                        generator=generator)                      # (M, d)
        # cross term: 2 * mean_{i,j} ||r_i - z_j||  per group
        cross = torch.cdist(r_groups, z.unsqueeze(0).expand(B, -1, -1))   # (B, m, M)
        term1 = 2.0 * cross.mean(dim=(1, 2))                              # (B,)
        # within term: mean_{i,i'} ||r_i - r_i'||  per group
        within = torch.cdist(r_groups, r_groups)                          # (B, m, m)
        term2 = within.mean(dim=(1, 2))                                   # (B,)
        return (term1 - term2).mean()
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_energy_calibration_loss.py -v`
Expected: 4 pass (calibrated group scores below mis-scaled and shifted; differentiable; scalar).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/losses/energy_calibration.py tests/unit/test_energy_calibration_loss.py
git commit -m "feat(loss): EnergyCalibrationLoss — reparam-invariant group energy score vs N(0,I)"
```

---

## Task 2: `EnergyCDSBIRunner` (grouped-by-θ₀ training)

**Files:**
- Create: `src/cdsbi/methods/cd_sbi_energy.py`
- Test: `tests/unit/test_energy_runner.py`

READ `src/cdsbi/methods/cd_sbi.py` (`CDSBIRunner.__init__`, and the end-of-`fit` block that builds `pivot_fn` + `encode_fn` + `PivotBasedProcedure` + `TrainedModel` — reuse that exactly) and `src/cdsbi/methods/base.py` (`TrainedModel`).

- [ ] **Step 1: Write the failing test**

```python
"""EnergyCDSBIRunner: grouped fit trains summary+flow under the energy loss."""
import copy
import torch


def test_energy_runner_fit_trains_and_exposes_procedure():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.energy_calibration import EnergyCalibrationLoss
    from cdsbi.methods.cd_sbi_energy import EnergyCDSBIRunner

    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=16)
    cond = DeepSetsConditioner(n_iid=sim.n_iid, d_out=2, hidden=16, standardize=False)
    before = {k: v.clone() for k, v in cond.state_dict().items() if v.dtype.is_floating_point}
    runner = EnergyCDSBIRunner(flow=flow, conditioner=cond, loss=EnergyCalibrationLoss(n_ref=128),
                               device="cpu")
    trained = runner.fit(simulator=sim, config={
        "lr": 3e-3, "n_steps": 40,
        "group": {"n_theta_per_batch": 16, "group_size": 32, "n_ref": 128},
    }, seed=0)
    # summary co-trained
    after = cond.state_dict()
    assert any(not torch.allclose(before[k], after[k]) for k in before if k.startswith(("phi", "rho")))
    # procedure + encode_fn exposed for the harness
    assert trained.procedure.encode_fn is not None
    import numpy as np
    _, x = sim.sample(20, np.random.default_rng(0))
    assert trained.procedure.pivot(torch.zeros(20, 2), x).shape == (20, 2)
    assert trained.arch_metadata["loss_class"] == "EnergyCalibrationLoss"
    assert np.isfinite(trained.final_loss)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_energy_runner.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `src/cdsbi/methods/cd_sbi_energy.py`**

```python
"""EnergyCDSBIRunner: Stage-B Arm II-A. End-to-end summary+pivot trained under the
calibration-invariant EnergyCalibrationLoss with grouped-by-θ₀ batches. No Jacobian
term ⇒ no collapse cheat; the summary co-adapts to emit calibratable coordinates.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import TrainedModel
from cdsbi.reproducibility.seeding import seed_everything


class EnergyCDSBIRunner:
    def __init__(self, flow, conditioner, loss, allow_ablation: bool = False, device: str = "auto"):
        self.flow = flow
        self.conditioner = conditioner
        self.loss = loss
        self.allow_ablation = allow_ablation
        self.device = get_device(device)
        if not allow_ablation:
            self.loss.check_guarantees(flow)

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        rngs = seed_everything(seed)
        dev = self.device
        self.flow.to(dev); self.flow.train()
        cond_is_module = isinstance(self.conditioner, torch.nn.Module)
        if cond_is_module:
            self.conditioner.to(dev); self.conditioner.train()

        grp = config["group"]
        B = int(grp["n_theta_per_batch"]); m = int(grp["group_size"])
        n_ref = int(grp.get("n_ref", self.loss.n_ref))
        n_steps = int(config["n_steps"]); lr = float(config["lr"])
        grad_clip = float(config.get("grad_clip_norm", 5.0))

        params = list(self.flow.parameters())
        if cond_is_module:
            params += [p for p in self.conditioner.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)

        d = int(simulator.d_theta)
        losses = []
        t0 = time.time()
        for step in range(n_steps):
            theta0 = simulator._draw_theta(B, rngs.train)               # (B, d) np
            thetas, xs = [], []
            for b in range(B):
                x_b = simulator.sample_x_given_theta(theta0[b], m, rngs.train)  # (m, n_iid)
                xs.append(x_b)
                thetas.append(torch.as_tensor(theta0[b], dtype=x_b.dtype).expand(m, d))
            x = torch.cat(xs, dim=0).to(dev)                            # (B*m, n_iid)
            theta = torch.cat(thetas, dim=0).to(dev)                    # (B*m, d)
            context, _ = self.conditioner.encode(x)
            r, _ = self.flow.forward(theta, context=context)            # (B*m, d)
            r_groups = r.view(B, m, d)
            gen = torch.Generator(device=r.device).manual_seed(seed * 100003 + step)
            loss_val = self.loss.score(r_groups, generator=gen)
            opt.zero_grad(); loss_val.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            opt.step()
            losses.append(loss_val.item())
        wall = time.time() - t0

        self.flow.eval()
        if cond_is_module:
            self.conditioner.eval()
        flow = self.flow; conditioner = self.conditioner; device = dev

        def pivot_fn(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            theta = theta.to(device); x = x.to(device)
            context, _ = conditioner.encode(x)
            r, _ = flow.forward(theta, context=context)
            return r

        def encode_fn(x: torch.Tensor) -> torch.Tensor:
            x = x.to(device)
            with torch.no_grad():
                feats, _ = conditioner.encode(x)
            return feats

        procedure = PivotBasedProcedure(pivot_fn=pivot_fn, d_theta=d, encode_fn=encode_fn)
        arch_meta = {
            "flow_class": type(self.flow).__name__,
            "loss_class": type(self.loss).__name__,
            "loss_history_tail": losses[-min(100, len(losses)):],
            "conditioner_class": type(self.conditioner).__name__,
            "conditioner_params": self.conditioner.n_params(),
            "group": {"n_theta_per_batch": B, "group_size": m, "n_ref": n_ref},
        }
        return TrainedModel(procedure=procedure, state_dict={}, final_loss=float(losses[-1]),
                            n_steps=n_steps, wall_clock_sec=wall, arch_metadata=arch_meta)

    def n_params(self) -> dict:
        backbone = sum(p.numel() for p in self.flow.parameters())
        head = self.conditioner.n_params() if hasattr(self.conditioner, "n_params") else 0
        return {"backbone": backbone, "head": head, "calibration_stage": 0,
                "total": backbone + head, "kind": "flow"}
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_energy_runner.py -v`
Expected: PASS (summary co-trained; procedure + encode_fn exposed; loss_class recorded; finite loss).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/methods/cd_sbi_energy.py tests/unit/test_energy_runner.py
git commit -m "feat(method): EnergyCDSBIRunner — grouped-by-θ₀ training under the energy loss (Arm II-A)"
```

---

## Task 3: run.py wiring (loss selector + group passthrough)

**Files:**
- Modify: `src/cdsbi/experiments/run.py`

READ `_build_method` (the `m.name == "cd_sbi"` branch building flow/conditioner/`NFMLELoss()`/runner) and `_fit_config`/`_recipe_dict`.

- [ ] **Step 1: Loss selector in `_build_method`**

In the `m.name == "cd_sbi"` branch, choose the loss from an optional `method.loss` field and keep the runner generic:
```python
    if m.name == "cd_sbi":
        from cdsbi.losses.nfmle import NFMLELoss
        flow = _build_flow(cfg, simulator)
        allow_ablation = bool(OmegaConf.select(cfg, "method.allow_ablation", default=False))
        # ... existing conditioner dispatch (unchanged) ...
        loss_name = OmegaConf.select(cfg, "method.loss", default="nfmle")
        if loss_name == "energy":
            from cdsbi.losses.energy_calibration import EnergyCalibrationLoss
            loss_obj = EnergyCalibrationLoss()
        else:
            loss_obj = NFMLELoss()
        return _instantiate(
            runner_class,
            flow=flow,
            conditioner=conditioner,
            loss=loss_obj,
            allow_ablation=allow_ablation,
            device=cfg.device,
        )
```
(Replace the hardcoded `loss=NFMLELoss()` with `loss=loss_obj`. The conditioner dispatch block above it is unchanged.)

- [ ] **Step 2: Group passthrough in `_fit_config`**

In the `cd_sbi` branch of `_fit_config`, attach the optional `group` block (alongside the `pretrain` passthrough idiom if present — if M3.1's pretrain passthrough did NOT land, just add `group`):
```python
    if method_name == "cd_sbi":
        d = _recipe_dict(t)
        grp = OmegaConf.select(t, "group", default=None)
        if grp is not None:
            d["group"] = OmegaConf.to_container(grp, resolve=True)
        return d
```
(`group` only appears when an experiment defines `training.group`; NF-MLE runs are unaffected.)

- [ ] **Step 3: Verify import + that the energy method composes (after configs land in Task 4 — run this at the end of Task 4)**

Deferred to Task 4 Step 3 (needs the configs).

- [ ] **Step 4: Commit**

```bash
git add src/cdsbi/experiments/run.py
git commit -m "feat(run): _build_method energy-loss selector + _fit_config group passthrough"
```

---

## Task 4: configs (method + experiment)

**Files:**
- Create: `configs/method/cd_sbi_energy.yaml`, `configs/experiment/mu_sigma_stage_b_energy.yaml`

- [ ] **Step 1: `configs/method/cd_sbi_energy.yaml`**

```yaml
name: cd_sbi                 # cd_sbi family (so _build_method's branch fires)
runner_class: cdsbi.methods.cd_sbi_energy.EnergyCDSBIRunner
flow: single_index_monotone
loss: energy                 # selects EnergyCalibrationLoss in _build_method
allow_ablation: false
```

- [ ] **Step 2: `configs/experiment/mu_sigma_stage_b_energy.yaml`**

```yaml
# @package _global_
defaults:
  - override /target: normal_mu_sigma
  - override /flow: single_index_monotone
  - override /conditioner: deep_sets
  - override /method: cd_sbi_energy
  - override /budget: medium

method:
  flow: single_index_monotone   # self-contained flow dispatch (cb1e08e guard)

conditioner:
  d_out: 2
  standardize: false            # no cheat channel under the energy loss

# grouped-by-θ₀ training settings consumed by EnergyCDSBIRunner.fit
training:
  group:
    n_theta_per_batch: 32
    group_size: 64
    n_ref: 256

experiment:
  name: mu_sigma_stage_b_energy
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
NOTE: `cd_sbi_energy.yaml` sets `loss: energy` — confirm `configs/method/cd_sbi.yaml` does NOT already define a conflicting `loss` key (it should not). The `training` group default supplies `lr`/`n_steps`/etc.; `group` is the only addition.

- [ ] **Step 3: Verify composition + wiring**

```bash
python -c "
from hydra import initialize, compose
from cdsbi.experiments.run import _build_method, _build_simulator, _fit_config
with initialize(version_base=None, config_path='configs'):
    cfg = compose(config_name='config', overrides=['experiment=mu_sigma_stage_b_energy'])
sim = _build_simulator(cfg)
runner = _build_method(cfg, sim)
fc = _fit_config(cfg, 'cd_sbi')
print('runner:', type(runner).__name__)
print('loss:', type(runner.loss).__name__)
print('group in fit_config:', fc.get('group'))
print('conditioner standardize:', cfg.conditioner.standardize)
"
```
Expected: `EnergyCDSBIRunner`, `EnergyCalibrationLoss`, group dict `{n_theta_per_batch:32, group_size:64, n_ref:256}`, standardize `False`. If `_build_method` returns a plain `CDSBIRunner` or `NFMLELoss`, the loss selector (Task 3 Step 1) or `method.loss` field is wrong — STOP and fix.

- [ ] **Step 4: Commit**

```bash
git add configs/method/cd_sbi_energy.yaml configs/experiment/mu_sigma_stage_b_energy.yaml
git commit -m "config: cd_sbi_energy method + mu_sigma_stage_b_energy experiment (Arm II-A)"
```

---

## Task 5: intensive replication on the harness (the verdict)

**Files:**
- Create: `tests/intensive/test_replicate_mu_sigma_stage_b_energy.py`

The Arm II-A verdict: end-to-end co-adaptation under the energy loss makes the learned summary (a) recover sufficiency on **both** coords (σ² included — the I-B casualty), and (b) calibrate near the Stage-A control. (No floor check — `FloorIntegrity` no-ops for the energy loss.)

- [ ] **Step 1: Write the test**

```python
"""Intensive: Arm II-A (energy calibration) — co-adapted learned summary recovers
sufficiency (incl. σ²) AND calibrates near the Stage-A oracle, with no cheat channel."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.intensive
def test_arm_iia_energy_calibrates_and_recovers_sufficiency():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.energy_calibration import EnergyCalibrationLoss
    from cdsbi.methods.cd_sbi_energy import EnergyCDSBIRunner
    from cdsbi.diagnostics.sufficiency_recovery import SufficiencyRecovery
    from scipy.stats import kstest, chi2

    torch.manual_seed(0)
    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=32, depth=2)
    cond = DeepSetsConditioner(n_iid=sim.n_iid, d_out=2, hidden=32, depth=2, standardize=False)
    runner = EnergyCDSBIRunner(flow=flow, conditioner=cond, loss=EnergyCalibrationLoss(n_ref=256))
    config = {"lr": 2e-3, "n_steps": 8000,
              "group": {"n_theta_per_batch": 32, "group_size": 64, "n_ref": 256}}
    trained = runner.fit(simulator=sim, config=config, seed=0)

    # (a) sufficiency recovered on BOTH coords (σ² is the device-1/I-B casualty)
    sr = SufficiencyRecovery(n_eval=4000)(trained, sim).value
    print(f"II-A sufficiency: log s²={sr['spearman_log_s2'].iloc[0]:.3f} "
          f"X̄={sr['spearman_xbar'].iloc[0]:.3f}")
    assert sr["spearman_log_s2"].iloc[0] > 0.9, "σ²-information not recovered"
    assert sr["spearman_xbar"].iloc[0] > 0.9, "μ-information not recovered"

    # (b) calibration near the Stage-A control: joint Mahalanobis PIT ~ χ²₂ at several θ₀
    for theta_0 in [(0.0, 0.0), (-0.69, 2.0), (0.69, -2.0)]:
        xv = sim.sample_x_given_theta(theta_0, 3000, np.random.default_rng(7))
        th = torch.tensor([list(theta_0)], dtype=xv.dtype).expand(xv.shape[0], -1)
        with torch.no_grad():
            r = trained.procedure.pivot(th, xv).cpu().numpy()
        ks = kstest(chi2.cdf((r ** 2).sum(1), df=2), "uniform").statistic
        print(f"II-A joint Mahalanobis KS @ {theta_0} = {ks:.3f}")
        assert ks < 0.08, f"II-A joint calibration KS {ks:.3f} too high at {theta_0}"
```

- [ ] **Step 2: Run it (intensive; grouped training, minutes on GPU)**

Run: `pytest tests/intensive/test_replicate_mu_sigma_stage_b_energy.py -v -s -m intensive`
Report the printed sufficiency + per-θ₀ KS values. Interpretation (do NOT loosen tolerances; report findings):
- **All pass** → **the core §1.5 question is answered yes:** co-adaptation under a cheat-free loss makes a learned summary emit calibratable coords (σ included). Record for the M3.3′ verdict.
- **σ²-Spearman < 0.9 OR σ-KS high** → co-adaptation did NOT un-warp σ; report values. This would mean even end-to-end the affine coord-0 combiner is the bottleneck — reopening the flow-architecture direction (§1.5: feature-rectifier / context-conditioned coord-0). STOP and report.
- **Training unstable** (loss NaN / oscillating) → report; the group size `m`, ref count `M`, or lr are the knobs (raise `m`/`M`, lower lr) — a recipe issue, not a tolerance one.

- [ ] **Step 3: Confirm fast suite green**

Run: `pytest -q` (intensive deselected). Report the count.

- [ ] **Step 4: Commit** (commit the test regardless; mark DONE_WITH_CONCERNS if step 2 surfaced a finding)

```bash
git add tests/intensive/test_replicate_mu_sigma_stage_b_energy.py
git commit -m "test(intensive): Arm II-A (energy) — sufficiency (incl σ²) + calibration near oracle"
```

---

## Self-review

- **Spec coverage (Arm II-A):** `EnergyCalibrationLoss` (Task 1) ✓; grouped-by-θ₀ `EnergyCDSBIRunner` (Task 2) ✓; run.py loss-selector + group passthrough (Task 3) ✓; configs (Task 4) ✓; harness verdict — `SufficiencyRecovery` (incl. σ²) + calibration (Task 5) ✓. The spec's §5 per-arm bar (Spearman > 0.9 both coords; calibration near control) is Task 5's assertions. `FloorIntegrity` no-ops for the energy loss (documented scope note; the empirical-floor fix is deferred to M3.2′).
- **Placeholder scan:** every step has runnable code + commands + expected output; no TBD.
- **Type consistency:** `EnergyCalibrationLoss(n_ref).score(r_groups (B,m,d), generator=...) -> scalar` (Task 1) is called by `EnergyCDSBIRunner.fit` (Task 2) and the unit tests; `required_guarantees={R1}` passes `check_guarantees` against the {R1,R2} flow. `EnergyCDSBIRunner(flow, conditioner, loss, allow_ablation, device)` matches `_build_method`'s instantiate args (Task 3). `config["group"]={n_theta_per_batch,group_size,n_ref}` written in the config (Task 4) is read in fit (Task 2) via the `_fit_config` passthrough (Task 3). `procedure.encode_fn` (from M3.0) is set in fit and read by `SufficiencyRecovery` (Task 5).
- **Risk — energy-loss training stability/variance.** Group size `m` and ref count `M` control gradient variance; defaults (m=64, B=32, M=256) are a reasonable start, the test flags instability as a recipe knob. The energy score is the strictly-proper objective; the monotone flow keeps confidence-set validity (R1).
- **Risk — does co-adaptation actually un-warp σ?** This is the empirical question (Task 5). If it fails, the documented escalation is the flow-architecture direction (§1.5) — reported, not forced.
- **Known follow-on:** M3.2′ Arm I-A (invertible benchmark) + the `FloorIntegrity` empirical-floor fix; M3.3′ verdict + I-B contrast + manuscript.