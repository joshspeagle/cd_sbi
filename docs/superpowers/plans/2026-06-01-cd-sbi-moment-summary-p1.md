# CD-SBI Moment-Summary — Phase 1 (regular consistency, μσ) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the sequential two-stage CD-SBI construction (regress a `d_θ`-dim
posterior-moment summary → freeze → calibrate an NF-MLE pivot) and show on (μ,σ²)
that it is valid, efficient, and does **not** collapse σ — with no Fisher term.

**Architecture:** Stage 1 trains a `MomentRegressionConditioner` (DeepSets backbone)
by L² regression to a target `m(θ)` (Phase 1: `m(θ)=θ`), then **freezes** it. Stage 2
trains a monotone pivot (`SingleIndexMonotoneFlow`, R1 on) by NF-MLE on `(θ, frozen
summary)`. A new `FisherRecovery` diagnostic measures efficiency `det I_h/det I_X`
from the true (autograd) score. The build is glued by a new `TwoStageCDSBIRunner`
mirroring the existing `ExactDensityCDSBIRunner` shape.

**Tech stack:** Python, PyTorch, Hydra configs, pytest. Reuses
`NormalUnknownMeanVar`, `SingleIndexMonotoneFlow`, `NFMLELoss`, `PivotBasedProcedure`,
`SufficientStatConditioner` (oracle baseline), `SufficiencyRecovery`, `Coverage`.

**Spec:** `docs/superpowers/specs/2026-05-31-cd-sbi-moment-summary-design.md` (Phase 1).

---

## File structure (Phase 1)

- Create `src/cdsbi/conditioners/moment_regression.py` — `MomentRegressionConditioner`
  (summary module + `regression_target`).
- Create `src/cdsbi/methods/cd_sbi_two_stage.py` — `TwoStageCDSBIRunner` (Stage-1
  regression + freeze, Stage-2 NF-MLE).
- Create `src/cdsbi/diagnostics/fisher_recovery.py` — `FisherRecovery` (`det I_h/I_X`).
- Create `configs/method/cd_sbi_moment.yaml`, `configs/experiment/moment_mu_sigma.yaml`.
- Modify `src/cdsbi/experiments/run.py` — dispatch the two-stage runner + moment conditioner.
- Tests: `tests/unit/test_moment_regression.py`, `tests/unit/test_fisher_recovery.py`,
  `tests/integration/test_two_stage_runner.py`, `tests/intensive/test_replicate_moment_mu_sigma.py`.

---

### Task 1: MomentRegressionConditioner

**Files:**
- Create: `src/cdsbi/conditioners/moment_regression.py`
- Test: `tests/unit/test_moment_regression.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_moment_regression.py
import torch
from cdsbi.conditioners.moment_regression import MomentRegressionConditioner


def test_encode_shape_and_logdet_zero():
    cond = MomentRegressionConditioner(n_iid=10, d_theta=2)
    x = torch.randn(7, 10)
    feats, log_det = cond.encode(x)
    assert feats.shape == (7, 2)
    assert torch.allclose(log_det, torch.zeros(7))


def test_regression_target_theta_is_identity():
    cond = MomentRegressionConditioner(n_iid=10, d_theta=2, target="theta")
    theta = torch.tensor([[0.3, -1.0], [0.5, 2.0]])
    assert torch.equal(cond.regression_target(theta), theta)


def test_regression_target_theta_sq_is_coordinatewise_square():
    cond = MomentRegressionConditioner(n_iid=10, d_theta=1, target="theta_sq")
    theta = torch.tensor([[2.0], [-3.0]])
    assert torch.equal(cond.regression_target(theta), torch.tensor([[4.0], [9.0]]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_moment_regression.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cdsbi.conditioners.moment_regression'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cdsbi/conditioners/moment_regression.py
"""MomentRegressionConditioner: a learned d_θ-dim summary h_φ(X) trained (Stage 1,
by the TwoStageCDSBIRunner) via L² regression to a d_θ-dim posterior-moment target
m(θ). Permutation-invariant DeepSets backbone (mean-pool of a per-element MLP). The
summary is a feature map, not part of the NF-MLE change-of-variables, so
log_det_contrib = 0 (mirrors DeepSetsConditioner)."""
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


class MomentRegressionConditioner(nn.Module):
    def __init__(self, n_iid: int, d_theta: int, hidden: int = 64, depth: int = 2,
                 target: str = "theta"):
        super().__init__()
        if target not in ("theta", "theta_sq"):
            raise ValueError(f"target must be 'theta' or 'theta_sq', got {target}")
        self.n_iid = n_iid
        self.d_theta = d_theta
        self.target = target
        self.phi = _tanh_mlp(1, hidden, hidden, depth)         # per-element ℝ→ℝ^h
        self.rho = _tanh_mlp(hidden, hidden, d_theta, depth)   # pooled ℝ^h→ℝ^{d_θ}

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[-1] == self.n_iid, (
            f"MomentRegressionConditioner expected {self.n_iid} iid obs, got {x.shape[-1]}"
        )
        n, m = x.shape
        h = self.phi(x.reshape(n * m, 1)).reshape(n, m, -1).mean(dim=1)   # (n, hidden)
        feats = self.rho(h)                                              # (n, d_θ)
        log_det = torch.zeros(n, dtype=x.dtype, device=x.device)
        return feats, log_det

    def regression_target(self, theta: torch.Tensor) -> torch.Tensor:
        if self.target == "theta":
            return theta
        return theta ** 2          # coordinatewise (d_θ-dim, respects the §16 cap)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_moment_regression.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/conditioners/moment_regression.py tests/unit/test_moment_regression.py
git commit -m "feat(conditioner): MomentRegressionConditioner for two-stage CD-SBI"
```

---

### Task 2: FisherRecovery diagnostic

**Files:**
- Create: `src/cdsbi/diagnostics/fisher_recovery.py`
- Test: `tests/unit/test_fisher_recovery.py`

Builds the efficiency metric `det I_h/det I_X` from the true (autograd) score
`U_X = ∇_θ log p(X|θ)`. `I_X = Cov_θ(U_X)`; `I_h = Cov(Ê[U_X|h])` with `Ê[U_X|h]` a
linear least-squares fit of `U_X` on the summary `h(X)` (a conservative — lower —
estimate of `I_h`, so passing the bar is a strong result).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_fisher_recovery.py
import math
import numpy as np
import torch
from cdsbi.diagnostics.fisher_recovery import fisher_det_ratio
from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar


def _sufficient_h(sim, x):
    # the oracle sufficient summary (log s², X̄) — det ratio must be ≈ 1
    return sim.oracle_summary(x)


def _lossy_h(sim, x):
    # drop the scale info: keep only X̄, duplicate it — strictly less Fisher info
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_fisher_recovery.py -v`
Expected: FAIL with `ModuleNotFoundError` / `cannot import name 'fisher_det_ratio'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cdsbi/diagnostics/fisher_recovery.py
"""FisherRecovery: efficiency = Fisher-information preservation (theory §11, Cor 12).
Reports det I_h / det I_X at a θ₀, with I_X the true Fisher info (Cov of the autograd
score ∇_θ log p(X|θ)) and I_h = Cov(Ê[U_X|h]), Ê[·|h] a degree-3 POLYNOMIAL LS fit of
the score on the summary. (Linear undershoots: the σ-score is linear in s² but the
feature is log s², so a linear fit caps even a sufficient summary at ~0.80; degree-3
recovers the oracle to ~1.0 — verified: oracle 0.999, σ-dropping 0.10.) Ratio → 1 ⟺
summary sufficient (efficient).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch


def _true_score(simulator, theta0: Sequence[float], x: torch.Tensor) -> torch.Tensor:
    """U_X = ∇_θ log p(X|θ₀), shape (n, d_θ), via autograd of simulator.log_prob."""
    d = len(theta0)
    # build a LEAF tensor with requires_grad (repeat→non-leaf, so .detach() first)
    th = (torch.tensor(theta0, dtype=torch.float32).unsqueeze(0)
          .repeat(x.shape[0], 1).detach().requires_grad_(True))
    lp = simulator.log_prob(x, th)                      # (n,)
    grad = torch.autograd.grad(lp.sum(), th)[0]         # (n, d_θ)
    return grad.reshape(-1, d)


def _cov(a: torch.Tensor) -> np.ndarray:
    a = a - a.mean(dim=0, keepdim=True)
    return (a.T @ a / (a.shape[0] - 1)).detach().cpu().numpy()


def _poly_features(h: np.ndarray, degree: int = 3) -> np.ndarray:
    """[1, h, h², …, h^degree (per-coord), pairwise cross terms] — captures the
    nonlinear sufficiency a linear fit misses."""
    n, d = h.shape
    cols = [np.ones((n, 1)), h]
    for p in range(2, degree + 1):
        cols.append(h ** p)
    for i in range(d):
        for j in range(i + 1, d):
            cols.append((h[:, i] * h[:, j]).reshape(-1, 1))
    return np.concatenate(cols, axis=1)


def fisher_det_ratio(simulator, theta0: Sequence[float],
                     encode_fn: Callable[[object, torch.Tensor], torch.Tensor],
                     n_samples: int = 40000, seed: int = 0) -> float:
    """det I_h / det I_X at θ₀. encode_fn(simulator, x) -> (n, d_θ) summary."""
    rng = np.random.default_rng(seed)
    x = simulator.sample_x_given_theta(np.asarray(theta0), n_samples, rng)
    u = _true_score(simulator, theta0, x)               # (n, d_θ)
    h = encode_fn(simulator, x).detach()                # (n, d_θ)
    I_X = _cov(u)
    # Ê[U_X | h] via degree-3 polynomial LS (linear caps the oracle at ~0.80).
    H = _poly_features(h.cpu().numpy(), degree=3)
    beta, *_ = np.linalg.lstsq(H, u.cpu().numpy(), rcond=None)
    u_hat = torch.from_numpy(H @ beta).float()
    I_h = _cov(u_hat)
    return float(np.linalg.det(I_h) / max(np.linalg.det(I_X), 1e-30))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_fisher_recovery.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/diagnostics/fisher_recovery.py tests/unit/test_fisher_recovery.py
git commit -m "feat(diagnostics): FisherRecovery det I_h/I_X efficiency metric"
```

---

### Task 3: TwoStageCDSBIRunner — Stage 1 (regress + freeze)

**Files:**
- Create: `src/cdsbi/methods/cd_sbi_two_stage.py`
- Test: `tests/integration/test_two_stage_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_two_stage_runner.py
import numpy as np
import torch
from cdsbi.conditioners.moment_regression import MomentRegressionConditioner
from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
from cdsbi.methods.cd_sbi_two_stage import TwoStageCDSBIRunner
from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar


def _make_runner():
    sim = NormalUnknownMeanVar()
    cond = MomentRegressionConditioner(n_iid=sim.n_iid, d_theta=sim.d_theta, target="theta")
    flow = SingleIndexMonotoneFlow(d=sim.d_theta, theta_signs=sim.theta_signs,
                                   feat_signs=sim.feat_signs, hidden=32)
    return sim, TwoStageCDSBIRunner(flow=flow, conditioner=cond, device="cpu")


def test_stage1_regression_recovers_theta_including_log_sigma():
    """The frozen regression summary predicts BOTH coords — crucially log σ (the
    coord M2 collapses). This is the structural no-collapse property."""
    sim, runner = _make_runner()
    cfg = {"lr": 2e-3, "stage1_steps": 1500, "stage2_steps": 0,
           "batch_size": 512, "fresh_batch": True}
    runner.fit(sim, cfg, seed=0)
    rng = np.random.default_rng(1)
    theta, x = sim.sample(4000, rng)
    with torch.no_grad():
        h, _ = runner.conditioner.encode(x)
    # corr of predicted vs true for each coord (coord 0 = log σ, coord 1 = μ)
    for k, name in [(0, "log_sigma"), (1, "mu")]:
        c = np.corrcoef(h[:, k].numpy(), theta[:, k].numpy())[0, 1]
        assert c > 0.8, f"summary failed to recover {name}: corr {c:.3f}"


def test_stage1_freezes_conditioner():
    """After Stage 1, conditioner params have requires_grad=False (cannot be moved
    by Stage 2's loss — the structural anti-collapse guarantee)."""
    sim, runner = _make_runner()
    cfg = {"lr": 2e-3, "stage1_steps": 50, "stage2_steps": 0,
           "batch_size": 256, "fresh_batch": True}
    runner.fit(sim, cfg, seed=0)
    assert all(not p.requires_grad for p in runner.conditioner.parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_two_stage_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cdsbi.methods.cd_sbi_two_stage'`.

- [ ] **Step 3: Write minimal implementation (Stage 1 only; Stage 2 stubbed)**

```python
# src/cdsbi/methods/cd_sbi_two_stage.py
"""TwoStageCDSBIRunner: (1) train a MomentRegressionConditioner by L² regression to
m(θ) and FREEZE it (regression to a function of θ structurally blocks the entropy-
floor cheat — validity-safety; efficiency is asymptotic/BvM, measured by
FisherRecovery, not structural); (2) calibrate the pivot by NF-MLE on the frozen
summary. Inference uses r(θ; frozen-summary(X)) ∈ ℝ^{d_θ}."""
from __future__ import annotations

import time

import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.device import get_device
from cdsbi.flows.base import Guarantee
from cdsbi.losses.base import MonotonicityMismatchError
from cdsbi.losses.nfmle import NFMLELoss
from cdsbi.methods.base import TrainedModel
from cdsbi.reproducibility.seeding import seed_everything


class TwoStageCDSBIRunner:
    def __init__(self, flow, conditioner, loss=None, allow_ablation: bool = False,
                 device: str = "auto"):
        self.flow = flow
        self.conditioner = conditioner       # MomentRegressionConditioner (has regression_target)
        self.loss = loss or NFMLELoss()
        self.allow_ablation = allow_ablation
        self.device = get_device(device)
        if not allow_ablation:
            self.loss.check_guarantees(flow)   # NFMLELoss requires {R1,R2}

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        rngs = seed_everything(seed)
        dev = self.device
        self.flow.to(dev); self.conditioner.to(dev)
        bs = int(config["batch_size"]); lr = float(config["lr"])
        # Stage split: explicit stage1_steps/stage2_steps (tests) override; else split
        # the budget's n_steps by stage1_frac (harness — _recipe_dict passes these
        # through after Task 5's additive edit; default frac 0.4).
        if config.get("stage1_steps") is not None and config.get("stage2_steps") is not None:
            s1, s2 = int(config["stage1_steps"]), int(config["stage2_steps"])
        else:
            n_steps = int(config["n_steps"])
            frac = float(config.get("stage1_frac", 0.4) or 0.4)
            s1 = int(frac * n_steps); s2 = n_steps - s1

        # ---- Stage 1: regression to m(θ), then FREEZE ----
        self.conditioner.train()
        opt1 = torch.optim.Adam(self.conditioner.parameters(), lr=lr)
        for _ in range(s1):
            theta, x = simulator.sample(bs, rngs.train)
            theta, x = theta.to(dev), x.to(dev)
            pred, _ = self.conditioner.encode(x)
            target = self.conditioner.regression_target(theta)
            loss1 = ((pred - target) ** 2).mean()
            opt1.zero_grad(); loss1.backward(); opt1.step()
        for p in self.conditioner.parameters():
            p.requires_grad_(False)
        self.conditioner.eval()

        # ---- Stage 2: NF-MLE pivot on the frozen summary (Task 4) ----
        losses, wall = self._fit_pivot(simulator, config, rngs, dev, s2)

        self.flow.eval()
        flow, conditioner, device = self.flow, self.conditioner, dev

        def pivot_fn(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            theta, x = theta.to(device), x.to(device)
            feats, _ = conditioner.encode(x)
            r, _ = flow.forward(theta, context=feats)
            return r

        def encode_fn(x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                feats, _ = conditioner.encode(x.to(device))
            return feats

        procedure = PivotBasedProcedure(pivot_fn=pivot_fn, d_theta=int(simulator.d_theta),
                                        encode_fn=encode_fn)
        arch_meta = {
            "flow_class": type(self.flow).__name__,
            "loss_class": "NFMLELoss",
            "conditioner_class": type(self.conditioner).__name__,
            "regression_target": getattr(self.conditioner, "target", None),
            "loss_history_tail": losses[-min(100, len(losses)):],
        }
        return TrainedModel(procedure=procedure, state_dict={},
                            final_loss=float(losses[-1]) if losses else float("nan"),
                            n_steps=s1 + s2, wall_clock_sec=wall, arch_metadata=arch_meta)

    def _fit_pivot(self, simulator, config, rngs, dev, n_steps):
        return [], 0.0   # stubbed in this task; implemented in Task 4

    def n_params(self) -> dict:
        backbone = sum(p.numel() for p in self.flow.parameters())
        head = self.conditioner.n_params()
        return {"backbone": backbone, "head": head, "total": backbone + head, "kind": "flow"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_two_stage_runner.py -v`
Expected: 2 passed (`stage2_steps=0` exercises Stage 1 + freeze only).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/methods/cd_sbi_two_stage.py tests/integration/test_two_stage_runner.py
git commit -m "feat(method): TwoStageCDSBIRunner Stage 1 (regress summary + freeze)"
```

---

### Task 4: TwoStageCDSBIRunner — Stage 2 (NF-MLE pivot on frozen summary)

**Files:**
- Modify: `src/cdsbi/methods/cd_sbi_two_stage.py` (replace `_fit_pivot`)
- Test: `tests/integration/test_two_stage_runner.py` (add a calibration test)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/integration/test_two_stage_runner.py
import math
from scipy.stats import chi2


def test_full_two_stage_calibrates():
    """End-to-end: regress→freeze→NF-MLE yields a pivot calibrated at the truth
    (coverage ≈ α) — validity (Thm 1)."""
    sim = NormalUnknownMeanVar()
    cond = MomentRegressionConditioner(n_iid=sim.n_iid, d_theta=sim.d_theta, target="theta")
    flow = SingleIndexMonotoneFlow(d=sim.d_theta, theta_signs=sim.theta_signs,
                                   feat_signs=sim.feat_signs, hidden=32)
    runner = TwoStageCDSBIRunner(flow=flow, conditioner=cond, device="cpu")
    cfg = {"lr": 2e-3, "stage1_steps": 1500, "stage2_steps": 3000,
           "batch_size": 512, "fresh_batch": True}
    trained = runner.fit(sim, cfg, seed=0)
    proc = trained.procedure
    rng = np.random.default_rng(2)
    theta0 = np.array([math.log(1.0), 0.0])
    xv = sim.sample_x_given_theta(theta0, 20000, rng)
    th = torch.tensor(theta0, dtype=torch.float32).expand(20000, -1)
    r = proc.pivot_fn(th, xv)
    sq = (r ** 2).sum(-1).numpy()
    for a in (0.8, 0.9):
        emp = float((sq <= chi2.ppf(a, df=2)).mean())
        assert abs(emp - a) < 0.06, f"miscovers at α={a}: {emp:.3f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_two_stage_runner.py::test_full_two_stage_calibrates -v`
Expected: FAIL — `_fit_pivot` is stubbed, the pivot is untrained, coverage is off.

- [ ] **Step 3: Replace `_fit_pivot` with the real NF-MLE training loop**

```python
    def _fit_pivot(self, simulator, config, rngs, dev, n_steps):
        self.flow.train()
        opt = torch.optim.Adam(self.flow.parameters(), lr=float(config["lr"]))
        grad_clip = float(config.get("grad_clip_norm", 5.0))
        losses = []
        t0 = time.time()
        for _ in range(n_steps):
            theta, x = simulator.sample(int(config["batch_size"]), rngs.train)
            theta, x = theta.to(dev), x.to(dev)
            with torch.no_grad():
                feats, _ = self.conditioner.encode(x)
            r, log_det = self.flow.forward(theta, context=feats)
            loss_val = self.loss(r, log_det)
            opt.zero_grad(); loss_val.backward()
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=grad_clip)
            opt.step()
            losses.append(loss_val.item())
        return losses, time.time() - t0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_two_stage_runner.py -v`
Expected: 3 passed (~30–60s on CPU). Stage 2 reuses `config["lr"]` with a constant
schedule (no warmup) — if the calibration band (±0.06) is flaky, raise
`stage2_steps` to 5000 or `lr` to 3e-3 (a recipe tune, not a code bug).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/methods/cd_sbi_two_stage.py tests/integration/test_two_stage_runner.py
git commit -m "feat(method): TwoStageCDSBIRunner Stage 2 (NF-MLE pivot on frozen summary)"
```

---

### Task 5: Hydra wiring (config-only + a 3-line `_recipe_dict` pass-through)

**Key fact (verified against `run.py`):** `_build_method` routes on `method.name`.
Setting `name: cd_sbi` enters the generic branch (`run.py:195`) that builds
flow+conditioner+loss and calls `_instantiate(runner_class, flow=, conditioner=,
loss=, allow_ablation=, device=)` — which **matches `TwoStageCDSBIRunner.__init__`
exactly**. So we need **no bespoke dispatch**: a method config with `name: cd_sbi`
but a different `runner_class`, a conditioner config group, and an experiment config.
The only code change is making the two-stage split reachable through `_recipe_dict`
(which today returns a fixed key set, `run.py:_recipe_dict`).

**Files:**
- Create: `configs/method/cd_sbi_moment.yaml`, `configs/conditioner/moment_regression.yaml`,
  `configs/experiment/moment_mu_sigma.yaml`
- Modify: `src/cdsbi/experiments/run.py` (`_recipe_dict` — 3 additive keys)

- [ ] **Step 1: Add the two-stage keys to `_recipe_dict` (additive, safe for all methods)**

In `src/cdsbi/experiments/run.py`, inside `_recipe_dict`'s returned dict, after the
`"grad_clip_norm": ...` line, add:

```python
        "stage1_steps": OmegaConf.select(t, "stage1_steps", default=None),
        "stage2_steps": OmegaConf.select(t, "stage2_steps", default=None),
        "stage1_frac": float(OmegaConf.select(t, "stage1_frac", default=0.4)),
```

(Other methods ignore these keys; `TwoStageCDSBIRunner.fit` reads them, falling back
to splitting `n_steps` by `stage1_frac`.)

- [ ] **Step 2: Write the three configs**

```yaml
# configs/method/cd_sbi_moment.yaml
# Two-stage moment-summary CD-SBI. name=cd_sbi routes through the generic branch;
# runner_class swaps in the two-stage runner. R1-on flow for Phase 1.
name: cd_sbi
runner_class: cdsbi.methods.cd_sbi_two_stage.TwoStageCDSBIRunner
flow: single_index_monotone        # method.flow guard (cb1e08e): set explicitly
loss: nfmle
allow_ablation: false              # R1-on flow gives {R1,R2}; NFMLELoss check passes
```

```yaml
# configs/conditioner/moment_regression.yaml
name: moment_regression
_target_: cdsbi.conditioners.moment_regression.MomentRegressionConditioner
n_iid: 10
d_theta: 2
target: theta                       # m(θ)=θ (posterior mean) for the regular case
```

```yaml
# configs/experiment/moment_mu_sigma.yaml
# @package _global_
defaults:
  - override /target: normal_mu_sigma
  - override /flow: single_index_monotone
  - override /conditioner: moment_regression
  - override /method: cd_sbi_moment
  - override /budget: medium
method:
  flow: single_index_monotone       # ensure the cb1e08e guard takes
training:
  stage1_steps: 4000
  stage2_steps: 8000
experiment:
  name: moment_mu_sigma
  n_eval: 6000
  eval_thetas_interior:             # 3×3 (log σ, μ) grid (same as mu_sigma_replication)
    - [-0.69, -2.0]
    - [-0.69, 0.0]
    - [-0.69, 2.0]
    - [0.0, -2.0]
    - [0.0, 0.0]
    - [0.0, 2.0]
    - [0.69, -2.0]
    - [0.69, 0.0]
    - [0.69, 2.0]
  eval_thetas_edge: []
  alpha_grid: [0.5, 0.68, 0.9, 0.95]
  n_eval_per_theta: 2000
  joint_mahalanobis_n_per_theta: 2000
```

(The `normal_mu_sigma` target + experiment defaults bring the (μ,σ²) diagnostic
battery — `Coverage`, `MarginalCDRecovery` — so Task 7's `coverage_error_max` and
`marginal_cd_sigma_ks` columns are produced, exactly as in `mu_sigma_replication`.)

- [ ] **Step 3: Smoke-run the experiment for 1 short seed**

Run:
```bash
python -m cdsbi.experiments.run experiment=moment_mu_sigma seed=0 \
  training.stage1_steps=200 training.stage2_steps=200 \
  hydra.run.dir=/tmp/moment_smoke
```
Expected: exits 0; `/tmp/moment_smoke/model.pt` exists. Verify the wiring took:
```bash
python -c "import torch; m=torch.load('/tmp/moment_smoke/model.pt',weights_only=False)['arch_metadata']; print(m['conditioner_class'], m['flow_class'])"
```
Expected: `MomentRegressionConditioner SingleIndexMonotoneFlow`.

- [ ] **Step 4: Commit**

```bash
git add configs/method/cd_sbi_moment.yaml configs/conditioner/moment_regression.yaml \
        configs/experiment/moment_mu_sigma.yaml src/cdsbi/experiments/run.py
git commit -m "feat(experiments): wire two-stage moment-summary method (config + recipe pass-through)"
```

---

### Task 6: Integration — no-collapse + efficiency vs oracle (in-process)

**Files:**
- Modify: `tests/integration/test_two_stage_runner.py` (add the comparison test)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/integration/test_two_stage_runner.py
from cdsbi.conditioners.sufficient_stat import SufficientStatConditioner
from cdsbi.diagnostics.fisher_recovery import fisher_det_ratio


def test_learned_summary_efficiency_approaches_oracle_and_no_collapse():
    """The frozen learned summary recovers σ-info (no collapse) and its Fisher-info
    ratio approaches the oracle — efficiency is MEASURED here, not assumed."""
    sim = NormalUnknownMeanVar()
    cond = MomentRegressionConditioner(n_iid=sim.n_iid, d_theta=sim.d_theta, target="theta")
    flow = SingleIndexMonotoneFlow(d=sim.d_theta, theta_signs=sim.theta_signs,
                                   feat_signs=sim.feat_signs, hidden=32)
    runner = TwoStageCDSBIRunner(flow=flow, conditioner=cond, device="cpu")
    runner.fit(sim, {"lr": 2e-3, "stage1_steps": 3000, "stage2_steps": 0,
                     "batch_size": 512, "fresh_batch": True}, seed=0)

    def learned_h(simulator, x):
        with torch.no_grad():
            h, _ = runner.conditioner.encode(x)
        return h

    learned_ratio = fisher_det_ratio(sim, (math.log(1.0), 0.0), learned_h, seed=0)
    # head-to-head: the oracle sufficient summary under the SAME estimator (→ ~1.0
    # with the degree-3 fit). The learned summary should be efficient AND approach it.
    oracle_ratio = fisher_det_ratio(
        sim, (math.log(1.0), 0.0), lambda s, x: s.oracle_summary(x), seed=0)
    assert learned_ratio > 0.8, f"learned summary not efficient: {learned_ratio:.3f}"
    assert learned_ratio > 0.85 * oracle_ratio, (
        f"learned ({learned_ratio:.3f}) should approach oracle ({oracle_ratio:.3f})")
```

- [ ] **Step 2: Run test to verify it fails (or errors) before any fix**

Run: `pytest tests/integration/test_two_stage_runner.py::test_learned_summary_efficiency_approaches_oracle_and_no_collapse -v`
Expected: PASS if Tasks 1–4 are correct; if it FAILS with ratio < 0.8, the Stage-1
recipe needs more steps (raise `stage1_steps` to 5000) — this is a recipe tune, not
a code bug. (The test encodes the Phase-1 efficiency bar.)

- [ ] **Step 3: If failing, bump the Stage-1 recipe**

Edit the test's `stage1_steps` to `5000` and re-run; the learned ratio should clear
0.8. Document the chosen `stage1_steps` in `configs/experiment/moment_mu_sigma.yaml`.

- [ ] **Step 4: Run the full integration file**

Run: `pytest tests/integration/test_two_stage_runner.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_two_stage_runner.py configs/experiment/moment_mu_sigma.yaml
git commit -m "test(integration): two-stage learned summary efficiency vs oracle (no collapse)"
```

---

### Task 7: Intensive replication — Phase-1 validity + no-collapse on (μ,σ²)

**Files:**
- Create: `tests/intensive/test_replicate_moment_mu_sigma.py`

Establishes the Phase-1 success criteria via the Hydra harness across seeds: validity
(`coverage_error_max ≤ 0.06`) and **no-collapse** (the σ marginal-CD is calibrated,
`marginal_cd_sigma_ks ≤ 0.10` — the learned summary recovered the scale info that the
joint-NF-MLE M2 collapses). The M2 collapse is the *documented contrast*
(`tests/integration/test_mu_sigma_stage_b_smoke.py`), not re-run here. Efficiency-vs-
oracle (`det I_h/I_X`) is covered in-process by Task 6. Marked `intensive` (minutes).

*Note (the no-collapse metric's assumption):* `marginal_cd_sigma_ks` reads `r[:,0]`
as the σ-pivot (`marginal_cd_spec.scale_coord=0`). With `target="theta"` the summary's
coord 0 ≈ `log σ̂` and the flow consumes features in θ-order, so `r[:,0]` is the log-σ
pivot — the alignment holds, but the no-collapse reading rides on it.

- [ ] **Step 1: Write the replication test**

```python
# tests/intensive/test_replicate_moment_mu_sigma.py
"""Intensive: Phase-1 regular consistency on (μ,σ²). Two-stage moment-summary should
be valid + efficient + no-collapse, matching the oracle and beating the M2 collapse."""
from pathlib import Path
import subprocess
import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.intensive
def test_replicate_moment_mu_sigma(tmp_path):
    out = tmp_path / "moment"
    for seed in range(3):
        cmd = ["python", "-m", "cdsbi.experiments.run",
               f"hydra.run.dir={out}/seed_{seed}",
               "experiment=moment_mu_sigma", f"seed={seed}", "training.fresh_batch=true"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, f"seed {seed} failed:\n{r.stderr[-2000:]}"

    from cdsbi.analysis.loaders import load_runs
    df = load_runs(str(out / "*"))
    assert len(df) == 3
    print(df[["coverage_error_max", "marginal_cd_sigma_ks"]].to_string())
    # validity + no-collapse (σ-recovery shows in the σ marginal-CD KS)
    assert df["coverage_error_max"].mean() <= 0.06
    assert df["marginal_cd_sigma_ks"].mean() <= 0.10
```

- [ ] **Step 2: Run it (opt-in)**

Run: `pytest tests/intensive/test_replicate_moment_mu_sigma.py -m intensive -v -s`
Expected: PASS — coverage_error_max ≤ 0.06 (validity), σ marginal-CD KS ≤ 0.10
(no collapse). Tune `stage1_steps`/`stage2_steps` in the experiment config if a band
is missed (recipe, not code).

- [ ] **Step 3: Confirm the fast suite is still green**

Run: `pytest -q -m "not intensive"`
Expected: all prior tests + the new unit/integration tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/intensive/test_replicate_moment_mu_sigma.py
git commit -m "test(intensive): Phase-1 moment-summary replication on (μ,σ²)"
```

---

## Self-review (run after writing; checklist)

- **Spec coverage (Phase 1):** moment-regression summary (Task 1), sequential
  regress→freeze→NF-MLE (Tasks 3–4), FisherRecovery efficiency metric (Task 2),
  validity (Task 4 calibration), no-collapse (Task 3 σ-recovery + Task 6), efficiency
  vs oracle (Task 6), the bake-off replication (Task 7), Hydra wiring (Task 5). ✓
- **Out of Phase-1 scope (correctly absent):** `NonMonotonePivotFlow`,
  disconnected-set extractor, `Disconnectedness`, sign model, S2b reweighting,
  modality detector — all P2+ (separate plans).
- **Open recipe knob:** `stage1_steps`/`stage2_steps` are tuned to clear the bars
  (Tasks 6–7 flag this as a recipe tune, not a code defect).
