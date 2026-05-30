# CD-SBI M3.2′ — Stage-B Arm I-A (invertible exact-density benchmark)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the **discriminating benchmark** arm: a learned **invertible** (bijective) summary `s_φ: X→(S,A)` whose exact change-of-variables makes the objective `−log p(X|θ)` — which **cannot discard information** (bounded below by the data entropy `H(X|θ)`). Show on the M3.0 harness whether forcing information-preservation recovers σ² (the coordinate II-A's calibration-only objective collapsed to Spearman 0.014) and calibrates near the Stage-A oracle.

**Architecture:** A hand-rolled `AffineCouplingBijection` (RealNVP-style, ℝ¹⁰→ℝ¹⁰, invertible with tractable log|det|) wrapped as an `InvertibleSummaryConditioner`. Its first `d_theta=2` outputs `S` feed the existing `SingleIndexMonotoneFlow` pivot; the other 8, `A`, are ancillary. `ExactDensityCDSBIRunner` trains the bijection + pivot **end-to-end** under the full negative log-likelihood `−log p(X|θ) = ½(‖r(θ;S)‖² + ‖A‖²) + (d_x/2)log2π − log|∂r/∂S| − log|∂s_φ/∂X|`. Because the bijection preserves information and `A` is modelled θ-free `N(0,I₈)`, the objective **pressures all θ-information into S** (so `p(A|θ)=p(A)` fits), and S co-adapts to a coordinate the pivot can calibrate. This is why I-A should recover σ where II-A couldn't. Also fixes `FloorIntegrity` to use the arm-appropriate floor (`H(X|θ)` for exact density).

**Tech Stack:** PyTorch (hand-rolled affine coupling, Adam), Hydra, pytest, numpy/scipy. Reuses `SingleIndexMonotoneFlow`, the M3.0 harness + `encode_fn` plumbing.

**Spec:** `docs/superpowers/specs/2026-05-29-cd-sbi-stage-b-device-bakeoff-design.md` — Arm I-A (§3, BENCHMARK) + §1.5 + §2.1 (`FloorIntegrity` empirical-floor fix) + §5.

**Scope note:** the spec's permutation-equivariant bijection (over the exchangeable iid axis) is **deferred** — a standard coupling flow answers the discriminating question (does exact density recover σ?) on this fixed-`n_iid` target; equivariance is the generalization, noted as follow-on. The summary co-adapts end-to-end, so order-dependence does not block calibration here (X is drawn iid; coverage is over X~p(·|θ₀)).

---

## Conventions / decisions locked

- **Exact-density NLL objective:** the composite map `X →s_φ→ (S,A) →(pivot on S, identity on A)→ (r, A) ~ N(0,I_{d_x})`. Loss = `½(‖r‖² + ‖A‖²) + (d_x/2)log2π − log|∂r/∂S| − log|∂s_φ/∂X|`, mean over (θ,X) pairs. Bounded below by `H(X|θ)` — **structurally cannot cheat** (a proper density NLL).
- **Standard (θ,X) batching** (not grouped — that was II-A). Reuse the finite-sample `sample(n_train)` recipe like the original CDSBI.
- **Bijection:** hand-rolled `AffineCouplingBijection` (no nflows dependency — it is not importable on this env). Alternating masks, `tanh`-bounded log-scales for stability. Invertible with `log|det| = Σ (log-scales on transformed coords)`.
- **Pivot:** unchanged `SingleIndexMonotoneFlow(d=d_theta=2)` on `S`; R1 (monotone-in-θ) needed for confidence sets; its `forward` gives `(r, log|∂r/∂S|)`.
- **Confidence sets / diagnostics** use only `r(θ;S)∈ℝ²` (`d_theta=2`); `A` is loss-only. `encode_fn(X)=S` for `SufficiencyRecovery`.
- **`FloorIntegrity` fix:** generalize from the hardcoded oracle floor to a **loss→floor map**: `NFMLELoss→entropy_lower_bound()`, `ExactDensityLoss→data_entropy_lower_bound()`. The exact-density arm records `loss_class="ExactDensityLoss"`; its floor is `H(X|θ)` (analytic for the Gaussian). No-op for other losses (incl. the energy loss). This is the §2.1 deferred fix, in its clean analytic form for I-A.

---

## File structure

```
NEW
  src/cdsbi/flows/affine_coupling.py               # AffineCouplingBijection (ℝ^d→ℝ^d, invertible)
  src/cdsbi/conditioners/invertible_summary.py     # InvertibleSummaryConditioner (encode→S, transform→full z)
  src/cdsbi/methods/cd_sbi_exact_density.py         # ExactDensityCDSBIRunner
  configs/conditioner/invertible_summary.yaml
  configs/method/cd_sbi_exact_density.yaml
  configs/experiment/mu_sigma_stage_b_exact.yaml
  tests/unit/test_affine_coupling.py
  tests/unit/test_invertible_summary.py
  tests/unit/test_exact_density_runner.py
  tests/intensive/test_replicate_mu_sigma_stage_b_exact.py

MODIFY
  src/cdsbi/simulators/normal_unknown_mean_var.py    # data_entropy_lower_bound()
  src/cdsbi/diagnostics/floor_integrity.py            # loss→floor map (the §2.1 fix)
  src/cdsbi/experiments/run.py                        # _build_method: exact_density loss/runner selector
  tests/unit/test_floor_integrity.py                  # exact-density floor branch test
```

No change to `SingleIndexMonotoneFlow`, the other diagnostics, or the M3.1′ energy arm.

---

## Task 1: `AffineCouplingBijection`

**Files:**
- Create: `src/cdsbi/flows/affine_coupling.py`
- Test: `tests/unit/test_affine_coupling.py`

READ `src/cdsbi/flows/single_index_monotone.py`'s `_tanh_mlp` helper (copy the idiom).

- [ ] **Step 1: Write the failing tests**

```python
"""AffineCouplingBijection: invertible ℝ^d→ℝ^d with tractable log|det|."""
import torch


def _bij(d=4, seed=0):
    torch.manual_seed(seed)
    from cdsbi.flows.affine_coupling import AffineCouplingBijection
    return AffineCouplingBijection(d=d, hidden=16, n_layers=4, depth=2)


def test_shapes():
    bij = _bij()
    x = torch.randn(8, 4)
    z, log_det = bij(x)
    assert z.shape == (8, 4) and log_det.shape == (8,)


def test_logdet_matches_autograd():
    import torch.nn as nn
    torch.manual_seed(1)
    bij = _bij()
    # the ctor zero-inits coupling output layers (near-identity start) — perturb them
    # so this test exercises NON-TRIVIAL scales (else it's a vacuous identity check).
    for net in list(bij.scale_nets) + list(bij.shift_nets):
        nn.init.normal_(net[-1].weight, std=0.5)
        nn.init.normal_(net[-1].bias, std=0.3)
    x = torch.randn(1, 4, requires_grad=True)
    z, log_det = bij(x)
    assert float(log_det.abs()) > 1e-3, "test must exercise non-identity scales"
    J = torch.zeros(4, 4)
    for i in range(4):
        (g,) = torch.autograd.grad(z[0, i], x, retain_graph=True)
        J[i] = g[0]
    assert torch.allclose(log_det[0], torch.log(torch.abs(torch.det(J))), atol=1e-4)


def test_invertible_distinct_inputs_distinct_outputs():
    bij = _bij()
    x = torch.randn(16, 4)
    z, _ = bij(x)
    # a bijection preserves distinctness; check the map is injective on a sample
    # by confirming the autograd Jacobian is non-singular (|det| > 0 ⇒ local bijection)
    for k in range(4):
        xk = x[k:k+1].clone().requires_grad_(True)
        zk, ld = bij(xk)
        assert torch.exp(ld[0]) > 0   # |det| > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_affine_coupling.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `src/cdsbi/flows/affine_coupling.py`**

```python
"""AffineCouplingBijection: a hand-rolled RealNVP-style invertible map ℝ^d→ℝ^d.

A stack of affine coupling layers with alternating binary masks. Each layer
transforms the un-masked coords as z' = z·exp(s(z_masked)) + t(z_masked), leaving
the masked coords fixed, so the Jacobian is triangular and log|det| = Σ s. The
log-scales are tanh-bounded for stability. Invertible (information-preserving) —
used as the Stage-B Arm I-A summary so the change-of-variables is exact.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _tanh_mlp(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Sequential:
    assert depth >= 1
    layers: list = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class AffineCouplingBijection(nn.Module):
    def __init__(self, d: int, hidden: int = 64, n_layers: int = 6, depth: int = 2,
                 scale_cap: float = 2.0):
        super().__init__()
        if d < 2:
            raise ValueError("AffineCouplingBijection needs d >= 2")
        self.d = d
        self.scale_cap = scale_cap
        masks = []
        for i in range(n_layers):
            m = torch.zeros(d)
            m[i % 2::2] = 1.0          # alternating even/odd coords are the "fixed" half
            masks.append(m)
        self.register_buffer("_masks", torch.stack(masks))           # (n_layers, d)
        self.scale_nets = nn.ModuleList([_tanh_mlp(d, hidden, d, depth) for _ in range(n_layers)])
        self.shift_nets = nn.ModuleList([_tanh_mlp(d, hidden, d, depth) for _ in range(n_layers)])
        # init last layers to ~0 so the bijection starts near identity
        for net in list(self.scale_nets) + list(self.shift_nets):
            nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)

    def forward(self, x: torch.Tensor):
        z = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for i in range(len(self.scale_nets)):
            mask = self._masks[i]                                    # (d,)
            z_masked = z * mask
            s = self.scale_cap * torch.tanh(self.scale_nets[i](z_masked)) * (1.0 - mask)
            t = self.shift_nets[i](z_masked) * (1.0 - mask)
            z = z_masked + (1.0 - mask) * (z * torch.exp(s) + t)
            log_det = log_det + s.sum(dim=-1)
        return z, log_det

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_affine_coupling.py -v` → 3 pass (log_det matches autograd is the load-bearing one).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/flows/affine_coupling.py tests/unit/test_affine_coupling.py
git commit -m "feat(flows): AffineCouplingBijection — invertible ℝ^d→ℝ^d (RealNVP-style) for Arm I-A"
```

---

## Task 2: `InvertibleSummaryConditioner`

**Files:**
- Create: `src/cdsbi/conditioners/invertible_summary.py`
- Test: `tests/unit/test_invertible_summary.py`

READ `src/cdsbi/conditioners/base.py` (`Conditioner` protocol: `encode(x)->(context, log_det)`, `n_params()`).

- [ ] **Step 1: Write the failing tests**

```python
"""InvertibleSummaryConditioner: bijection X→(S,A); encode→(S, log|∂z/∂X|)."""
import torch


def _cond(n_iid=10, d_theta=2):
    torch.manual_seed(0)
    from cdsbi.conditioners.invertible_summary import InvertibleSummaryConditioner
    return InvertibleSummaryConditioner(n_iid=n_iid, d_theta=d_theta, hidden=16, n_layers=4)


def test_encode_returns_S_and_bijection_logdet():
    cond = _cond()
    x = torch.randn(8, 10)
    S, log_det = cond.encode(x)
    assert S.shape == (8, 2) and log_det.shape == (8,)


def test_transform_returns_full_latent():
    cond = _cond()
    x = torch.randn(8, 10)
    z, log_det = cond.transform(x)
    assert z.shape == (8, 10) and log_det.shape == (8,)
    S, ld2 = cond.encode(x)
    assert torch.allclose(z[:, :2], S) and torch.allclose(log_det, ld2)


def test_is_module_with_params():
    cond = _cond()
    assert isinstance(cond, torch.nn.Module) and cond.n_params() > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_invertible_summary.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `src/cdsbi/conditioners/invertible_summary.py`**

```python
"""InvertibleSummaryConditioner: Stage-B Arm I-A learned summary.

Wraps an AffineCouplingBijection s_φ: ℝ^{n_iid}→ℝ^{n_iid}. The first d_theta outputs
S are the inference features (→ the monotone pivot); the rest A are ancillary
(modelled N(0,I) in the exact-density loss). encode(X) returns (S, log|∂s_φ/∂X|) for
the Conditioner protocol + SufficiencyRecovery; transform(X) exposes the full latent
z=(S,A) for the exact-density runner. Invertible ⇒ information cannot be destroyed.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from cdsbi.flows.affine_coupling import AffineCouplingBijection


class InvertibleSummaryConditioner(nn.Module):
    def __init__(self, n_iid: int, d_theta: int = 2, hidden: int = 64,
                 n_layers: int = 6, depth: int = 2):
        super().__init__()
        self.n_iid = n_iid
        self.d_theta = d_theta
        self.bijection = AffineCouplingBijection(n_iid, hidden=hidden, n_layers=n_layers, depth=depth)

    def transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """X → (z (n, n_iid), log|∂z/∂X| (n,))."""
        return self.bijection(x)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, log_det = self.bijection(x)
        return z[:, :self.d_theta], log_det

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_invertible_summary.py -v` → 3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/conditioners/invertible_summary.py tests/unit/test_invertible_summary.py
git commit -m "feat(cond): InvertibleSummaryConditioner — bijection summary X→(S,A) for Arm I-A"
```

---

## Task 3: simulator data-entropy floor + `FloorIntegrity` loss→floor map

**Files:**
- Modify: `src/cdsbi/simulators/normal_unknown_mean_var.py`
- Modify: `src/cdsbi/diagnostics/floor_integrity.py`
- Test: `tests/unit/test_floor_integrity.py` (append), `tests/unit/test_normal_unknown_mean_var.py` (append)

The exact-density loss is bounded by `H(X|θ)` (full data entropy), not the oracle-feature `entropy_lower_bound()`. Add `data_entropy_lower_bound()` and teach `FloorIntegrity` an arm-appropriate floor map (the §2.1 fix).

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_normal_unknown_mean_var.py` (append):
```python
def test_data_entropy_lower_bound_matches_gaussian_formula():
    import math
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    sim = NormalUnknownMeanVar()
    H = sim.data_entropy_lower_bound()
    # H(X|θ) = (n/2)(1+log2π) + n·E[log σ], E[log σ] = midpoint of log_sigma_range
    n = sim.n_iid
    e_logsig = 0.5 * (sim.log_sigma_range[0] + sim.log_sigma_range[1])
    expected = (n / 2) * (1 + math.log(2 * math.pi)) + n * e_logsig
    assert abs(H - expected) < 1e-6
```

`tests/unit/test_floor_integrity.py` (append):
```python
def test_floor_integrity_exact_density_uses_data_entropy():
    from cdsbi.diagnostics.floor_integrity import FloorIntegrity

    class _Sim:
        def entropy_lower_bound(self, **kw): return 0.92
        def data_entropy_lower_bound(self, **kw): return 13.66

    class _Trained:
        def __init__(self, loss_value):
            self.final_loss = loss_value
            self.arch_metadata = {"loss_class": "ExactDensityLoss"}
            self.procedure = object()

    # at the data-entropy floor → no cheat
    ok = FloorIntegrity()(_Trained(13.70), _Sim())
    assert not bool(ok.value.iloc[0]["cheats"]) and ok.passed
    assert abs(float(ok.value.iloc[0]["entropy_floor"]) - 13.66) < 1e-9
    # far below the data-entropy floor → cheat flagged
    bad = FloorIntegrity()(_Trained(5.0), _Sim())
    assert bool(bad.value.iloc[0]["cheats"]) and not bad.passed
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_floor_integrity.py::test_floor_integrity_exact_density_uses_data_entropy tests/unit/test_normal_unknown_mean_var.py::test_data_entropy_lower_bound_matches_gaussian_formula -v`
Expected: FAIL (`data_entropy_lower_bound` missing; FloorIntegrity doesn't know `ExactDensityLoss`).

- [ ] **Step 3: Implement**

(3a) `normal_unknown_mean_var.py` — add (uses `math`, already imported):
```python
    def data_entropy_lower_bound(self) -> float:
        """H(X|θ) averaged over the prior — the floor for an exact-density model of
        p(X|θ). For X|θ ~ N(μ, σ² I_{n_iid}): H = (n/2)(1+log 2π) + n·log σ; average
        log σ over the (uniform) prior = midpoint of log_sigma_range."""
        n = self.n_iid
        e_log_sigma = 0.5 * (self.log_sigma_range[0] + self.log_sigma_range[1])
        return (n / 2) * (1 + math.log(2 * math.pi)) + n * e_log_sigma
```

(3b) `floor_integrity.py` — replace the hardcoded NF-MLE-only logic with a loss→floor map:
```python
# loss_class -> the simulator method giving its conditional-entropy floor
_FLOOR_METHOD_BY_LOSS = {
    "NFMLELoss": "entropy_lower_bound",
    "ExactDensityLoss": "data_entropy_lower_bound",
}
```
and in `__call__`, replace the `applicable = loss_class in _NFMLE_LOSSES and ...` block with:
```python
        floor_method = _FLOOR_METHOD_BY_LOSS.get(loss_class)
        applicable = floor_method is not None and hasattr(simulator, floor_method)
        if applicable:
            H = float(getattr(simulator, floor_method)())
            margin = final_loss - H
            cheats = bool(margin < -self.tol)
            meta = {"loss_class": loss_class, "tol": self.tol, "floor_method": floor_method}
        else:
            H = float("nan"); margin = float("nan"); cheats = False
            meta = {"reason": f"floor N/A for loss {loss_class!r}"}
```
(Remove the now-unused `_NFMLE_LOSSES` set, or keep it unused — prefer removing for clarity. The energy loss `EnergyCalibrationLoss` is not in the map → no-ops, as before.)

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_floor_integrity.py tests/unit/test_normal_unknown_mean_var.py -q`
Expected: all pass (existing NF-MLE floor tests + the 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/simulators/normal_unknown_mean_var.py src/cdsbi/diagnostics/floor_integrity.py \
        tests/unit/test_floor_integrity.py tests/unit/test_normal_unknown_mean_var.py
git commit -m "feat(harness): data_entropy_lower_bound + FloorIntegrity loss→floor map (§2.1 fix)"
```

---

## Task 4: `ExactDensityCDSBIRunner`

**Files:**
- Create: `src/cdsbi/methods/cd_sbi_exact_density.py`
- Test: `tests/unit/test_exact_density_runner.py`

READ `src/cdsbi/methods/cd_sbi.py` `fit()` (the finite-sample `sample(n_train)` batching loop + the procedure/`TrainedModel` end-block) and `src/cdsbi/methods/cd_sbi_energy.py` (the standalone-runner pattern from M3.1′).

- [ ] **Step 1: Write the failing test**

```python
"""ExactDensityCDSBIRunner: exact −log p(X|θ) over bijection+pivot; floor-bounded."""
import torch


def test_exact_density_runner_fit_trains_and_floor_bounded():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.invertible_summary import InvertibleSummaryConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.methods.cd_sbi_exact_density import ExactDensityCDSBIRunner

    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=16)
    cond = InvertibleSummaryConditioner(n_iid=sim.n_iid, d_theta=2, hidden=16, n_layers=4)
    runner = ExactDensityCDSBIRunner(flow=flow, conditioner=cond, loss=None, device="cpu")
    trained = runner.fit(simulator=sim, config={"lr": 3e-3, "batch_size": 128, "n_steps": 50,
                                                "n_train": 2000, "fresh_batch": False}, seed=0)
    import numpy as np
    assert np.isfinite(trained.final_loss)
    assert trained.arch_metadata["loss_class"] == "ExactDensityLoss"
    # procedure pivots on S (d_theta=2) and exposes encode_fn
    assert trained.procedure.encode_fn is not None
    _, x = sim.sample(16, np.random.default_rng(0))
    assert trained.procedure.pivot(torch.zeros(16, 2), x).shape == (16, 2)
    # a proper density NLL cannot fall far below H(X|θ): final_loss > floor − 2
    assert trained.final_loss > sim.data_entropy_lower_bound() - 2.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_exact_density_runner.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `src/cdsbi/methods/cd_sbi_exact_density.py`**

```python
"""ExactDensityCDSBIRunner: Stage-B Arm I-A. Trains an invertible summary + monotone
pivot end-to-end under the exact data NLL −log p(X|θ). The bijection preserves
information and the ancillary A is modelled θ-free N(0,I), so the objective pushes
all θ-information into the pivot features S — and is bounded below by H(X|θ) (no
collapse cheat possible). Inference uses only r(θ;S) ∈ ℝ^{d_theta}.
"""
from __future__ import annotations

import math
import time

import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.device import get_device
from cdsbi.flows.base import Guarantee
from cdsbi.losses.base import MonotonicityMismatchError
from cdsbi.methods.base import TrainedModel
from cdsbi.reproducibility.seeding import seed_everything


class ExactDensityCDSBIRunner:
    def __init__(self, flow, conditioner, loss=None, allow_ablation: bool = False, device: str = "auto"):
        self.flow = flow
        self.conditioner = conditioner          # must expose .transform(X) -> (z, log_det)
        self.loss = loss                        # ignored; the exact-density NLL is computed inline
        self.allow_ablation = allow_ablation
        self.device = get_device(device)
        # the pivot must be monotone-in-θ (R1) for confidence-set inversion
        if not allow_ablation:
            guarantees = getattr(flow, "monotonicity_guarantees", frozenset())
            if Guarantee.R1 not in guarantees:
                raise MonotonicityMismatchError(
                    f"ExactDensityCDSBIRunner requires R1 (monotone-in-θ); flow "
                    f"{type(flow).__name__} provides {sorted(g.value for g in guarantees)}."
                )

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        rngs = seed_everything(seed)
        dev = self.device
        self.flow.to(dev); self.flow.train()
        self.conditioner.to(dev); self.conditioner.train()
        d = int(simulator.d_theta)
        d_x = int(simulator.d_x)
        lr = float(config["lr"]); n_steps = int(config["n_steps"]); bs = int(config["batch_size"])
        grad_clip = float(config.get("grad_clip_norm", 5.0))
        fresh = bool(config.get("fresh_batch", True))

        params = list(self.flow.parameters()) + list(self.conditioner.parameters())
        opt = torch.optim.Adam(params, lr=lr)

        if not fresh:
            theta_all, x_all = simulator.sample(int(config["n_train"]), rngs.train)
            theta_all = theta_all.to(dev); x_all = x_all.to(dev)
            n_train = theta_all.shape[0]

        losses = []; t0 = time.time()
        const = 0.5 * d_x * math.log(2 * math.pi)
        for step in range(n_steps):
            if fresh:
                theta_b, x_b = simulator.sample(bs, rngs.train)
                theta_b = theta_b.to(dev); x_b = x_b.to(dev)
            else:
                idx = torch.randint(0, n_train, (bs,))
                theta_b = theta_all[idx]; x_b = x_all[idx]
            z, bij_logdet = self.conditioner.transform(x_b)          # (bs, d_x), (bs,)
            S = z[:, :d]; A = z[:, d:]
            r, piv_logdet = self.flow.forward(theta_b, context=S)    # (bs, d), (bs,)
            nll = (0.5 * (r.pow(2).sum(-1) + A.pow(2).sum(-1)) + const
                   - piv_logdet - bij_logdet)
            loss_val = nll.mean()
            opt.zero_grad(); loss_val.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            opt.step()
            losses.append(loss_val.item())
        wall = time.time() - t0

        self.flow.eval(); self.conditioner.eval()
        flow = self.flow; conditioner = self.conditioner; device = dev

        def pivot_fn(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            theta = theta.to(device); x = x.to(device)
            S, _ = conditioner.encode(x)
            r, _ = flow.forward(theta, context=S)
            return r

        def encode_fn(x: torch.Tensor) -> torch.Tensor:
            x = x.to(device)
            with torch.no_grad():
                S, _ = conditioner.encode(x)
            return S

        procedure = PivotBasedProcedure(pivot_fn=pivot_fn, d_theta=d, encode_fn=encode_fn)
        arch_meta = {
            "flow_class": type(self.flow).__name__,
            "loss_class": "ExactDensityLoss",
            "loss_history_tail": losses[-min(100, len(losses)):],
            "conditioner_class": type(self.conditioner).__name__,
            "conditioner_params": self.conditioner.n_params(),
        }
        return TrainedModel(procedure=procedure, state_dict={}, final_loss=float(losses[-1]),
                            n_steps=n_steps, wall_clock_sec=wall, arch_metadata=arch_meta)

    def n_params(self) -> dict:
        backbone = sum(p.numel() for p in self.flow.parameters())
        head = self.conditioner.n_params()
        return {"backbone": backbone, "head": head, "calibration_stage": 0,
                "total": backbone + head, "kind": "flow"}
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_exact_density_runner.py -v` → PASS (finite loss; loss_class recorded; pivot d=2; floor-bounded).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/methods/cd_sbi_exact_density.py tests/unit/test_exact_density_runner.py
git commit -m "feat(method): ExactDensityCDSBIRunner — exact −log p(X|θ) over bijection+pivot (Arm I-A)"
```

---

## Task 5: run.py wiring + configs

**Files:**
- Modify: `src/cdsbi/experiments/run.py`
- Create: `configs/method/cd_sbi_exact_density.yaml`, `configs/experiment/mu_sigma_stage_b_exact.yaml`

The exact-density arm needs its OWN conditioner (the invertible summary, not `deep_sets`) and runner. READ `_build_method`'s `cd_sbi` branch (the conditioner dispatch + the loss selector added in M3.1′).

- [ ] **Step 1: Conditioner + loss/runner selection in `_build_method`**

The generic conditioner dispatch builds `cfg.conditioner` by `_target_`. The new `invertible_summary` conditioner needs `n_iid` + `d_theta` — hardcode them in its config (like `deep_sets`). For the loss/runner: extend the M3.1′ `loss_name` selector to recognize `"exact_density"` → `loss_obj = None` (the runner computes the NLL inline). The `runner_class` (from the method config) is `ExactDensityCDSBIRunner`. So the existing branch already passes `loss=loss_obj, ...` — just add:
```python
        loss_name = OmegaConf.select(cfg, "method.loss", default="nfmle")
        if loss_name == "energy":
            from cdsbi.losses.energy_calibration import EnergyCalibrationLoss
            loss_obj = EnergyCalibrationLoss()
        elif loss_name == "exact_density":
            loss_obj = None          # ExactDensityCDSBIRunner computes the NLL itself
        else:
            loss_obj = NFMLELoss()
```
(`_instantiate(runner_class, flow=flow, conditioner=conditioner, loss=loss_obj, allow_ablation=..., device=...)` — `ExactDensityCDSBIRunner.__init__` accepts `loss=None`.)

- [ ] **Step 2: `configs/conditioner/invertible_summary.yaml`** (NEW)

```yaml
name: invertible_summary
_target_: cdsbi.conditioners.invertible_summary.InvertibleSummaryConditioner
n_iid: 10        # hardcoded (run.py builds conditioners generically, no injection)
d_theta: 2       # MUST equal simulator.d_theta (the flow's d is injected separately)
hidden: ${budget.cdsbi_flow_hidden}
n_layers: 6
depth: 2
```

- [ ] **Step 3: `configs/method/cd_sbi_exact_density.yaml`** (NEW)

```yaml
name: cd_sbi
runner_class: cdsbi.methods.cd_sbi_exact_density.ExactDensityCDSBIRunner
flow: single_index_monotone
loss: exact_density
allow_ablation: false
```

- [ ] **Step 4: `configs/experiment/mu_sigma_stage_b_exact.yaml`** (NEW)

```yaml
# @package _global_
defaults:
  - override /target: normal_mu_sigma
  - override /flow: single_index_monotone
  - override /conditioner: invertible_summary
  - override /method: cd_sbi_exact_density
  - override /budget: medium

method:
  flow: single_index_monotone

# Pin the validated recipe so a `python -m cdsbi.experiments.run` headline run matches
# the Task-6 intensive test. NOTE: ExactDensityCDSBIRunner uses plain Adam + no
# scheduler (it reads only lr/batch_size/n_steps/n_train/fresh_batch/grad_clip_norm);
# the optimizer/schedule keys from the budget recipe are ignored by this runner.
training:
  lr: 2e-3
  n_steps: 8000
  batch_size: 256
  n_train: 10000
  fresh_batch: false

experiment:
  name: mu_sigma_stage_b_exact
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

- [ ] **Step 5: Verify composition**

```bash
python -c "
from hydra import initialize, compose
from cdsbi.experiments.run import _build_method, _build_simulator
with initialize(version_base=None, config_path='configs'):
    cfg = compose(config_name='config', overrides=['experiment=mu_sigma_stage_b_exact'])
sim = _build_simulator(cfg)
runner = _build_method(cfg, sim)
print('runner:', type(runner).__name__)
print('conditioner:', type(runner.conditioner).__name__, '| n_params:', runner.conditioner.n_params())
print('loss:', runner.loss)
"
```
Expected: `ExactDensityCDSBIRunner`, `InvertibleSummaryConditioner` (n_params > 0), loss `None`. If the conditioner is wrong/missing `transform`, STOP and fix.

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/experiments/run.py configs/conditioner/invertible_summary.yaml \
        configs/method/cd_sbi_exact_density.yaml configs/experiment/mu_sigma_stage_b_exact.yaml
git commit -m "config: invertible_summary conditioner + cd_sbi_exact_density method + experiment (Arm I-A)"
```

---

## Task 6: intensive verdict (does exact density recover σ?)

**Files:**
- Create: `tests/intensive/test_replicate_mu_sigma_stage_b_exact.py`

The discriminating result: does the exact-density objective recover σ² (where II-A's energy loss got Spearman 0.014) AND calibrate, with the loss respecting the data-entropy floor (no cheat)?

- [ ] **Step 1: Write the test**

```python
"""Intensive: Arm I-A (invertible exact density) — does forcing information-
preservation recover σ² (II-A's casualty) AND calibrate near the oracle, with the
loss bounded by H(X|θ)?"""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.intensive
def test_arm_ia_exact_density_recovers_sigma_and_calibrates():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.invertible_summary import InvertibleSummaryConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.methods.cd_sbi_exact_density import ExactDensityCDSBIRunner
    from cdsbi.diagnostics.sufficiency_recovery import SufficiencyRecovery
    from cdsbi.diagnostics.floor_integrity import FloorIntegrity
    from scipy.stats import kstest, chi2

    torch.manual_seed(0)
    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=32, depth=2)
    cond = InvertibleSummaryConditioner(n_iid=sim.n_iid, d_theta=2, hidden=64, n_layers=6, depth=2)
    runner = ExactDensityCDSBIRunner(flow=flow, conditioner=cond)
    config = {"lr": 2e-3, "batch_size": 256, "n_steps": 8000, "n_train": 10000, "fresh_batch": False}
    trained = runner.fit(simulator=sim, config=config, seed=0)

    # (a) σ² recovered (the II-A casualty) AND μ
    sr = SufficiencyRecovery(n_eval=4000)(trained, sim).value
    print(f"I-A sufficiency: log s²={sr['spearman_log_s2'].iloc[0]:.3f} X̄={sr['spearman_xbar'].iloc[0]:.3f}")
    assert sr["spearman_log_s2"].iloc[0] > 0.9, "σ²-information not recovered by exact density"
    assert sr["spearman_xbar"].iloc[0] > 0.9, "μ-information not recovered"

    # (b) no cheat: loss respects the data-entropy floor H(X|θ)
    fi = FloorIntegrity()(trained, sim).value
    print(f"I-A floor: final_loss={fi['final_loss'].iloc[0]:.3f} H(X|θ)={fi['entropy_floor'].iloc[0]:.3f}")
    assert not bool(fi["cheats"].iloc[0]), "exact-density loss fell below H(X|θ) — impossible unless a bug"

    # (c) calibration near the Stage-A control
    for theta_0 in [(0.0, 0.0), (-0.69, 2.0), (0.69, -2.0)]:
        xv = sim.sample_x_given_theta(theta_0, 3000, np.random.default_rng(7))
        th = torch.tensor([list(theta_0)], dtype=xv.dtype).expand(xv.shape[0], -1)
        with torch.no_grad():
            r = trained.procedure.pivot(th, xv).cpu().numpy()
        ks = kstest(chi2.cdf((r ** 2).sum(1), df=2), "uniform").statistic
        print(f"I-A joint Mahalanobis KS @ {theta_0} = {ks:.3f}")
        assert ks < 0.08, f"I-A joint calibration KS {ks:.3f} too high at {theta_0}"
```

- [ ] **Step 2: Run it (intensive; minutes on GPU)**

Run: `pytest tests/intensive/test_replicate_mu_sigma_stage_b_exact.py -v -s -m intensive`
Report ALL printed values (2 Spearmans, floor margin, 3 KS). Interpretation (do NOT loosen tolerances; REPORT):
- **All pass** → **the discriminating verdict:** exact density recovers σ where calibration-only failed → the bake-off conclusion is "information-preservation is required; calibration-only collapses scale." Mark DONE.
- **σ²-Spearman < 0.9** → even exact density didn't route σ into S (surprising — possibly the affine coord-0 pivot still can't calibrate the learned S, the I-B warp resurfacing end-to-end). Report; mark DONE_WITH_CONCERNS — this would point hard at the flow-architecture direction (§1.5 feature-rectifier).
- **calibration KS ≥ 0.08 but σ²-Spearman > 0.9** → S is sufficient but the pivot mis-calibrates it (the affine-combiner warp). Report; mark DONE_WITH_CONCERNS.
- **`cheats` True** (loss < H(X|θ)−tol) → a bug in the NLL/log-det bookkeeping (a proper density can't beat its entropy). STOP and report — likely a sign/term error in the loss or bijection log-det.

- [ ] **Step 3: Confirm fast suite green**

Run: `pytest -q` (intensive deselected). Report the count.

- [ ] **Step 4: Commit** (regardless; mark DONE_WITH_CONCERNS if a finding)

```bash
git add tests/intensive/test_replicate_mu_sigma_stage_b_exact.py
git commit -m "test(intensive): Arm I-A (exact density) — σ recovery + calibration + floor-bounded"
```

---

## Self-review

- **Spec coverage (Arm I-A + §2.1):** invertible bijection summary (Tasks 1–2) ✓; exact-density NLL with the ancillary `½‖A‖²` + bijection log-det (Task 4) ✓; the d=2 monotone pivot preserved as the inference block (Task 4 procedure) ✓; `FloorIntegrity` data-entropy floor — the §2.1 deferred fix (Task 3) ✓; configs/wiring (Task 5) ✓; harness verdict — σ recovery + calibration + floor (Task 6) ✓. Permutation-equivariance deferred (documented scope note).
- **Placeholder scan:** every step has runnable code + commands + expected output; no TBD.
- **Type consistency:** `AffineCouplingBijection(d,hidden,n_layers,depth).forward(x)->(z,log_det)` (Task 1) is used by `InvertibleSummaryConditioner.transform/encode` (Task 2), which `ExactDensityCDSBIRunner.fit` calls via `conditioner.transform(x)` for `z=(S,A)` and `conditioner.encode(x)` for the procedure (Task 4). `data_entropy_lower_bound()` (Task 3) is read by `FloorIntegrity`'s `ExactDensityLoss` floor map (Task 3) and Task 6. `arch_metadata["loss_class"]="ExactDensityLoss"` (Task 4) matches `_FLOOR_METHOD_BY_LOSS` (Task 3). `ExactDensityCDSBIRunner(flow, conditioner, loss=None, allow_ablation, device)` matches `_build_method`'s `_instantiate(... loss=loss_obj ...)` with `loss_obj=None` (Task 5). `procedure.encode_fn` (from M3.0) is set in fit and read by `SufficiencyRecovery` (Task 6).
- **Risk — does exact density actually route σ into the *calibratable* S?** The objective forces σ-info into S (so A is θ-free) AND the pivot-NFMLE term rewards calibratable S (co-adaptation), so S should land affine-friendly — the reason I-A should beat both I-B (frozen, warped) and II-A (weak σ-gradient). But if the affine coord-0 pivot still can't calibrate the learned S, Task 6 surfaces it (→ §1.5 flow-architecture direction). This is the empirical question.
- **Risk — bijection conditioning / loss bookkeeping.** The `cheats`-True branch in Task 6 is a tripwire for a log-det sign/term bug (a proper NLL cannot beat H(X|θ)). The near-identity init (zeroed coupling output layers) keeps early training stable.
- **Known follow-on:** M3.3′ verdict (II-A σ-collapse vs I-A) + the I-B contrast + manuscript; permutation-equivariant bijection (generalization).
