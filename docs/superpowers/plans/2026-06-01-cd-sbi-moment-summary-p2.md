# CD-SBI Moment-Summary — Phase 2 (symmetry-multimodal) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show CD-SBI handles a multimodal posterior — on the sign-unidentifiable
model `X|θ ~ N(θ², 1)` — producing **valid, efficient, correctly-disconnected**
confidence sets, by dropping R1 (a non-monotone pivot) and regressing the right
`d_θ`-dim moment (`θ²`).

**Architecture:** Reuses P1's `TwoStageCDSBIRunner` + `MomentRegressionConditioner`
(target `theta_sq`). Adds the two **load-bearing** P2 components the spec flags: a
**`NonMonotonePivotFlow`** (monotone in the feature → C1; *unconstrained* in θ → R1
off, so it can be non-injective in θ) and a **disconnected-set extractor** +
`Disconnectedness` diagnostic. A regression-R² **modality detector** selects target
`θ` vs `θ²` and R1 on/off.

**Tech stack:** Python, PyTorch, Hydra, pytest. Builds on P1 (must be merged first).

**Spec:** `docs/superpowers/specs/2026-05-31-cd-sbi-moment-summary-design.md` (Phase 2).
**Theory:** note §15 (R1 ⟹ connected; refuted impossibility), §16 (moment target).

---

## File structure (Phase 2)

- Create `src/cdsbi/simulators/sign_normal_1d.py` — `SignNormal1D` (bimodal test model).
- Create `src/cdsbi/flows/non_monotone_pivot.py` — `NonMonotonePivotFlow` (R1-off).
- Create `src/cdsbi/confidence_set/disconnected.py` — connected-component set extractor.
- Create `src/cdsbi/diagnostics/disconnectedness.py` — `Disconnectedness` (pivot-direct).
- Create `configs/target/sign_normal_1d.yaml`, `configs/flow/non_monotone_pivot.yaml`,
  `configs/method/cd_sbi_moment_r1off.yaml`, `configs/conditioner/moment_regression_sq.yaml`,
  `configs/experiment/moment_sign.yaml`.
- Modify `src/cdsbi/experiments/run.py::_build_flow` — recognize `non_monotone_pivot`.
- Create `src/cdsbi/methods/modality_detector.py` — `select_target_and_r1` (R²-based).
- Tests: `tests/unit/test_sign_normal_1d.py`, `tests/unit/test_non_monotone_pivot.py`,
  `tests/unit/test_disconnected_extractor.py`, `tests/unit/test_modality_detector.py`,
  `tests/integration/test_sign_model_disconnected.py`,
  `tests/intensive/test_replicate_moment_sign.py`.

---

### Task 1: SignNormal1D simulator (bimodal test model)

**Files:**
- Create: `src/cdsbi/simulators/sign_normal_1d.py`
- Test: `tests/unit/test_sign_normal_1d.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sign_normal_1d.py
import math
import numpy as np
import torch
from cdsbi.simulators.sign_normal_1d import SignNormal1D


def test_shapes_and_attrs():
    sim = SignNormal1D(n_iid=10)
    assert sim.d_theta == 1 and sim.d_x == 10
    rng = np.random.default_rng(0)
    theta, x = sim.sample(16, rng)
    assert theta.shape == (16, 1) and x.shape == (16, 10)


def test_sign_symmetry_of_data():
    """X depends on θ only through θ² — so +θ and −θ give the same data law."""
    sim = SignNormal1D(n_iid=10)
    rng = np.random.default_rng(0)
    xp = sim.sample_x_given_theta(np.array([1.5]), 50000, rng).mean().item()
    xn = sim.sample_x_given_theta(np.array([-1.5]), 50000, rng).mean().item()
    assert abs(xp - xn) < 0.05 and abs(xp - 1.5 ** 2) < 0.05


def test_log_prob_matches_gaussian():
    sim = SignNormal1D(n_iid=3)
    x = torch.tensor([[1.0, 2.0, 0.5]])
    theta = torch.tensor([[1.0]])             # θ²=1 → mean 1
    expected = sum(-0.5 * (xi - 1.0) ** 2 - 0.5 * math.log(2 * math.pi)
                   for xi in [1.0, 2.0, 0.5])
    assert abs(sim.log_prob(x, theta).item() - expected) < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sign_normal_1d.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cdsbi/simulators/sign_normal_1d.py
"""SignNormal1D: X = (X_1,…,X_{n_iid}) iid N(θ², 1), θ ∈ ℝ. The map θ↦data depends
only on θ², so the posterior is bimodal at ±√·: the canonical multimodal-with-a-
1-dim-sufficient-reduction target (X̄ is sufficient). Used to show CD-SBI yields a
disconnected, exactly-covering C_α with a non-monotone pivot (theory §15)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class SignNormal1D:
    n_iid: int = 10
    theta_abs_max: float = 3.0          # prior: θ ~ U(−3, 3) (symmetric ⟹ bimodal)
    d_theta: int = 1

    @property
    def d_x(self) -> int:
        return self.n_iid

    @property
    def theta_range(self) -> Tuple[float, float]:
        return (-self.theta_abs_max, self.theta_abs_max)

    @property
    def theta_lower(self) -> Tuple[float]:
        return (-self.theta_abs_max,)

    @property
    def feat_signs(self) -> Tuple[float]:
        """The summary (predicts θ²≈X̄) enters the pivot increasingly: r ∝ (T−θ²)."""
        return (1.0,)

    def _draw_theta(self, n, rng):
        return rng.uniform(-self.theta_abs_max, self.theta_abs_max, size=(n, 1))

    def sample(self, n, rng):
        theta = self._draw_theta(n, rng)
        x = rng.normal(loc=theta ** 2, scale=1.0, size=(n, self.n_iid))
        return torch.from_numpy(theta).float(), torch.from_numpy(x).float()

    def sample_x_given_theta(self, theta_0, n, rng):
        tv = np.atleast_1d(np.asarray(theta_0, dtype=np.float64))
        assert tv.shape == (1,), f"theta_0 shape {tv.shape}, expected (1,)"
        x = rng.normal(loc=float(tv[0]) ** 2, scale=1.0, size=(n, self.n_iid))
        return torch.from_numpy(x).float()

    def log_prob(self, x, theta):
        mean = theta[:, 0:1] ** 2                       # (n,1), θ²
        per_obs = -0.5 * (x - mean) ** 2 - 0.5 * math.log(2 * math.pi)
        return per_obs.sum(dim=-1)

    def r_star(self, theta, x):
        """Analytic calibrated pivot r*=√n(X̄−θ²) (non-monotone in θ). Also lets
        the harness PivotRMSE diagnostic run (it checks `simulator.r_star is None`)."""
        xbar = x.mean(dim=-1, keepdim=True)             # (n,1)
        return math.sqrt(self.n_iid) * (xbar - theta[:, 0:1] ** 2)
```

Add to the Task 1 test (Step 1): `assert torch.allclose(SignNormal1D(n_iid=4).r_star(torch.tensor([[1.0]]), torch.zeros(1, 4)), torch.tensor([[-2.0]]))` (X̄=0, θ²=1 ⟹ √4·(0−1)=−2).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_sign_normal_1d.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/simulators/sign_normal_1d.py tests/unit/test_sign_normal_1d.py
git commit -m "feat(sim): SignNormal1D bimodal test model X|θ~N(θ²,1)"
```

---

### Task 2: NonMonotonePivotFlow (R1-off, monotone-in-feature)

**Files:**
- Create: `src/cdsbi/flows/non_monotone_pivot.py`
- Test: `tests/unit/test_non_monotone_pivot.py`

The pivot keeps a monotone feature channel (`∂r/∂feat` constant-sign ⟹ C1, honest
log-det for NF-MLE) but makes the θ-channel an **unconstrained MLP** ⟹ non-monotone
in θ. Advertises `{R2}` only (no R1), so the runner needs `allow_ablation=True`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_non_monotone_pivot.py
import torch
from cdsbi.flows.base import Guarantee
from cdsbi.flows.non_monotone_pivot import NonMonotonePivotFlow


def test_guarantees_r2_only():
    flow = NonMonotonePivotFlow(d=1, feat_signs=(1.0,))
    assert flow.monotonicity_guarantees == frozenset({Guarantee.R2})


def test_forward_shapes_and_feature_monotone():
    flow = NonMonotonePivotFlow(d=1, feat_signs=(1.0,))
    theta = torch.randn(32, 1)
    feats = torch.randn(32, 1, requires_grad=True)
    r, log_det = flow.forward(theta, context=feats)
    assert r.shape == (32, 1) and log_det.shape == (32,)
    # ∂r/∂feat must be single-signed (C1): check positivity (feat_signs=+1)
    g = torch.autograd.grad(r.sum(), feats)[0]
    assert (g > 0).all(), "feature channel must be monotone increasing (C1)"


def test_can_be_non_monotone_in_theta():
    """Unlike SingleIndexMonotoneFlow, r(θ) can be non-monotone (it must, to give
    disconnected sets). Fit nothing — just check the architecture permits ∂r/∂θ<0
    somewhere after a random init with a curved θ-net."""
    torch.manual_seed(0)
    flow = NonMonotonePivotFlow(d=1, feat_signs=(1.0,), theta_hidden=32)
    feats = torch.zeros(200, 1)
    theta = torch.linspace(-3, 3, 200).unsqueeze(1).requires_grad_(True)
    r, _ = flow.forward(theta, context=feats)
    g = torch.autograd.grad(r.sum(), theta)[0]
    assert (g < 0).any() and (g > 0).any(), "θ-channel should be unconstrained in sign"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_non_monotone_pivot.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cdsbi/flows/non_monotone_pivot.py
"""NonMonotonePivotFlow — R1-OFF pivot. Per coord k:
    z_k = a_k(θ_{≤k}, ctx) + s_fk·softplus(p_fk(ctx))·feat_k      # a_k UNCONSTRAINED
    r_k = G_k(z_k; ctx)                                           # G_k monotone ↑
so ∂r_k/∂feat_k = s_fk·softplus·G' has a fixed sign (C1 ✓, honest log-det) while
∂r_k/∂θ is free (R1 OFF ✓ — permits r(θ)=r(−θ), hence disconnected C_α). Mirrors
SingleIndexMonotoneFlow but swaps the monotone θ-channel for an unconstrained MLP.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cdsbi.flows.base import Flow, Guarantee

_NODES_NP, _WEIGHTS_NP = np.polynomial.legendre.leggauss(12)


def _tanh_mlp(in_dim, hidden, out_dim, depth):
    layers: list = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class _CtxScalar(nn.Module):
    def __init__(self, context_dim, hidden, depth=2):
        super().__init__()
        self.context_dim = context_dim
        if context_dim == 0:
            self.param = nn.Parameter(torch.zeros(1)); self.net = None
        else:
            self.param = None; self.net = _tanh_mlp(context_dim, hidden, 1, depth)

    def forward(self, ctx, n):
        if self.context_dim == 0 or ctx is None:
            return self.param.view(1, 1).expand(n, 1)
        return self.net(ctx)


class _MonotoneG(nn.Module):
    """G(z;ctx)=∫_0^z softplus(MLP([t,ctx]))dt — monotone ↑ in z, G(0)=0."""
    def __init__(self, context_dim, hidden=32, depth=2):
        super().__init__()
        self.context_dim = context_dim
        self.mlp = _tanh_mlp(1 + context_dim, hidden, 1, depth)
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("_nodes", torch.tensor(_NODES_NP, dtype=torch.float32))
        self.register_buffer("_weights", torch.tensor(_WEIGHTS_NP, dtype=torch.float32))

    def _integrand(self, t, ctx):
        inp = t if (self.context_dim == 0 or ctx is None) else torch.cat([t, ctx], -1)
        return F.softplus(self.mlp(inp)) + 1e-3

    def derivative(self, z, ctx):
        return self._integrand(z, ctx)

    def forward(self, z, ctx):
        n = z.shape[0]; K = self._nodes.shape[0]
        u = self._nodes.view(1, -1, 1).expand(n, -1, 1)
        ze = z.view(n, 1, 1).expand(-1, K, -1)
        t = 0.5 * ze * (u + 1.0)
        ctx_rep = None if (self.context_dim == 0 or ctx is None) else \
            ctx.unsqueeze(1).expand(-1, K, -1).reshape(n * K, -1)
        integ = self._integrand(t.reshape(n * K, 1), ctx_rep).view(n, K, 1)
        return 0.5 * z * (self._weights.view(1, -1, 1) * integ).sum(dim=1)


class NonMonotonePivotFlow(nn.Module, Flow):
    monotonicity_guarantees = frozenset({Guarantee.R2})    # feature-monotone only

    def __init__(self, d: int, feat_signs: Sequence[float], hidden: int = 32,
                 depth: int = 2, theta_hidden: int = 32):
        super().__init__()
        if len(feat_signs) != d:
            raise ValueError(f"feat_signs must have length d={d}")
        self.d = d
        self.register_buffer("_s_feat", torch.tensor([float(s) for s in feat_signs]))
        self._a = nn.ModuleList()        # unconstrained θ-channel a_k(θ_{≤k}, ctx)
        self._p_feat = nn.ModuleList()
        self._G = nn.ModuleList()
        for k in range(d):
            cdim = 2 * k                                   # ctx = (θ_{<k}, feat_{<k})
            self._a.append(_tanh_mlp(cdim + 1, theta_hidden, 1, depth))  # +1 = θ_k
            self._p_feat.append(_CtxScalar(cdim, hidden))
            self._G.append(_MonotoneG(cdim, hidden, depth))

    def forward(self, theta: torch.Tensor, context: Optional[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert context is not None and context.shape[-1] == self.d
        feats = context; n = theta.shape[0]
        r_cols: List[torch.Tensor] = []; logdet: List[torch.Tensor] = []
        for k in range(self.d):
            feat_k = feats[:, k:k + 1]
            ctx_k = None if k == 0 else torch.cat([theta[:, :k], feats[:, :k]], -1)
            a_in = theta[:, :k + 1] if ctx_k is None else torch.cat([theta[:, :k + 1], feats[:, :k]], -1)
            a = self._a[k](a_in)                                       # unconstrained in θ
            w_f = self._s_feat[k] * F.softplus(self._p_feat[k](ctx_k, n))
            z = a + w_f * feat_k
            r_k = self._G[k](z, ctx_k)
            g_prime = self._G[k].derivative(z, ctx_k)
            dr_dfeat = w_f * g_prime                                   # ∂r_k/∂feat_k (signed)
            r_cols.append(r_k)
            logdet.append(torch.log(dr_dfeat.abs().clamp_min(1e-12)).squeeze(-1))
        return torch.cat(r_cols, -1), torch.stack(logdet, -1).sum(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_non_monotone_pivot.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/flows/non_monotone_pivot.py tests/unit/test_non_monotone_pivot.py
git commit -m "feat(flow): NonMonotonePivotFlow (R1-off, monotone-in-feature pivot)"
```

---

### Task 3: Disconnected-set extractor + Disconnectedness diagnostic

**Files:**
- Create: `src/cdsbi/confidence_set/disconnected.py`
- Create: `src/cdsbi/diagnostics/disconnectedness.py`
- Test: `tests/unit/test_disconnected_extractor.py`

Both existing set paths (`_confidence_set_1d`, `_ray_sample_set_boundary`) return a
single interval/blob. For d_θ=1 we enumerate `{θ:‖r(θ;X)‖²≤χ²}` connected components
on a θ-grid. `Disconnectedness` counts components pivot-directly (no set object).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_disconnected_extractor.py
import math
import numpy as np
import torch
from scipy.stats import chi2
from cdsbi.confidence_set.disconnected import connected_components_1d
from cdsbi.diagnostics.disconnectedness import n_components


def _sign_pivot(theta, x):
    # analytic sign-model pivot r = √n (X̄ − θ²); x is (1, n_iid) observed
    n = x.shape[-1]
    xbar = x.mean()
    return (math.sqrt(n) * (xbar - theta ** 2))


def test_two_intervals_recovered():
    x = torch.full((1, 10), 4.0)             # X̄=4 ⟹ θ²∈4±band ⟹ two intervals ±√·
    thr = chi2.ppf(0.9, df=1)
    comps = connected_components_1d(lambda th: _sign_pivot(th, x) ** 2,
                                    lo=-5.0, hi=5.0, thresh=thr, n_grid=4001)
    assert len(comps) == 2, f"expected 2 disjoint intervals, got {len(comps)}"
    (a1, b1), (a2, b2) = comps
    assert a1 < b1 < 0 < a2 < b2 and abs(b1 + a2) < 0.1   # symmetric about 0


def test_single_interval_when_band_covers_zero():
    x = torch.full((1, 10), 0.0)             # X̄≈0 ⟹ θ² small ⟹ one interval around 0
    thr = chi2.ppf(0.9, df=1)
    comps = connected_components_1d(lambda th: _sign_pivot(th, x) ** 2,
                                    lo=-5.0, hi=5.0, thresh=thr, n_grid=4001)
    assert len(comps) == 1


def test_n_components_diagnostic():
    x = torch.full((1, 10), 4.0)
    assert n_components(lambda th: _sign_pivot(th, x) ** 2,
                        lo=-5.0, hi=5.0, thresh=chi2.ppf(0.9, 1), n_grid=4001) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_disconnected_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cdsbi/confidence_set/disconnected.py
"""Disconnected confidence-set extraction for d_θ=1 (R1-off pivots). The built-in
PivotBasedProcedure paths assume a single interval / radial convexity; here we
enumerate connected components of {θ : sq(θ) ≤ thresh} on a grid."""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import torch


def connected_components_1d(sq_fn: Callable[[torch.Tensor], torch.Tensor],
                            lo: float, hi: float, thresh: float,
                            n_grid: int = 4001) -> List[Tuple[float, float]]:
    """sq_fn(θ) → ‖r(θ;X)‖² for a (G,1) grid θ. Returns merged [a,b] intervals
    where sq ≤ thresh (each a maximal run of in-set grid points)."""
    grid = torch.linspace(lo, hi, n_grid).unsqueeze(1)
    sq = sq_fn(grid).reshape(-1).detach().cpu().numpy()
    g = grid.reshape(-1).cpu().numpy()
    inside = sq <= thresh
    comps: List[Tuple[float, float]] = []
    i = 0
    while i < n_grid:
        if inside[i]:
            j = i
            while j + 1 < n_grid and inside[j + 1]:
                j += 1
            comps.append((float(g[i]), float(g[j])))
            i = j + 1
        else:
            i += 1
    return comps
```

```python
# src/cdsbi/diagnostics/disconnectedness.py
"""Disconnectedness: count connected components of {θ:‖r(θ;X)‖²≤χ²} on a θ-grid,
pivot-directly (no set construction). >1 ⟹ the procedure produced a disconnected
confidence set — the correct shape for a multimodal posterior (theory §15)."""
from __future__ import annotations

from typing import Callable

import torch

from cdsbi.confidence_set.disconnected import connected_components_1d


def n_components(sq_fn: Callable[[torch.Tensor], torch.Tensor],
                 lo: float, hi: float, thresh: float, n_grid: int = 4001) -> int:
    return len(connected_components_1d(sq_fn, lo, hi, thresh, n_grid))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_disconnected_extractor.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/confidence_set/disconnected.py src/cdsbi/diagnostics/disconnectedness.py tests/unit/test_disconnected_extractor.py
git commit -m "feat(confidence-set): disconnected-set extractor + Disconnectedness diagnostic (d_θ=1)"
```

---

### Task 4: Wire the R1-off path (flow group + configs)

**Files:**
- Modify: `src/cdsbi/experiments/run.py::_build_flow` (recognize `non_monotone_pivot`)
- Create: `configs/flow/non_monotone_pivot.yaml`, `configs/target/sign_normal_1d.yaml`,
  `configs/conditioner/moment_regression_sq.yaml`, `configs/method/cd_sbi_moment_r1off.yaml`,
  `configs/experiment/moment_sign.yaml`

- [ ] **Step 1: Locate the DEDICATED `single_index_monotone` branch and the fast-path guard**

Run: `grep -n "single_index_monotone\|method_flow_label ==\|method_flow_label is None\|_target_" src/cdsbi/experiments/run.py`
Expected: a **dedicated** `single_index_monotone` branch (~`run.py:58`) that injects
`theta_signs`/`feat_signs` from the simulator, placed **before** the generic
fast-path guard (~`run.py:80`, `method_flow_label is None or == hydra_flow_name`)
that pops `_target_`. The new branch MUST go *with* the dedicated branch (before the
guard) — placing it after the guard makes it unreachable, and the guard's
`flow_dict.pop("_target_")` would KeyError on our `_target_`-less flow config.

- [ ] **Step 2: Add the `non_monotone_pivot` DEDICATED branch (before the fast-path guard)**

```python
# in _build_flow, immediately after the `single_index_monotone` branch (i.e. BEFORE
# the `method_flow_label is None or method_flow_label == hydra_flow_name` fast path):
if method_flow_label == "non_monotone_pivot":
    from cdsbi.flows.non_monotone_pivot import NonMonotonePivotFlow
    return NonMonotonePivotFlow(
        d=int(simulator.d_theta),
        feat_signs=tuple(simulator.feat_signs),          # injected from the simulator
        hidden=int(OmegaConf.select(cfg, "flow.hidden", default=32)),
        depth=int(OmegaConf.select(cfg, "flow.depth", default=2)),
    )
```
(The `non_monotone_pivot.yaml` flow config carries only `name/hidden/depth` — no
`_target_` — because this dedicated branch instantiates the class directly.)

- [ ] **Step 3: Write the configs**

```yaml
# configs/flow/non_monotone_pivot.yaml
name: non_monotone_pivot
hidden: 32
depth: 2
```
```yaml
# configs/target/sign_normal_1d.yaml
name: sign_normal_1d
_target_: cdsbi.simulators.sign_normal_1d.SignNormal1D
n_iid: 10
```
```yaml
# configs/conditioner/moment_regression_sq.yaml
name: moment_regression
_target_: cdsbi.conditioners.moment_regression.MomentRegressionConditioner
n_iid: 10
d_theta: 1
target: theta_sq                    # regress θ² (the modality-resolving moment)
```
```yaml
# configs/method/cd_sbi_moment_r1off.yaml
name: cd_sbi                        # generic branch (P1 wiring)
runner_class: cdsbi.methods.cd_sbi_two_stage.TwoStageCDSBIRunner
flow: non_monotone_pivot
loss: nfmle
allow_ablation: true                # NonMonotonePivotFlow provides {R2} only (no R1)
```
```yaml
# configs/experiment/moment_sign.yaml
# @package _global_
defaults:
  - override /target: sign_normal_1d
  - override /flow: non_monotone_pivot
  - override /conditioner: moment_regression_sq
  - override /method: cd_sbi_moment_r1off
  - override /budget: medium
method:
  flow: non_monotone_pivot
training:
  stage1_steps: 4000
  stage2_steps: 8000
experiment:
  name: moment_sign
  n_eval: 6000
  eval_thetas_interior: [[1.5], [0.8], [2.0], [-1.5]]
  eval_thetas_edge: []
  alpha_grid: [0.5, 0.68, 0.9, 0.95]
  n_eval_per_theta: 2000
```

- [ ] **Step 4: Smoke-run**

Run:
```bash
python -m cdsbi.experiments.run experiment=moment_sign seed=0 \
  training.stage1_steps=200 training.stage2_steps=200 hydra.run.dir=/tmp/sign_smoke
```
Expected: exits 0; `arch_metadata.flow_class == "NonMonotonePivotFlow"`. Verify:
`python -c "import torch; print(torch.load('/tmp/sign_smoke/model.pt',weights_only=False)['arch_metadata']['flow_class'])"`

- [ ] **Step 5: Commit**

```bash
git add configs/flow/non_monotone_pivot.yaml configs/target/sign_normal_1d.yaml \
  configs/conditioner/moment_regression_sq.yaml configs/method/cd_sbi_moment_r1off.yaml \
  configs/experiment/moment_sign.yaml src/cdsbi/experiments/run.py
git commit -m "feat(experiments): wire R1-off non-monotone pivot + sign-model experiment"
```

---

### Task 5: Integration — sign model is valid + disconnected + efficient

**Files:**
- Create: `tests/integration/test_sign_model_disconnected.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_sign_model_disconnected.py
import math
import numpy as np
import torch
from scipy.stats import chi2, kstest
from cdsbi.conditioners.moment_regression import MomentRegressionConditioner
from cdsbi.flows.non_monotone_pivot import NonMonotonePivotFlow
from cdsbi.methods.cd_sbi_two_stage import TwoStageCDSBIRunner
from cdsbi.simulators.sign_normal_1d import SignNormal1D
from cdsbi.diagnostics.disconnectedness import n_components


def test_sign_model_valid_and_disconnected():
    sim = SignNormal1D(n_iid=10)
    cond = MomentRegressionConditioner(n_iid=10, d_theta=1, target="theta_sq")
    flow = NonMonotonePivotFlow(d=1, feat_signs=sim.feat_signs, hidden=32)
    runner = TwoStageCDSBIRunner(flow=flow, conditioner=cond,
                                 allow_ablation=True, device="cpu")
    trained = runner.fit(sim, {"lr": 2e-3, "stage1_steps": 3000, "stage2_steps": 6000,
                               "batch_size": 512, "fresh_batch": True}, seed=0)
    proc = trained.procedure

    # (1) validity: pivot N(0,1) at the truth ⟹ coverage = α
    rng = np.random.default_rng(3)
    theta0 = np.array([1.5])
    xv = sim.sample_x_given_theta(theta0, 20000, rng)
    th = torch.tensor(theta0, dtype=torch.float32).expand(20000, 1)
    r = proc.pivot_fn(th, xv)
    assert kstest(r[:, 0].detach().numpy(), "norm").pvalue > 1e-3
    for a in (0.8, 0.9):
        emp = float(((r ** 2).sum(-1) <= chi2.ppf(a, 1)).float().mean())
        assert abs(emp - a) < 0.06

    # (2) disconnected: for a single observed X at θ0=1.5, C_α should be two intervals
    x_obs = sim.sample_x_given_theta(theta0, 1, np.random.default_rng(7))
    thr = chi2.ppf(0.9, 1)
    def sq(theta_grid):
        return (proc.pivot_fn(theta_grid, x_obs) ** 2).sum(-1)
    assert n_components(sq, lo=-5.0, hi=5.0, thresh=thr, n_grid=4001) == 2
```

- [ ] **Step 2: Run test to verify it fails (pre-implementation of upstream tasks)**

Run: `pytest tests/integration/test_sign_model_disconnected.py -v`
Expected: PASS once Tasks 1–3 are implemented (it exercises them end-to-end). If
validity fails, bump `stage2_steps` to 9000 (recipe). If `n_components != 2`, the
non-monotone θ-net under-fit the parabola — bump `theta_hidden`/`stage2_steps`.

- [ ] **Step 3: (only if failing) tune the recipe** as noted; re-run.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/integration/test_sign_model_disconnected.py -v`
Expected: 1 passed (~30–60s).

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_sign_model_disconnected.py
git commit -m "test(integration): sign model is valid + correctly disconnected"
```

---

### Task 6: Modality detector (regression-R² target/R1 selector)

**Files:**
- Create: `src/cdsbi/methods/modality_detector.py`
- Test: `tests/unit/test_modality_detector.py`

Selects the regression target and R1 on/off from simulations: fit R²(θ) and R²(θ²);
if R²(θ) is high → unimodal-regular (target `theta`, R1 on); if R²(θ) is low but
R²(θ²) high → symmetry-multimodal (target `theta_sq`, R1 off).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_modality_detector.py
import numpy as np
from cdsbi.methods.modality_detector import select_target_and_r1
from cdsbi.simulators.sign_normal_1d import SignNormal1D
from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar


def test_detects_multimodal_sign_model():
    sel = select_target_and_r1(SignNormal1D(n_iid=10), n=20000, seed=0)
    assert sel["target"] == "theta_sq" and sel["r1"] is False


def test_detects_regular_mu_sigma():
    sel = select_target_and_r1(NormalUnknownMeanVar(), n=20000, seed=0)
    assert sel["target"] == "theta" and sel["r1"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_modality_detector.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cdsbi/methods/modality_detector.py
"""Modality detector: choose the regression target m(θ) and R1 on/off from sims.
For each coord, R²(θ) vs R²(θ²) under a degree-3 polynomial fit of the moment on a
cheap summary (the raw per-obs mean). Low R²(θ) + high R²(θ²) ⟹ sign-symmetry ⟹
target θ², R1 off. (Phase-2 scope: a single global decision; per-coord is §8 OOS.)"""
from __future__ import annotations

import numpy as np


def _r2_poly(y: np.ndarray, x: np.ndarray, degree: int = 3) -> float:
    H = np.concatenate([x ** p for p in range(degree + 1)], axis=1)  # [1,x,…,x^deg]
    beta, *_ = np.linalg.lstsq(H, y, rcond=None)
    yhat = H @ beta
    ss_res = float(((y - yhat) ** 2).sum()); ss_tot = float(((y - y.mean(0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def select_target_and_r1(simulator, n: int = 20000, seed: int = 0,
                         low: float = 0.3, high: float = 0.7) -> dict:
    rng = np.random.default_rng(seed)
    theta, x = simulator.sample(n, rng)
    theta = theta.numpy(); xbar = x.numpy().mean(axis=1, keepdims=True)   # cheap summary
    r2_theta = _r2_poly(theta, xbar)
    r2_sq = _r2_poly(theta ** 2, xbar)
    if r2_theta < low <= high < r2_sq:           # mean uninformative, square informative
        return {"target": "theta_sq", "r1": False, "r2_theta": r2_theta, "r2_sq": r2_sq}
    return {"target": "theta", "r1": True, "r2_theta": r2_theta, "r2_sq": r2_sq}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_modality_detector.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/methods/modality_detector.py tests/unit/test_modality_detector.py
git commit -m "feat(method): regression-R² modality detector (target/R1 selector)"
```

---

### Task 7: Intensive replication — sign model valid + disconnected across seeds

**Files:**
- Create: `tests/intensive/test_replicate_moment_sign.py`

The full Hydra harness assumes the standard (μ,σ²)-style diagnostic battery, which
the bespoke sign model does not provide; so P2's replication validates **in-process**
across seeds (validity + disconnectedness), not via `run.py`.

- [ ] **Step 1: Write the replication test**

```python
# tests/intensive/test_replicate_moment_sign.py
import math
import numpy as np
import torch
import pytest
from scipy.stats import chi2
from cdsbi.conditioners.moment_regression import MomentRegressionConditioner
from cdsbi.flows.non_monotone_pivot import NonMonotonePivotFlow
from cdsbi.methods.cd_sbi_two_stage import TwoStageCDSBIRunner
from cdsbi.simulators.sign_normal_1d import SignNormal1D
from cdsbi.diagnostics.disconnectedness import n_components


@pytest.mark.intensive
def test_replicate_moment_sign():
    sim = SignNormal1D(n_iid=10)
    cov_errs, disc = [], []
    for seed in range(3):
        cond = MomentRegressionConditioner(n_iid=10, d_theta=1, target="theta_sq")
        flow = NonMonotonePivotFlow(d=1, feat_signs=sim.feat_signs, hidden=32)
        runner = TwoStageCDSBIRunner(flow=flow, conditioner=cond,
                                     allow_ablation=True, device="cpu")
        proc = runner.fit(sim, {"lr": 2e-3, "stage1_steps": 3000, "stage2_steps": 6000,
                                "batch_size": 512, "fresh_batch": True}, seed=seed).procedure
        rng = np.random.default_rng(100 + seed)
        worst = 0.0
        for t0 in (1.5, 0.8, 2.0):
            xv = sim.sample_x_given_theta(np.array([t0]), 20000, rng)
            th = torch.tensor([[t0]], dtype=torch.float32).expand(20000, 1)
            sq = (proc.pivot_fn(th, xv) ** 2).sum(-1)
            for a in (0.8, 0.9, 0.95):
                worst = max(worst, abs(float((sq <= chi2.ppf(a, 1)).float().mean()) - a))
        cov_errs.append(worst)
        x_obs = sim.sample_x_given_theta(np.array([1.5]), 1, np.random.default_rng(seed))
        disc.append(n_components(lambda g: (proc.pivot_fn(g, x_obs) ** 2).sum(-1),
                                 -5.0, 5.0, chi2.ppf(0.9, 1), 4001))
    assert np.mean(cov_errs) <= 0.06, f"coverage error {np.mean(cov_errs):.3f}"
    assert all(d == 2 for d in disc), f"disconnected set not recovered: {disc}"
```

- [ ] **Step 2: Run it (opt-in)**

Run: `pytest tests/intensive/test_replicate_moment_sign.py -m intensive -v -s`
Expected: PASS — mean coverage error ≤ 0.06, all seeds give 2-component sets.

- [ ] **Step 3: Confirm fast suite green**

Run: `pytest -q -m "not intensive"`
Expected: all pass (P1 + P2 unit/integration).

- [ ] **Step 4: Commit**

```bash
git add tests/intensive/test_replicate_moment_sign.py
git commit -m "test(intensive): sign-model replication (valid + disconnected)"
```

---

## Self-review (checklist)

- **Spec coverage (Phase 2):** NonMonotonePivotFlow R1-off (Task 2), disconnected
  extractor + Disconnectedness (Task 3), sign model (Task 1), target=θ² + allow_ablation
  wiring (Task 4), valid+disconnected+efficient (Task 5), modality detector (Task 6),
  replication (Task 7). ✓
- **Load-bearing components present:** the two the spec flagged (R1-off flow C1,
  disconnected extractor C2) are Tasks 2–3. ✓
- **Depends on P1** (TwoStageCDSBIRunner, MomentRegressionConditioner) — P1 must merge first.
- **Recipe knobs** (`stage*_steps`, `theta_hidden`) flagged as tunes, not code bugs.
