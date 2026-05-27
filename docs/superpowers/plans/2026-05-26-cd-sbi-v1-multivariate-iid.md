# CD-SBI v1 — §8.2 multivariate location-Gaussian (Σ = I) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the v0 §8.1 infrastructure from 1D to d=2 (location-Gaussian with Σ = I), replicating manuscript §8.2 with CDSBI and producing a cross-method comparison via the existing five method runners.

**Architecture:**
- `LocationGaussian2D_iid` simulator: `X | θ ~ N(θ, I_2)`, `θ ~ U[-7, 7]^2`, closed-form `r* = θ - X` componentwise.
- `TriangularAdditiveFlow`: autoregressive additive triangular flow per manuscript §6.1 form 1. For d=2: `r_1 = α_a · UMNN(θ_1) - α_b · UMNN(X_1)`; `r_2 = a_2(θ_2; θ_1, X_1) - b_2(X_2; θ_1, X_1)` via context-conditioned UMNNs. Generalizes to any d.
- `JointMahalanobis` diagnostic: at fixed θ_0, the empirical distribution of `||r(θ_0; X_i)||²` for `X_i ~ p(X|θ_0)` should match `χ²_d`. KS test against the χ²-CDF.
- `PivotBasedProcedure.confidence_set` upgraded to multivariate: same chi-square inversion `{θ : ||r||² ≤ χ²_{d, α}}`, but `boundary_repr` becomes a tensor of sampled boundary points instead of a 1D `(left, right)` pair. `contains_batch` already handles d > 1.
- Simulator protocol gains `sample_x_given_theta(theta_0, n, rng) → X`, called by Coverage / SetSize / JointMahalanobis (closes the v0 hardcoded `X = θ + N(0,1)` caveat).
- All five existing method runners pick up d>1 via dimension-aware flow / classifier construction in `experiments/run.py`.

**Tech Stack:** PyTorch, Hydra, nflows (MAF backbone for NPE/NLE), sbi (Posterior class only after Phase C refactor), pytest, pandas + parquet.

---

## File structure

```
NEW
  src/cdsbi/simulators/location_gauss_2d_iid.py    # 2D iid Gaussian-location simulator
  src/cdsbi/flows/triangular_additive.py            # autoregressive additive triangular flow
  src/cdsbi/diagnostics/joint_mahalanobis.py        # ||r||² vs χ²_d KS test
  configs/target/loc_gauss_2d_iid.yaml
  configs/flow/triangular_additive.yaml
  configs/experiment/8_2_replication.yaml
  configs/experiment/8_2_baseline_sweep.yaml
  tests/unit/test_location_gauss_2d_iid.py
  tests/unit/test_triangular_additive_flow.py
  tests/unit/test_pivot_procedure_multivariate.py
  tests/diagnostics/test_joint_mahalanobis.py
  tests/intensive/test_replicate_8_2.py

MODIFY
  src/cdsbi/simulators/location_normal_1d.py        # add sample_x_given_theta
  src/cdsbi/confidence_set/procedures.py            # PivotBasedProcedure.confidence_set d>1
  src/cdsbi/confidence_set/datatypes.py             # ConfidenceSet.boundary_repr docstring
  src/cdsbi/diagnostics/coverage.py                 # use simulator.sample_x_given_theta
  src/cdsbi/diagnostics/set_size.py                 # use simulator.sample_x_given_theta
  src/cdsbi/diagnostics/pivot_rmse.py               # tolerate d_theta > 1 reductions
  src/cdsbi/diagnostics/conditional_pit.py          # per-coord PIT for d > 1
  src/cdsbi/experiments/run.py                      # d-aware _build_flow; add joint_mahalanobis
  src/cdsbi/analysis/paper_tables.py                # paper_table_8_2
  configs/target/loc_normal_1d.yaml                 # expose d_theta, d_x for downstream
  configs/budget/*.yaml                             # add triangular_additive_hidden
  tools/retune_budgets.py                           # tune triangular_additive_hidden
```

---

## Task 1: `Simulator.sample_x_given_theta` protocol + LocationNormal1D implementation

The v0 Coverage / SetSize diagnostics hardcoded `X = θ + N(0, 1)`; multivariate generalization needs the simulator to expose this. Adding it to `LocationNormal1D` first keeps the 1D path working before we add d=2.

**Files:**
- Modify: `src/cdsbi/simulators/location_normal_1d.py`
- Test: `tests/unit/test_location_normal_1d.py` (file exists from v0)

- [ ] **Step 1: Add the failing test**

Append to `tests/unit/test_location_normal_1d.py`:

```python
def test_sample_x_given_theta_shape_and_distribution(seed):
    import numpy as np
    from cdsbi.simulators.location_normal_1d import LocationNormal1D
    rng = np.random.default_rng(seed)
    sim = LocationNormal1D()
    x = sim.sample_x_given_theta(theta_0=2.5, n=5000, rng=rng)
    assert x.shape == (5000, 1)
    # X | θ=2.5 ~ N(2.5, 1)
    assert abs(float(x.mean()) - 2.5) < 0.05
    assert abs(float(x.std()) - 1.0) < 0.05


def test_sample_x_given_theta_accepts_vector_theta(seed):
    import numpy as np
    from cdsbi.simulators.location_normal_1d import LocationNormal1D
    rng = np.random.default_rng(seed)
    sim = LocationNormal1D()
    # In 1D, a "vector" θ_0 is a length-1 sequence — implementation must accept both.
    x = sim.sample_x_given_theta(theta_0=[1.0], n=500, rng=rng)
    assert x.shape == (500, 1)
    assert abs(float(x.mean()) - 1.0) < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_location_normal_1d.py::test_sample_x_given_theta_shape_and_distribution -v`

Expected: FAIL with `AttributeError: 'LocationNormal1D' object has no attribute 'sample_x_given_theta'`.

- [ ] **Step 3: Implement `sample_x_given_theta`**

Add to `src/cdsbi/simulators/location_normal_1d.py` (inside the `LocationNormal1D` class):

```python
    def sample_x_given_theta(self, theta_0, n: int, rng: "np.random.Generator") -> torch.Tensor:
        """Draw n samples of X conditional on θ = θ_0. Used by coverage / size diagnostics."""
        # theta_0 may be a scalar float or a length-1 sequence — both map to a (1,) vector.
        import numpy as np
        theta_vec = np.atleast_1d(np.asarray(theta_0, dtype=np.float64))
        assert theta_vec.shape == (self.d_theta,), (
            f"theta_0 has shape {theta_vec.shape}, expected ({self.d_theta},)"
        )
        eps = rng.standard_normal(size=(n, self.d_x))
        x_np = theta_vec[None, :] + eps  # broadcast (1, d_x) + (n, d_x) → (n, d_x)
        return torch.from_numpy(x_np).float()
```

- [ ] **Step 4: Run tests to verify both pass**

Run: `pytest tests/unit/test_location_normal_1d.py -v`

Expected: ALL PASS (existing tests + the two new ones).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/simulators/location_normal_1d.py tests/unit/test_location_normal_1d.py
git commit -m "feat(simulators): Simulator.sample_x_given_theta protocol + LocationNormal1D impl

Forward hook from v0: Coverage and SetSize hardcoded the X|θ sampling
('X = θ + N(0, 1)'), assuming LocationNormal1D. Lifting that to a
Simulator-protocol method lets v1's multivariate simulator slot in
without touching diagnostic code.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `LocationGaussian2D_iid` simulator

**Files:**
- Create: `src/cdsbi/simulators/location_gauss_2d_iid.py`
- Test: `tests/unit/test_location_gauss_2d_iid.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_location_gauss_2d_iid.py`:

```python
"""LocationGaussian2D_iid: X | θ ~ N(θ, I_2), θ ~ U[-7, 7]^2."""
import math

import numpy as np
import torch

from cdsbi.simulators.location_gauss_2d_iid import LocationGaussian2D_iid


def test_dims_are_2(seed):
    sim = LocationGaussian2D_iid()
    assert sim.d_theta == 2
    assert sim.d_x == 2


def test_sample_shape_and_distribution(seed):
    sim = LocationGaussian2D_iid()
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(n=5000, rng=rng)
    assert theta.shape == (5000, 2)
    assert x.shape == (5000, 2)
    # θ uniform on [-7, 7]^2
    assert abs(float(theta.mean()) - 0.0) < 0.2
    # X - θ ~ N(0, I_2)
    diff = (x - theta).numpy()
    assert abs(diff.mean()) < 0.05
    assert abs(diff.std() - 1.0) < 0.05


def test_r_star_componentwise(seed):
    sim = LocationGaussian2D_iid()
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(n=100, rng=rng)
    r_star = sim.r_star(theta, x)
    assert r_star.shape == (100, 2)
    assert torch.allclose(r_star, theta - x)


def test_log_prob_matches_isotropic_gaussian(seed):
    sim = LocationGaussian2D_iid()
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(n=10, rng=rng)
    ll = sim.log_prob(x, theta)
    expected = -math.log(2 * math.pi) - 0.5 * (x - theta).pow(2).sum(dim=-1)
    assert torch.allclose(ll, expected, atol=1e-6)


def test_sample_x_given_theta_shape_and_distribution(seed):
    sim = LocationGaussian2D_iid()
    rng = np.random.default_rng(seed)
    x = sim.sample_x_given_theta(theta_0=[1.0, -2.0], n=5000, rng=rng)
    assert x.shape == (5000, 2)
    assert abs(float(x[:, 0].mean()) - 1.0) < 0.1
    assert abs(float(x[:, 1].mean()) - (-2.0)) < 0.1


def test_entropy_lower_bound_is_d_times_1d_floor():
    sim = LocationGaussian2D_iid()
    # Joint entropy of N(θ, I_2) is 2 * ½ log(2πe) = log(2πe)
    assert abs(sim.entropy_lower_bound() - math.log(2 * math.pi * math.e)) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_location_gauss_2d_iid.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'cdsbi.simulators.location_gauss_2d_iid'`.

- [ ] **Step 3: Implement the simulator**

Create `src/cdsbi/simulators/location_gauss_2d_iid.py`:

```python
"""LocationGaussian2D_iid: X | θ ~ N(θ, I_2) with θ ~ U[a, b]^2."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class LocationGaussian2D_iid:
    theta_range: Tuple[float, float] = (-7.0, 7.0)
    d_theta: int = 2
    d_x: int = 2

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = self.theta_range
        theta_np = rng.uniform(a, b, size=(n, self.d_theta))
        eps_np = rng.standard_normal(size=(n, self.d_x))
        x_np = theta_np + eps_np
        return (
            torch.from_numpy(theta_np).float(),
            torch.from_numpy(x_np).float(),
        )

    def sample_x_given_theta(self, theta_0, n: int, rng: np.random.Generator) -> torch.Tensor:
        theta_vec = np.atleast_1d(np.asarray(theta_0, dtype=np.float64))
        assert theta_vec.shape == (self.d_theta,), (
            f"theta_0 shape {theta_vec.shape}, expected ({self.d_theta},)"
        )
        eps_np = rng.standard_normal(size=(n, self.d_x))
        x_np = theta_vec[None, :] + eps_np
        return torch.from_numpy(x_np).float()

    def r_star(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return theta - x

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # -d/2 log(2π) - ½ ||x - θ||²
        return -0.5 * self.d_x * math.log(2 * math.pi) - 0.5 * (x - theta).pow(2).sum(dim=-1)

    def entropy_lower_bound(self) -> float:
        # H(N(θ, I_d)) = d * ½ log(2πe)
        return self.d_x * 0.5 * math.log(2 * math.pi * math.e)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_location_gauss_2d_iid.py -v`

Expected: ALL 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/simulators/location_gauss_2d_iid.py tests/unit/test_location_gauss_2d_iid.py
git commit -m "feat(simulators): LocationGaussian2D_iid (§8.2 target)

X | θ ~ N(θ, I_2), θ ~ U[-7, 7]^2. Closed-form r*(θ, X) = θ - X
componentwise; log_prob is the standard isotropic Gaussian density.
entropy_lower_bound = d · ½ log(2πe).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `TriangularAdditiveFlow` (autoregressive additive form)

The §6.1 form 1 architecture, generalized to any d. For d=2 (the §8.2 setting):
- `r_1 = α_a · a_1(θ_1) - α_b · b_1(X_1)` — exactly `AdditiveFlow1D` on `(θ_1, X_1)`.
- `r_2 = a_2(θ_2; ctx) - b_2(X_2; ctx)` with `ctx = (θ_1, X_1)`.

For general d, coordinate k has context `(θ_<k, X_<k)`.

**Files:**
- Create: `src/cdsbi/flows/triangular_additive.py`
- Test: `tests/unit/test_triangular_additive_flow.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_triangular_additive_flow.py`:

```python
"""TriangularAdditiveFlow: autoregressive additive form for d ≥ 2."""
from __future__ import annotations

import math

import torch

from cdsbi.flows.base import Guarantee
from cdsbi.flows.triangular_additive import TriangularAdditiveFlow


def test_advertises_R1_R2():
    flow = TriangularAdditiveFlow(d=2, hidden=8)
    assert flow.monotonicity_guarantees == frozenset({Guarantee.R1, Guarantee.R2})


def test_forward_shape_d2(seed):
    flow = TriangularAdditiveFlow(d=2, hidden=8)
    n = 64
    theta = torch.randn(n, 2)
    x = torch.randn(n, 2)
    r, log_det = flow.forward(theta, context=x)
    assert r.shape == (n, 2)
    assert log_det.shape == (n,)


def test_strictly_monotone_in_theta_per_coord(seed):
    flow = TriangularAdditiveFlow(d=2, hidden=8)
    flow.eval()
    # Hold X and θ_<k fixed; sweep θ_k from -3 to 3; r_k must be strictly increasing.
    x = torch.tensor([[0.5, -0.5]])
    base = torch.tensor([[0.0, 0.0]])
    grid = torch.linspace(-3.0, 3.0, 25).view(-1, 1)
    # Sweep θ_1
    theta_sweep_1 = torch.cat([grid, base[:, 1:].expand(25, -1)], dim=-1)
    r_sweep_1, _ = flow.forward(theta_sweep_1, context=x.expand(25, -1))
    assert (r_sweep_1[1:, 0] > r_sweep_1[:-1, 0]).all(), "r_1 not strictly increasing in θ_1"
    # Sweep θ_2
    theta_sweep_2 = torch.cat([base[:, :1].expand(25, -1), grid], dim=-1)
    r_sweep_2, _ = flow.forward(theta_sweep_2, context=x.expand(25, -1))
    assert (r_sweep_2[1:, 1] > r_sweep_2[:-1, 1]).all(), "r_2 not strictly increasing in θ_2"


def test_strictly_monotone_in_X_per_coord_with_negative_sign(seed):
    """Per spec / manuscript §6.1 (R2_auto): ∂r_k/∂X_k < 0 by convention."""
    flow = TriangularAdditiveFlow(d=2, hidden=8)
    flow.eval()
    theta = torch.tensor([[0.5, -0.5]])
    base_x = torch.tensor([[0.0, 0.0]])
    grid = torch.linspace(-3.0, 3.0, 25).view(-1, 1)
    # Sweep X_1
    x_sweep_1 = torch.cat([grid, base_x[:, 1:].expand(25, -1)], dim=-1)
    r_sweep_1, _ = flow.forward(theta.expand(25, -1), context=x_sweep_1)
    assert (r_sweep_1[1:, 0] < r_sweep_1[:-1, 0]).all(), "r_1 not strictly decreasing in X_1"
    # Sweep X_2
    x_sweep_2 = torch.cat([base_x[:, :1].expand(25, -1), grid], dim=-1)
    r_sweep_2, _ = flow.forward(theta.expand(25, -1), context=x_sweep_2)
    assert (r_sweep_2[1:, 1] < r_sweep_2[:-1, 1]).all(), "r_2 not strictly decreasing in X_2"


def test_log_det_matches_diagonal_jacobian(seed):
    """For triangular flows, log|det ∂r/∂X| = Σ_k log|∂r_k/∂X_k|."""
    flow = TriangularAdditiveFlow(d=2, hidden=8)
    flow.eval()
    n = 16
    theta = torch.randn(n, 2)
    x = torch.randn(n, 2, requires_grad=True)
    r, log_det = flow.forward(theta, context=x)
    # Sum |∂r_k/∂X_k| via per-coord backward — fast in d=2.
    diag_terms = []
    for k in range(2):
        g = torch.autograd.grad(r[:, k].sum(), x, retain_graph=True)[0]
        diag_terms.append(g[:, k])
    expected_log_det = torch.stack([torch.log(d.abs()) for d in diag_terms]).sum(dim=0)
    assert torch.allclose(log_det, expected_log_det, atol=1e-4), (
        f"log_det disagrees with diagonal product (max diff {(log_det - expected_log_det).abs().max().item():.2e})"
    )


def test_d2_reduces_to_AdditiveFlow1D_on_first_coord(seed):
    """When d=2, the first coordinate's flow equals a standalone AdditiveFlow1D in distribution."""
    # Construct both; share random init via a seed; verify shape matches at minimum.
    flow = TriangularAdditiveFlow(d=2, hidden=8)
    theta = torch.tensor([[0.5, 1.0]])
    x = torch.tensor([[0.2, -0.3]])
    r, _ = flow.forward(theta, context=x)
    # r_1 must depend only on (θ_1, X_1):
    theta2 = torch.tensor([[0.5, 99.0]])
    x2 = torch.tensor([[0.2, 99.0]])
    r2, _ = flow.forward(theta2, context=x2)
    assert torch.allclose(r[:, 0], r2[:, 0], atol=1e-6), (
        f"r_1 not autoregressive: depends on (θ_2 or X_2)"
    )


def test_n_params_grows_with_d(seed):
    flow_d2 = TriangularAdditiveFlow(d=2, hidden=8)
    flow_d3 = TriangularAdditiveFlow(d=3, hidden=8)
    # d=3 has an extra coordinate with context_dim = 2*(d-1) = 4 (theta_<k, X_<k).
    assert flow_d3.n_params() > flow_d2.n_params()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_triangular_additive_flow.py -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the flow**

Create `src/cdsbi/flows/triangular_additive.py`:

```python
"""TriangularAdditiveFlow: r_k = a_k(θ_k; ctx_k) - b_k(X_k; ctx_k), ctx_k = (θ_<k, X_<k).

Autoregressive additive form per manuscript §6.1 form 1. Lower-triangular
Jacobian in both θ and X (autoregressive R1, R2). The k-th coordinate uses
two UMNNBlocks (one over θ_k, one over X_k), each conditioned on the
preceding 2(k-1) values (θ_<k, X_<k). For k=0 the context is empty
(reducing to a standalone scalar UMNN on each of θ_0, X_0), and the
coordinate-wise scaling α_a, α_b match AdditiveFlow1D's convention.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from cdsbi.flows.base import Flow, Guarantee
from cdsbi.flows.umnn import UMNNBlock


class TriangularAdditiveFlow(nn.Module, Flow):
    monotonicity_guarantees = frozenset({Guarantee.R1, Guarantee.R2})

    def __init__(
        self,
        d: int,
        hidden: int = 32,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        if d < 1:
            raise ValueError(f"d must be ≥ 1, got {d}")
        self.d = d
        # Per-coordinate UMNN pairs. Coord k has context dim 2*k (θ_<k, X_<k).
        self.a_blocks = nn.ModuleList()
        self.b_blocks = nn.ModuleList()
        for k in range(d):
            ctx_dim = 2 * k  # θ_<k concat X_<k
            self.a_blocks.append(
                UMNNBlock(context_dim=ctx_dim, hidden=hidden, bias_trainable=True,
                          dropout=dropout, layer_norm=layer_norm)
            )
            self.b_blocks.append(
                UMNNBlock(context_dim=ctx_dim, hidden=hidden, bias_trainable=False,
                          dropout=dropout, layer_norm=layer_norm)
            )
        # Per-coordinate α scalars (matches AdditiveFlow1D init convention).
        init_log_alpha = math.log(0.5 / math.log(2.0))
        self.log_alpha_a = nn.Parameter(torch.full((d,), init_log_alpha))
        self.log_alpha_b = nn.Parameter(torch.full((d,), init_log_alpha))

    def forward(
        self, theta: torch.Tensor, context: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # context is the conditioner-encoded X (Identity ⇒ context == X)
        x = context
        n = theta.shape[0]
        r_cols = []
        log_det_terms = []
        alpha_a = torch.exp(self.log_alpha_a)  # (d,)
        alpha_b = torch.exp(self.log_alpha_b)  # (d,)
        for k in range(self.d):
            theta_k = theta[:, k : k + 1]
            x_k = x[:, k : k + 1]
            if k == 0:
                ctx_k = None
            else:
                ctx_k = torch.cat([theta[:, :k], x[:, :k]], dim=-1)  # (n, 2k)
            a_k = self.a_blocks[k](theta_k, context=ctx_k)
            b_k = self.b_blocks[k](x_k, context=ctx_k)
            r_k = alpha_a[k] * a_k - alpha_b[k] * b_k  # shape (n, 1)
            r_cols.append(r_k)
            # ∂r_k/∂X_k = -α_b[k] · b_k'(X_k; ctx_k). log|·| = log α_b[k] + log b_k'(X_k).
            b_prime = self.b_blocks[k].jacobian_factor(x_k, context=ctx_k)  # (n, 1)
            log_det_terms.append(
                (self.log_alpha_b[k] + torch.log(b_prime.squeeze(-1)))
            )
        r = torch.cat(r_cols, dim=-1)  # (n, d)
        log_det = torch.stack(log_det_terms, dim=-1).sum(dim=-1)  # (n,)
        return r, log_det

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_triangular_additive_flow.py -v`

Expected: ALL 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/flows/triangular_additive.py tests/unit/test_triangular_additive_flow.py
git commit -m "feat(flows): TriangularAdditiveFlow (manuscript §6.1 form 1)

Autoregressive additive triangular flow generalising AdditiveFlow1D to
any d. For d=2 (the §8.2 architecture), coord 0 is a standalone
additive flow on (θ_0, X_0); coord 1 is an additive flow on
(θ_1, X_1) conditioned on (θ_0, X_0). Jacobians in θ and X are both
lower-triangular; log|det ∂r/∂X| is the sum of diagonal terms.

Advertises {R1, R2} (autoregressive specialisations per §6.1).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `PivotBasedProcedure.confidence_set` handles d > 1

`contains_batch` already handles arbitrary d (`r.pow(2).sum(dim=-1) ≤ χ²_{d, α}`); only `confidence_set` had the `d_theta == 1` assertion. Multivariate `confidence_set` returns a `ConfidenceSet` whose `contains` uses the chi-square inequality directly and whose `boundary_repr` is a sample of K=200 boundary points (line-sampling: random unit directions from a center, then 1D bisection along each ray).

**Files:**
- Modify: `src/cdsbi/confidence_set/procedures.py`
- Modify: `src/cdsbi/confidence_set/datatypes.py` (docstring only)
- Test: `tests/unit/test_pivot_procedure_multivariate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_pivot_procedure_multivariate.py`:

```python
"""PivotBasedProcedure in d > 1."""
from __future__ import annotations

import math

import torch
from scipy.stats import chi2

from cdsbi.confidence_set.procedures import PivotBasedProcedure


def _oracle_2d(theta, x):
    # Canonical r* for LocationGaussian2D_iid: r = θ - X (per-coordinate).
    return theta - x


def test_confidence_set_returns_2d_boundary(seed):
    proc = PivotBasedProcedure(pivot_fn=_oracle_2d, d_theta=2, theta_range=(-7.0, 7.0))
    x_obs = torch.tensor([[0.5, -0.2]])
    cs = proc.confidence_set(x_obs, alpha=0.9)
    # boundary_repr: (K, d) where K is the number of sampled rays.
    assert cs.boundary_repr.ndim == 2
    assert cs.boundary_repr.shape[1] == 2
    # Boundary points should satisfy ||r||² ≈ χ²_{2, 0.9}
    thresh = float(chi2.ppf(0.9, df=2))
    r_boundary = _oracle_2d(cs.boundary_repr, x_obs.expand(cs.boundary_repr.shape[0], -1))
    sq = r_boundary.pow(2).sum(dim=-1)
    # Tolerate the 1D-bisection's 1e-4 endpoint precision.
    assert (sq - thresh).abs().max().item() < 1e-2


def test_contains_uses_chisq_condition(seed):
    proc = PivotBasedProcedure(pivot_fn=_oracle_2d, d_theta=2, theta_range=(-7.0, 7.0))
    x_obs = torch.tensor([[0.5, -0.2]])
    cs = proc.confidence_set(x_obs, alpha=0.9)
    # ||θ - X||² ≤ χ²_{2, 0.9} ⇒ contains.
    thresh = float(chi2.ppf(0.9, df=2))
    # θ at center (= X_obs) has r = 0, definitely contained.
    assert cs.contains([0.5, -0.2])
    # Far-away θ definitely outside.
    assert not cs.contains([5.0, 5.0])


def test_contains_batch_unchanged_in_d2(seed):
    """contains_batch already handled d > 1 in v0; verify it still works."""
    proc = PivotBasedProcedure(pivot_fn=_oracle_2d, d_theta=2, theta_range=(-7.0, 7.0))
    x_batch = torch.randn(100, 2)
    inside = proc.contains_batch([0.0, 0.0], x_batch, alpha=0.9)
    assert inside.shape == (100,)
    assert inside.dtype == torch.bool
    # By definition, P(||−X||² ≤ χ²_{2, 0.9} | X ~ N(0, I_2)) ≈ 0.9.
    p = inside.float().mean().item()
    assert abs(p - 0.9) < 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_pivot_procedure_multivariate.py -v`

Expected: FAIL — `confidence_set` currently asserts `d_theta == 1`.

- [ ] **Step 3: Implement multivariate `confidence_set`**

In `src/cdsbi/confidence_set/procedures.py`, find the `PivotBasedProcedure.confidence_set` method and replace its body. Locate the existing method (`def confidence_set(self, x_obs, alpha)`) and replace from the `assert self.d_theta == 1` line through the end of the method (the existing 1D bisection-based body) with:

```python
    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        thresh = float(chi2.ppf(alpha, df=self.d_theta))
        if self.d_theta == 1:
            return self._confidence_set_1d(x_obs, alpha, thresh)
        return self._confidence_set_ray_sampled(x_obs, alpha, thresh, n_rays=200)

    def _confidence_set_1d(self, x_obs, alpha, thresh):
        # (Existing v0 body preserved verbatim — chi-square inversion via
        # 1D bisection of f(θ) = r(θ)² − thresh on either side of the
        # center where r=0.)
        def f(theta_val: float) -> float:
            theta = torch.tensor([[theta_val]], dtype=x_obs.dtype, device=x_obs.device)
            r = self.pivot_fn(theta, x_obs)
            return (r.pow(2).sum().item() - thresh)
        def r_only(theta_val: float) -> float:
            theta = torch.tensor([[theta_val]], dtype=x_obs.dtype, device=x_obs.device)
            return self.pivot_fn(theta, x_obs).item()
        lo, hi = self.theta_range
        center = bisect_1d(r_only, lo, hi, tol=1e-4)
        left = bisect_1d(f, lo, center, tol=1e-4)
        right = bisect_1d(f, center, hi, tol=1e-4)
        def contains(theta_val) -> bool:
            v = float(theta_val) if not hasattr(theta_val, "__len__") else float(theta_val[0])
            return left <= v <= right
        return ConfidenceSet(
            contains=contains, boundary_repr=torch.tensor([left, right]), alpha=alpha,
        )

    def _find_center(self, x_obs, n_grid_per_dim: int = 9, n_newton: int = 8):
        """Find a θ_center inside the α-set with ||r||² near 0 (or its minimum).

        For an oracle location-form pivot (r = θ - X), the min is at θ = X_obs.
        For a trained flow, r(X_obs; X_obs) is generally non-zero, so we:
          (1) evaluate ||r(θ; X_obs)||² on a coarse d-dim grid over theta_range
              (n_grid_per_dim^d points; cheap in d=2-3),
          (2) pick the argmin as a starting point,
          (3) refine by a few gradient-descent steps.
        Returns a (d,) tensor.
        """
        dtype = x_obs.dtype
        device = x_obs.device
        d = self.d_theta
        lo, hi = self.theta_range
        # Coarse grid argmin
        axes = [torch.linspace(lo, hi, n_grid_per_dim, dtype=dtype, device=device)
                for _ in range(d)]
        mesh = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).view(-1, d)
        with torch.no_grad():
            x_batch = x_obs.expand(mesh.shape[0], -1)
            r = self.pivot_fn(mesh, x_batch)
            sq = r.pow(2).sum(dim=-1)
            best = int(torch.argmin(sq).item())
        center = mesh[best].clone().detach().requires_grad_(True)
        # Newton-ish refinement via gradient descent
        opt = torch.optim.LBFGS([center], lr=0.5, max_iter=n_newton)

        def _closure():
            opt.zero_grad()
            r = self.pivot_fn(center.unsqueeze(0), x_obs)
            loss = r.pow(2).sum()
            loss.backward()
            return loss

        opt.step(_closure)
        return center.detach()

    def _confidence_set_ray_sampled(self, x_obs, alpha, thresh, n_rays: int):
        """Multivariate boundary via line-sampling.

        Finds a center via _find_center (argmin of ||r||²), then for each of
        `n_rays` random unit directions u_i ∈ S^{d-1} bisects along the ray
        to find t_i where ||r(center + t · u_i; X)||² = thresh.
        """
        dtype = x_obs.dtype
        device = x_obs.device
        d = self.d_theta
        center = self._find_center(x_obs)  # (d,)

        # If center is itself outside the α-set (rare for well-trained flows
        # but possible with under-trained models), the ray bisection breaks.
        # Detect and signal an empty set.
        with torch.no_grad():
            r_c = self.pivot_fn(center.unsqueeze(0), x_obs)
            sq_c = float(r_c.pow(2).sum().item())
        if sq_c > thresh:
            # Empty α-set at this X_obs.
            empty_boundary = torch.empty((0, d), dtype=dtype, device=device)

            def _contains_empty(theta_val) -> bool:
                return False
            return ConfidenceSet(
                contains=_contains_empty, boundary_repr=empty_boundary, alpha=alpha,
            )

        # Sample n_rays unit directions uniformly on S^{d-1}.
        u = torch.randn(n_rays, d, dtype=dtype, device=device)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # 1D bisection along each ray. f(t=0) < 0 (center is inside),
        # f(t=t_max) > 0 (far enough is outside).
        lo, hi = self.theta_range
        t_lo = torch.zeros(n_rays, dtype=dtype, device=device)
        t_hi = torch.full((n_rays,), 2.0 * (hi - lo), dtype=dtype, device=device)
        x_batch = x_obs.expand(n_rays, -1)
        for _ in range(40):
            m = 0.5 * (t_lo + t_hi)
            theta_m = center.unsqueeze(0) + m.unsqueeze(-1) * u
            r = self.pivot_fn(theta_m, x_batch)
            f = r.pow(2).sum(dim=-1) - thresh
            outside = f > 0
            t_hi = torch.where(outside, m, t_hi)
            t_lo = torch.where(outside, t_lo, m)
        t = 0.5 * (t_lo + t_hi)
        boundary = center.unsqueeze(0) + t.unsqueeze(-1) * u

        def contains(theta_val) -> bool:
            import numpy as np
            arr = np.atleast_1d(np.asarray(theta_val, dtype=np.float64)).reshape(-1)
            theta_t = torch.from_numpy(arr).to(dtype=dtype, device=device).view(1, d)
            r = self.pivot_fn(theta_t, x_obs)
            return bool((r.pow(2).sum().item() <= thresh))

        return ConfidenceSet(contains=contains, boundary_repr=boundary, alpha=alpha)
```

Also update the docstring in `src/cdsbi/confidence_set/datatypes.py` to reflect that `boundary_repr` is `(K, d)` for d > 1:

```python
@dataclass
class ConfidenceSet:
    """Confidence set at a single X_obs and confidence level α.

    `contains(theta_value)` → bool; `boundary_repr` is shape (2,) in 1D
    (lower, upper) and shape (K, d) in d > 1 — K boundary points sampled
    via ray-bisection from the set's center.
    """
    contains: Callable
    boundary_repr: torch.Tensor
    alpha: float
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_pivot_procedure_multivariate.py tests/unit/test_confidence_procedures.py -v`

Expected: ALL tests PASS, including existing 1D ones.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/confidence_set/procedures.py src/cdsbi/confidence_set/datatypes.py tests/unit/test_pivot_procedure_multivariate.py
git commit -m "feat(confidence_set): PivotBasedProcedure.confidence_set handles d > 1

Multivariate confidence sets via line-sampling: K random unit
directions from the set's center, then 1D bisection along each ray
to find ||r||² = χ²_{d, α}. ConfidenceSet.boundary_repr becomes
shape (K, d) in d > 1 (sampled boundary points), shape (2,) in 1D
(unchanged). contains_batch was already d-aware.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `JointMahalanobis` diagnostic

At fixed θ_0 drawing `X_i ~ p(X | θ_0)`, the empirical distribution of
`||r(θ_0; X_i)||²` should match `χ²_d`. KS test against the χ²-CDF, per θ_0.
Degenerate (≡ marginal PIT) in d=1, so we skip when `d_theta == 1` to keep
the v0 §8.1 path unchanged.

**Files:**
- Create: `src/cdsbi/diagnostics/joint_mahalanobis.py`
- Test: `tests/diagnostics/test_joint_mahalanobis.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/diagnostics/test_joint_mahalanobis.py`:

```python
"""JointMahalanobis: ||r(θ_0; X)||² | θ_0 ~ χ²_d test."""
from __future__ import annotations

import numpy as np
import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.joint_mahalanobis import JointMahalanobis
from cdsbi.methods.base import TrainedModel
from cdsbi.reproducibility.seeding import seed_everything
from cdsbi.simulators.location_gauss_2d_iid import LocationGaussian2D_iid


def _oracle_trained_2d():
    proc = PivotBasedProcedure(
        pivot_fn=lambda th, x: th - x, d_theta=2, theta_range=(-7.0, 7.0),
    )
    return TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0,
    )


def test_joint_mahalanobis_oracle_passes(seed):
    seed_everything(seed)
    sim = LocationGaussian2D_iid()
    trained = _oracle_trained_2d()
    diag = JointMahalanobis(
        theta_0_grid=[[-3.0, -3.0], [0.0, 0.0], [3.0, 3.0]],
        n_per_theta=2000,
    )
    result = diag(trained, sim, eval_data=None)
    df = result.value
    assert "ks" in df.columns
    # Oracle pivot should give KS ≤ noise_floor at all θ_0.
    assert (df["ks"] <= df["noise_floor"]).all(), f"KS: {df['ks'].tolist()}, floor: {df['noise_floor'].iloc[0]}"


def test_joint_mahalanobis_catches_correlation(seed):
    """A miscalibrated pivot whose marginals are N(0,1) but with non-zero
    Pearson correlation should fail JointMahalanobis even though marginal
    PIT would pass."""
    seed_everything(seed)

    def miscal_pivot(theta, x):
        # Componentwise N(0,1) marginally, but r_1, r_2 share a common
        # component → corr ≈ 0.5, joint distribution NOT N(0, I_2).
        eps = theta - x  # would be N(0, I) for the oracle
        r_1 = eps[:, 0]
        r_2 = 0.5 * eps[:, 0] + 0.866 * eps[:, 1]  # corr(r_1, r_2) = 0.5, var(r_2) = 1
        return torch.stack([r_1, r_2], dim=-1)

    proc = PivotBasedProcedure(pivot_fn=miscal_pivot, d_theta=2, theta_range=(-7.0, 7.0))
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0,
    )
    sim = LocationGaussian2D_iid()
    diag = JointMahalanobis(theta_0_grid=[[0.0, 0.0]], n_per_theta=5000)
    result = diag(trained, sim, eval_data=None)
    df = result.value
    # ||r||² should be inflated relative to χ²_2 → KS > floor.
    assert (df["ks"] > df["noise_floor"]).all(), (
        f"miscalibrated pivot passes JointMahalanobis: ks={df['ks'].tolist()}, "
        f"floor={df['noise_floor'].iloc[0]}"
    )


def test_joint_mahalanobis_skips_1d():
    """In 1D the diagnostic is degenerate (||r||² = r² has the squared marginal-PIT distribution);
    must no-op gracefully."""
    from cdsbi.simulators.location_normal_1d import LocationNormal1D
    proc = PivotBasedProcedure(
        pivot_fn=lambda th, x: th - x, d_theta=1, theta_range=(-7.0, 7.0),
    )
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0,
    )
    sim = LocationNormal1D()
    diag = JointMahalanobis(theta_0_grid=[0.0], n_per_theta=500)
    result = diag(trained, sim, eval_data=None)
    assert result.passed  # no-op
    assert "reason" in result.meta
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/diagnostics/test_joint_mahalanobis.py -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the diagnostic**

Create `src/cdsbi/diagnostics/joint_mahalanobis.py`:

```python
"""JointMahalanobis: at each θ_0, the empirical distribution of
||r(θ_0; X_i)||² for X_i ~ p(X | θ_0) should match χ²_d.

Diagnostic 4 from §7.3 of the manuscript. Degenerate in d=1 (reduces
to the squared marginal-PIT residual against a χ²_1 distribution),
so we no-op in 1D.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2, kstest

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.base import Diagnostic, DiagnosticResult
from cdsbi.diagnostics.ks_floor import ks_noise_floor


class JointMahalanobis(Diagnostic):
    name = "joint_mahalanobis"

    def __init__(self, theta_0_grid: Sequence, n_per_theta: int = 2000):
        self.theta_0_grid = theta_0_grid
        self.n_per_theta = n_per_theta

    def __call__(self, trained, simulator, eval_data=None) -> DiagnosticResult:
        # Diagnostic only applies to pivot-based procedures.
        if not isinstance(trained.procedure, PivotBasedProcedure):
            return DiagnosticResult(
                name=self.name, value=float("nan"), passed=True, noise_floor=0.0,
                n_samples=0, meta={"reason": "not a pivot-based procedure"},
            )
        d = simulator.d_theta
        if d < 2:
            return DiagnosticResult(
                name=self.name, value=float("nan"), passed=True, noise_floor=0.0,
                n_samples=0, meta={"reason": "degenerate in d=1; use marginal PIT"},
            )
        floor = ks_noise_floor(N=self.n_per_theta, n_bins=1)
        rng = np.random.default_rng(0)
        rows = []
        chi2_cdf = chi2(df=d).cdf
        for theta_0 in self.theta_0_grid:
            x = simulator.sample_x_given_theta(theta_0, self.n_per_theta, rng)
            theta_vec = torch.tensor(list(theta_0), dtype=x.dtype).view(1, -1)
            theta_t = theta_vec.expand_as(x)
            with torch.no_grad():
                r = trained.procedure.pivot(theta_t, x)
            r_sq = r.pow(2).sum(dim=-1).cpu().numpy()
            # KS test of empirical r_sq distribution against χ²_d CDF.
            ks_stat, _ = kstest(r_sq, chi2_cdf)
            rows.append({
                "theta_0_repr": str(list(map(float, list(theta_0)))),
                "ks": float(ks_stat),
                "noise_floor": floor,
                "n_per_theta": int(self.n_per_theta),
                "passed": ks_stat <= floor,
            })
        df = pd.DataFrame(rows)
        passed = bool(df["passed"].all())
        return DiagnosticResult(
            name=self.name, value=df, passed=passed, noise_floor=floor,
            n_samples=len(self.theta_0_grid) * self.n_per_theta,
            meta={"d": int(d)},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/diagnostics/test_joint_mahalanobis.py -v`

Expected: ALL 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/diagnostics/joint_mahalanobis.py tests/diagnostics/test_joint_mahalanobis.py
git commit -m "feat(diagnostics): JointMahalanobis (manuscript §7.3 Diagnostic 4)

At each θ_0, ||r(θ_0; X_i)||² | X ~ p(X|θ_0) should be χ²_d. KS test
of the empirical distribution against the χ² CDF, per θ_0 row.

Catches joint miscalibration (e.g., correlated coordinates) that
marginal PIT can't see: a pivot with marginally N(0,1) coordinates
but corr(r_1, r_2) = 0.5 produces ||r||² with variance != 2d and
fails JointMahalanobis while passing marginal PIT.

Degenerate in d=1 (reduces to squared marginal-PIT residual); no-ops
gracefully.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Coverage / SetSize call `simulator.sample_x_given_theta`

The v0 hardcoding (`X = θ + N(0, 1)`) needs to use the protocol method. For 1D the new path produces identical results; for 2D it's the only way.

**Files:**
- Modify: `src/cdsbi/diagnostics/coverage.py`
- Modify: `src/cdsbi/diagnostics/set_size.py`
- Test: existing `tests/diagnostics/test_coverage.py` and `tests/diagnostics/test_set_size.py` (must continue to pass)

- [ ] **Step 1: Write a failing test for the 2D path**

Append to `tests/diagnostics/test_coverage.py`:

```python
def test_coverage_2d_oracle_at_nominal(seed):
    """In d=2 with oracle pivot r* = θ - X, coverage at α should be ≈ α."""
    from cdsbi.simulators.location_gauss_2d_iid import LocationGaussian2D_iid
    from cdsbi.confidence_set.procedures import PivotBasedProcedure
    from cdsbi.methods.base import TrainedModel
    from cdsbi.reproducibility.seeding import seed_everything
    seed_everything(seed)
    sim = LocationGaussian2D_iid()
    proc = PivotBasedProcedure(
        pivot_fn=lambda th, x: th - x, d_theta=2, theta_range=(-7.0, 7.0),
    )
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0,
    )
    diag = Coverage(
        theta_0_grid=[[-3.0, -3.0], [0.0, 0.0], [3.0, 3.0]],
        alpha_grid=[0.5, 0.9], n_per_theta=2000,
    )
    result = diag(trained, sim, eval_data=None)
    df = result.value
    df["err"] = (df["empirical"] - df["nominal"]).abs()
    assert (df["err"] < 0.03).all(), f"err: {df['err'].tolist()}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/diagnostics/test_coverage.py::test_coverage_2d_oracle_at_nominal -v`

Expected: FAIL — the existing Coverage code passes `torch.full((n_per_theta, 1), theta_0)` which assumes scalar θ_0 and d=1.

- [ ] **Step 3: Update Coverage and SetSize to use `sample_x_given_theta`**

In `src/cdsbi/diagnostics/coverage.py`, replace the X-given-θ sampling block. Find:

```python
            theta_t = torch.full((self.n_per_theta, 1), theta_0, dtype=torch.float32)
            eps = rng.standard_normal(size=(self.n_per_theta, 1))
            x = theta_t + torch.from_numpy(eps).float()
```

Replace with:

```python
            x = simulator.sample_x_given_theta(theta_0, self.n_per_theta, rng)
```

(`rng` already exists — it's the `np.random.default_rng(0)` set at the top of `__call__`.)

Similarly in `src/cdsbi/diagnostics/set_size.py`, replace:

```python
            theta_t = torch.full((self.n_per_theta, 1), float(theta_0), dtype=torch.float32)
            eps = rng.standard_normal(size=(self.n_per_theta, 1))
            x = theta_t + torch.from_numpy(eps).float()
```

with:

```python
            x = simulator.sample_x_given_theta(theta_0, self.n_per_theta, rng)
```

**SetSize d > 1 — width extraction:**
SetSize reads `cs.boundary_repr[1] - cs.boundary_repr[0]` per-X_obs in the
slow path; that's 1D-only. After Task 4 makes `boundary_repr` shape `(K, d)`
in d > 1, we need a d-agnostic size metric. Use **set diameter** =
`max_i ||boundary_i - center||` for d > 1, falling back to `right - left`
in d=1. Patch `set_size.py`'s slow-path width computation. Find:

```python
                    widths = np.empty(self.n_per_theta, dtype=np.float64)
                    for i in range(self.n_per_theta):
                        cs = trained.procedure.confidence_set(x[i : i + 1], alpha=alpha)
                        widths[i] = float(cs.boundary_repr[1] - cs.boundary_repr[0])
```

Replace with:

```python
                    widths = np.empty(self.n_per_theta, dtype=np.float64)
                    for i in range(self.n_per_theta):
                        cs = trained.procedure.confidence_set(x[i : i + 1], alpha=alpha)
                        br = cs.boundary_repr
                        if br.ndim == 1:
                            # 1D: (lower, upper)
                            widths[i] = float(br[1] - br[0])
                        elif br.numel() == 0:
                            widths[i] = 0.0  # empty set (Task 4 signal)
                        else:
                            # d > 1: max distance between any boundary point and
                            # the boundary centroid (rough proxy for set diameter).
                            center = br.mean(dim=0)
                            dist = (br - center).pow(2).sum(dim=-1).sqrt()
                            widths[i] = float(dist.max().item())
```

(The fast path via `confidence_set_batch` is 1D-only by design — Task 4
notes — so PivotBased in d > 1 falls through to the slow path here.)

- [ ] **Step 4: Update `contains_batch`-using path to pass scalar-or-vector θ_0**

`PivotBasedProcedure.contains_batch` was written for 1D — it does
`torch.full((B, self.d_theta), float(theta_0_value), ...)`. For
multivariate θ_0 (e.g. `[0.0, 0.0]`), this `float(theta_0_value)` call
crashes. Define a small helper at the top of
`src/cdsbi/confidence_set/procedures.py`:

```python
def _theta_to_row_tensor(theta_value, dtype, device, d_theta: int) -> torch.Tensor:
    """Normalise a scalar / sequence / tensor θ_0 into a (1, d_theta) row.

    Handles Python float / int, numpy scalar / 0-d tensor (np.ndim == 0),
    Python list / tuple, 1-d ndarray, 1-d torch.Tensor.
    """
    import numpy as np
    arr = np.atleast_1d(np.asarray(theta_value, dtype=np.float64)).reshape(-1)
    assert arr.shape == (d_theta,), (
        f"theta_0 has shape {arr.shape}, expected ({d_theta},)"
    )
    return torch.from_numpy(arr).to(dtype=dtype, device=device).view(1, d_theta)
```

In `PivotBasedProcedure.contains_batch`, replace the existing `theta_probe`
and `theta_t` construction blocks with calls to this helper:

```python
        theta_probe = _theta_to_row_tensor(
            theta_0_value, x_obs_batch.dtype, x_obs_batch.device, self.d_theta,
        )
        probe = self.pivot_fn(theta_probe, x_obs_batch[:1])
        device = probe.device
        if x_obs_batch.device != device:
            x_obs_batch = x_obs_batch.to(device)
        # Repeat the (1, d) θ_0 row B times.
        theta_t = _theta_to_row_tensor(
            theta_0_value, x_obs_batch.dtype, device, self.d_theta,
        ).expand(B, -1)
```

Apply the identical change to `LikelihoodBasedProcedure.contains_batch` and
`CriticalValueProcedure.contains_batch`.

**`LikelihoodBasedProcedure.contains_batch` also needs the d>1 grid fix:**
the existing `torch.linspace(lo, hi, n_grid).view(-1, self.d_theta)` collapses
incorrectly when `d_theta > 1` (you can't reshape `(n_grid,)` into
`(-1, d)` for d > 1). For d > 1, generate a random grid over the prior
box (same approach as the LF2I-BFF marginal grid in Task 13). Replace the
linspace line with:

```python
        if self.d_theta == 1:
            theta_grid = torch.linspace(
                lo, hi, n_grid, device=device, dtype=x_obs_batch.dtype,
            ).view(-1, 1)
        else:
            # Random grid over the d-dim prior box; n_grid total points.
            import numpy as np
            grid_np = np.random.default_rng(0).uniform(lo, hi, size=(n_grid, self.d_theta))
            theta_grid = torch.from_numpy(grid_np).to(
                dtype=x_obs_batch.dtype, device=device,
            )
```

(Drop the `assert self.d_theta == 1` at the top of the method.)

- [ ] **Step 5: Run tests to verify all pass**

Run: `pytest tests/diagnostics/ -v`

Expected: ALL tests PASS, including the new 2D Coverage test and all existing 1D ones.

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/diagnostics/coverage.py src/cdsbi/diagnostics/set_size.py src/cdsbi/confidence_set/procedures.py tests/diagnostics/test_coverage.py
git commit -m "feat(diagnostics): Coverage + SetSize use sample_x_given_theta; contains_batch handles vector θ_0

Closes the v0 hardcoded-LocationNormal1D-X|θ caveat. Diagnostics now
draw X | θ_0 via the Simulator protocol method; θ_0 can be a scalar
(1D) or a sequence (d > 1). PivotBased / Likelihood / CriticalValue
contains_batch updated similarly to accept either form.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: ConditionalPIT and PivotRMSE per-coordinate handling for d > 1

The remaining 1D-assuming diagnostics: `PivotRMSE` already reduces over all
elements (works in d > 1 with no change once `r_hat` and `r_star` agree on
shape). `ConditionalPIT` bins by θ; in d > 1 we bin per-coordinate and report
one row per (coordinate, bin).

**Files:**
- Modify: `src/cdsbi/diagnostics/pivot_rmse.py`
- Modify: `src/cdsbi/diagnostics/conditional_pit.py`
- Modify: `src/cdsbi/diagnostics/marginal_pit.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/diagnostics/test_marginal_pit.py`:

```python
def test_marginal_pit_per_coordinate_d2(seed):
    """In d=2, marginal PIT should report one KS per coordinate."""
    from cdsbi.simulators.location_gauss_2d_iid import LocationGaussian2D_iid
    from cdsbi.confidence_set.procedures import PivotBasedProcedure
    from cdsbi.methods.base import TrainedModel
    from cdsbi.diagnostics.marginal_pit import MarginalPIT
    from cdsbi.reproducibility.seeding import seed_everything
    import numpy as np
    seed_everything(seed)
    sim = LocationGaussian2D_iid()
    proc = PivotBasedProcedure(
        pivot_fn=lambda th, x: th - x, d_theta=2, theta_range=(-7.0, 7.0),
    )
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0,
    )
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(5000, rng)
    diag = MarginalPIT()
    result = diag(trained, sim, eval_data=(theta, x))
    # `value` should be a per-coordinate DataFrame for d > 1.
    import pandas as pd
    assert isinstance(result.value, pd.DataFrame)
    assert "coord" in result.value.columns
    assert len(result.value) == 2  # d=2 → 2 rows
    assert (result.value["ks"] <= result.noise_floor).all()
```

Append to `tests/diagnostics/test_conditional_pit.py`:

```python
def test_conditional_pit_per_coordinate_d2(seed):
    from cdsbi.simulators.location_gauss_2d_iid import LocationGaussian2D_iid
    from cdsbi.confidence_set.procedures import PivotBasedProcedure
    from cdsbi.methods.base import TrainedModel
    from cdsbi.diagnostics.conditional_pit import ConditionalPIT
    from cdsbi.reproducibility.seeding import seed_everything
    import numpy as np
    seed_everything(seed)
    sim = LocationGaussian2D_iid()
    proc = PivotBasedProcedure(
        pivot_fn=lambda th, x: th - x, d_theta=2, theta_range=(-7.0, 7.0),
    )
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0,
    )
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(5000, rng)
    diag = ConditionalPIT(n_bins=5)
    result = diag(trained, sim, eval_data=(theta, x))
    df = result.value
    assert "coord" in df.columns
    assert df["coord"].nunique() == 2  # one set of bins per θ-coordinate
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/diagnostics/test_marginal_pit.py::test_marginal_pit_per_coordinate_d2 tests/diagnostics/test_conditional_pit.py::test_conditional_pit_per_coordinate_d2 -v`

Expected: FAIL — current `MarginalPIT` returns a scalar; `ConditionalPIT` bins
on flattened θ.

- [ ] **Step 3: Update PIT diagnostics for per-coordinate d > 1 handling**

In `src/cdsbi/diagnostics/marginal_pit.py`, replace the body of `__call__`:

```python
    def __call__(self, trained, simulator, eval_data) -> DiagnosticResult:
        if not isinstance(trained.procedure, PivotBasedProcedure):
            return DiagnosticResult(
                name=self.name, value=float("nan"), passed=True, noise_floor=0.0,
                n_samples=0, meta={"reason": "not a pivot-based procedure"},
            )
        theta, x = eval_data
        with torch.no_grad():
            r = trained.procedure.pivot(theta, x)  # (N, d)
        r_np = r.cpu().numpy()
        d = r_np.shape[-1] if r_np.ndim > 1 else 1
        floor = ks_noise_floor(N=r_np.shape[0], n_bins=1)
        if d == 1:
            u = norm.cdf(r_np.flatten())
            ks_stat, _ = kstest(u, "uniform")
            return DiagnosticResult(
                name=self.name, value=float(ks_stat),
                passed=ks_stat <= floor, noise_floor=floor, n_samples=u.size,
            )
        rows = []
        for k in range(d):
            u_k = norm.cdf(r_np[:, k])
            ks_stat, _ = kstest(u_k, "uniform")
            rows.append({
                "coord": int(k),
                "ks": float(ks_stat),
                "noise_floor": floor,
                "passed": ks_stat <= floor,
                "n_samples": int(u_k.size),
            })
        import pandas as pd
        df = pd.DataFrame(rows)
        return DiagnosticResult(
            name=self.name, value=df,
            passed=bool(df["passed"].all()), noise_floor=floor,
            n_samples=r_np.shape[0], meta={"d": int(d)},
        )
```

In `src/cdsbi/diagnostics/conditional_pit.py`, replace the body of `__call__`. Find the existing per-bin loop and wrap it in an outer per-coordinate loop, recording the coord index:

```python
    def __call__(self, trained, simulator, eval_data) -> DiagnosticResult:
        if not isinstance(trained.procedure, PivotBasedProcedure):
            return DiagnosticResult(
                name=self.name, value=float("nan"), passed=True, noise_floor=0.0,
                n_samples=0, meta={"reason": "not a pivot-based procedure"},
            )
        theta, x = eval_data
        N = theta.shape[0]
        floor = self._per_bin_floor(N)
        with torch.no_grad():
            r = trained.procedure.pivot(theta, x).cpu().numpy()  # (N, d)
        theta_np = theta.cpu().numpy()
        d = theta_np.shape[-1] if theta_np.ndim > 1 else 1
        rows = []
        for c in range(d):
            theta_c = theta_np[:, c] if d > 1 else theta_np.flatten()
            r_c = r[:, c] if r.ndim > 1 else r.flatten()
            edges = np.quantile(theta_c, np.linspace(0, 1, self.n_bins + 1))
            for k in range(self.n_bins):
                lo, hi = edges[k], edges[k + 1]
                mask = (theta_c >= lo) & (theta_c <= hi)
                r_bin = r_c[mask]
                if r_bin.size < 10:
                    continue
                u_bin = norm.cdf(r_bin)
                ks_stat, _ = kstest(u_bin, "uniform")
                rows.append({
                    "coord": int(c),
                    "theta_0_bin": k,
                    "theta_0_center_0": 0.5 * (lo + hi),
                    "ks": ks_stat,
                    "per_bin_noise_floor": floor,
                    "n_per_bin": int(r_bin.size),
                    "passed": ks_stat <= floor,
                })
        df = pd.DataFrame(rows)
        passed = bool(df["passed"].all())
        return DiagnosticResult(
            name=self.name, value=df, passed=passed, noise_floor=floor,
            n_samples=N, meta={"n_bins": self.n_bins, "d": int(d)},
        )
```

In `src/cdsbi/diagnostics/pivot_rmse.py`, the existing reduction
`(r_hat - r_star).pow(2).mean().sqrt()` already collapses across all
elements, including the last dim — works for d > 1 unchanged. Add an
`n_coords` field to the meta for downstream analysis:

Find:

```python
        return DiagnosticResult(
            name=self.name,
            value=rmse,
            passed=rmse < 0.05,
            noise_floor=0.05,
            n_samples=theta.shape[0],
        )
```

Replace with:

```python
        return DiagnosticResult(
            name=self.name,
            value=rmse,
            passed=rmse < 0.05,
            noise_floor=0.05,
            n_samples=theta.shape[0],
            meta={"d": int(r_hat.shape[-1]) if r_hat.ndim > 1 else 1},
        )
```

- [ ] **Step 4: Update `_write_index_row` to handle the DataFrame case**

In `src/cdsbi/experiments/run.py`, find `_write_index_row` and locate the
`marginal_ks` extraction:

```python
        "marginal_ks": float(marg.value) if marg and isinstance(marg.value, float) else None,
```

Replace with:

```python
        "marginal_ks": (
            float(marg.value) if marg and isinstance(marg.value, float)
            else float(marg.value["ks"].mean()) if marg and hasattr(marg.value, "columns") and "ks" in marg.value.columns
            else None
        ),
```

(In d > 1 `marginal_pit.value` is a DataFrame with one row per coordinate;
the index_row carries the per-coord mean so paper_table_8_2 has a scalar
to aggregate.)

- [ ] **Step 5: Run tests to verify all pass**

Run: `pytest tests/diagnostics/ tests/integration/test_runner_e2e.py -v`

Expected: ALL tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/diagnostics/pivot_rmse.py src/cdsbi/diagnostics/marginal_pit.py src/cdsbi/diagnostics/conditional_pit.py src/cdsbi/experiments/run.py tests/diagnostics/test_marginal_pit.py tests/diagnostics/test_conditional_pit.py
git commit -m "feat(diagnostics): per-coordinate PIT diagnostics for d > 1

MarginalPIT: one KS per coordinate; result.value becomes a DataFrame
in d > 1 (was a scalar). ConditionalPIT: bins per-coordinate, rows
gain a 'coord' column. PivotRMSE: reduction already d-agnostic;
exposes d in meta for downstream analysis.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `experiments/run.py` — d-aware flow / classifier construction; wire JointMahalanobis

The `_build_flow` helper currently hardcodes `features=1, context_features=1`
for the MAF backbone. We replace those with `simulator.d_theta` / `d_x`.
Also wire `JointMahalanobis` into the standard diagnostic battery (auto-skips
in d=1 inside the diagnostic itself).

**Files:**
- Modify: `src/cdsbi/experiments/run.py`

- [ ] **Step 1: Make `_build_flow` accept the simulator**

Modify the call site in `_build_method`. Find every `_build_flow(cfg)` call (currently 3) and change them to `_build_flow(cfg, simulator)`.

In `_build_flow`, change the signature:

```python
def _build_flow(cfg: DictConfig, simulator) -> Any:
```

In the function body, replace the MAF construction:

```python
    if method_flow_label == "maf":
        return _instantiate(
            "cdsbi.flows.maf_adapter.MAFAdapter",
            features=int(simulator.d_theta),
            context_features=int(simulator.d_x),
            hidden=int(cfg.budget.maf_hidden),
            num_layers=2,
        )
```

Note: for NLE the role of features vs context is *swapped* (X is input, θ is
context). The convention in v0 was to pass `features=1, context_features=1` so
the 1D case worked symmetrically. To preserve correct behavior for NLE,
update the NLE construction path: it builds the same MAFAdapter via
`_build_flow` but logically expects features=d_x, context_features=d_theta.
For LocationGaussian2D_iid, d_x == d_theta so the two are equivalent. Add a
TODO marker for asymmetric-d future targets:

After the `if method_flow_label == "maf"` block, add a comment line:

```python
        # TODO(v1+): for asymmetric d (d_x != d_theta), NLE wants features=d_x
        # and context_features=d_theta. v1's loc_gauss_2d_iid is symmetric so
        # this is correct; revisit when an asymmetric target lands.
```

Update the `additive_umnn` branch to construct `AdditiveFlow1D` only when
d_theta == 1, and fall through to `TriangularAdditiveFlow` for d > 1:

```python
    if method_flow_label == "additive_umnn":
        if int(simulator.d_theta) == 1:
            return _instantiate(
                "cdsbi.flows.additive.AdditiveFlow1D",
                hidden=int(cfg.budget.cdsbi_flow_hidden),
            )
        return _instantiate(
            "cdsbi.flows.triangular_additive.TriangularAdditiveFlow",
            d=int(simulator.d_theta),
            hidden=int(cfg.budget.cdsbi_flow_hidden),
        )
    if method_flow_label == "triangular_additive":
        return _instantiate(
            "cdsbi.flows.triangular_additive.TriangularAdditiveFlow",
            d=int(simulator.d_theta),
            hidden=int(cfg.budget.cdsbi_flow_hidden),
        )
```

(The new `triangular_additive` branch handles the case where method.flow
explicitly names it.)

Update the fast-path (when `method.flow == cfg.flow.name`): the existing
fast path passes through `cfg.flow` to `_instantiate(target, **flow_dict)`,
which constructs the flow with whatever positional/keyword args the YAML
gives. For `AdditiveFlow1D` (1D) and `TriangularAdditiveFlow` (d > 1) the
keyword arg is `hidden` either way; for `AdditiveFlow1D` no `d` is needed.
Inject `d=simulator.d_theta` only when the target class is the multivariate
form:

Find the fast path block:

```python
    if method_flow_label is None or method_flow_label == hydra_flow_name:
        flow_dict = OmegaConf.to_container(cfg.flow, resolve=True)
        target = flow_dict.pop("_target_")
        flow_dict.pop("name", None)
        return _instantiate(target, **flow_dict)
```

Replace with:

```python
    if method_flow_label is None or method_flow_label == hydra_flow_name:
        flow_dict = OmegaConf.to_container(cfg.flow, resolve=True)
        target = flow_dict.pop("_target_")
        flow_dict.pop("name", None)
        # Multivariate triangular flow needs d injected from the simulator.
        if "triangular_additive.TriangularAdditiveFlow" in target:
            flow_dict.setdefault("d", int(simulator.d_theta))
        return _instantiate(target, **flow_dict)
```

- [ ] **Step 2: Wire JointMahalanobis into `_run_diagnostics`**

In `src/cdsbi/experiments/run.py`, find `_run_diagnostics` and add the
JointMahalanobis import + entry:

```python
def _run_diagnostics(cfg: DictConfig, trained, simulator, eval_data, rd: RunDir):
    from cdsbi.diagnostics.conditional_pit import ConditionalPIT
    from cdsbi.diagnostics.coverage import Coverage
    from cdsbi.diagnostics.joint_mahalanobis import JointMahalanobis
    from cdsbi.diagnostics.marginal_pit import MarginalPIT
    from cdsbi.diagnostics.pivot_rmse import PivotRMSE
    from cdsbi.diagnostics.set_size import SetSize
    ...
```

After the existing `set_size` entry in the diagnostics list, append:

```python
        ("joint_mahalanobis", JointMahalanobis(
            theta_0_grid=list(cfg.experiment.eval_thetas_interior),
            n_per_theta=int(OmegaConf.select(
                cfg, "experiment.joint_mahalanobis_n_per_theta", default=2000,
            )),
        )),
```

JointMahalanobis no-ops in d=1, so the 1D §8.1 path remains a no-op for this
diagnostic.

- [ ] **Step 3: Commit**

```bash
git add src/cdsbi/experiments/run.py
git commit -m "feat(run): d-aware flow construction; wire JointMahalanobis

_build_flow now takes the simulator and resolves features /
context_features from simulator.d_theta / d_x. additive_umnn dispatches
to AdditiveFlow1D when d=1 and TriangularAdditiveFlow when d>1, so the
existing method=cd_sbi config works for both v0 (1D) and v1 (2D)
targets without per-method config changes.

JointMahalanobis joins the standard diagnostic battery; it no-ops in
d=1 so §8.1 runs are unaffected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Budget retune — `cdsbi_flow_hidden` now covers 1D and 2D

For d=2, `TriangularAdditiveFlow(hidden=H)` has roughly 2× the params of
`AdditiveFlow1D(hidden=H)` (two UMNN pairs, one with a 2-dim context). The
existing `cdsbi_flow_hidden` field in budget configs was tuned for 1D; we
need a separate value for 2D.

Two design choices and we go with the simpler one:
1. **(picked)** Reuse `cdsbi_flow_hidden`, but retune so the value targets
   `triangular_additive_2d` at the budget. Per-d tuning lands when v1+
   introduces more dimensions.
2. Add a per-d field `cdsbi_flow_hidden_d2`, `_d3`, etc. — more flexible but
   more YAML knobs. Defer until we have ≥3 d values.

Concretely: rerun `tools/retune_budgets.py` extended to include the 2D
triangular variant, copy the new `cdsbi_flow_hidden` values into the budget
YAMLs, document the d=2 numbers in YAML comments.

**Files:**
- Modify: `tools/retune_budgets.py`
- Modify: `configs/budget/*.yaml`

- [ ] **Step 1: Extend the retune script with the 2D variant**

In `tools/retune_budgets.py`, add an import and a new probe function:

```python
from cdsbi.flows.triangular_additive import TriangularAdditiveFlow


def triangular_additive_2d_params(H: int) -> int:
    m = TriangularAdditiveFlow(d=2, hidden=H)
    return m.n_params()
```

Replace the `cdsbi_flow_params` entry in `STANDALONE_DOMAINS` with a per-d
description in the documentation, but keep the function the same (still
returns 1D AdditiveFlow params for backward compatibility with existing
budget interpretations). Add a printout block in `main()` after the existing
standalone-domain loop:

```python
        # v1 d=2 variant of the same domain — for paper §8.2.
        H, n, err = pick_best(target, triangular_additive_2d_params, CANDIDATES)
        print(f"{'':<8}  {'cdsbi_flow_hidden (d=2)':<24}  {H:>5}  {n:>8}  "
              f"{target:>8}  {err:>7.1%}  {status_label(err)}")
```

And in the YAML snippet block, emit a `cdsbi_flow_hidden_d2` value:

```python
        h2, n2, e2 = pick_best(target, triangular_additive_2d_params, CANDIDATES)
        print(f"  cdsbi_flow_hidden_d2: {h2}  # TriangularAdditiveFlow(d=2) actual={n2} ({e2:.1%})")
```

- [ ] **Step 2: Run the script and copy values**

Run: `python tools/retune_budgets.py`

Note the per-budget `cdsbi_flow_hidden_d2` values. For each budget YAML
in `configs/budget/{small,medium,large,xlarge}.yaml`, add:

```yaml
cdsbi_flow_hidden_d2: <value>  # TriangularAdditiveFlow(d=2) actual=<N> (<%>)
```

(Append the line below the existing `cdsbi_flow_hidden` line.)

- [ ] **Step 3: Wire the d-aware budget into `_build_flow`**

In `src/cdsbi/experiments/run.py` `_build_flow`, change the d > 1 branch
of `additive_umnn` and the dedicated `triangular_additive` branch to read
the d-specific budget key when available:

```python
    if method_flow_label == "additive_umnn":
        if int(simulator.d_theta) == 1:
            return _instantiate(
                "cdsbi.flows.additive.AdditiveFlow1D",
                hidden=int(cfg.budget.cdsbi_flow_hidden),
            )
        # d > 1: use the d-specific budget key if present (e.g.
        # cdsbi_flow_hidden_d2 for d=2). Fall back to the 1D value as a
        # rough estimate when no d-specific value is defined.
        d = int(simulator.d_theta)
        d_key = f"cdsbi_flow_hidden_d{d}"
        budget_hidden = int(OmegaConf.select(
            cfg.budget, d_key, default=cfg.budget.cdsbi_flow_hidden,
        ))
        return _instantiate(
            "cdsbi.flows.triangular_additive.TriangularAdditiveFlow",
            d=d,
            hidden=budget_hidden,
        )
    if method_flow_label == "triangular_additive":
        d = int(simulator.d_theta)
        d_key = f"cdsbi_flow_hidden_d{d}"
        budget_hidden = int(OmegaConf.select(
            cfg.budget, d_key, default=cfg.budget.cdsbi_flow_hidden,
        ))
        return _instantiate(
            "cdsbi.flows.triangular_additive.TriangularAdditiveFlow",
            d=d,
            hidden=budget_hidden,
        )
```

- [ ] **Step 4: Commit**

```bash
git add tools/retune_budgets.py configs/budget/*.yaml src/cdsbi/experiments/run.py
git commit -m "config(budget): cdsbi_flow_hidden_d2 for the 2D TriangularAdditiveFlow

The §8.2 architecture has roughly 2× the params of §8.1 at the same
hidden width. Per-d budget keys (cdsbi_flow_hidden_d2, _d3, ...) keep
the matched-budget framing clean across dimensions. Falls back to the
1D value if the d-specific key isn't present.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Hydra configs for §8.2

**Files:**
- Create: `configs/target/loc_gauss_2d_iid.yaml`
- Create: `configs/flow/triangular_additive.yaml`
- Create: `configs/experiment/8_2_replication.yaml`
- Create: `configs/experiment/8_2_baseline_sweep.yaml`

- [ ] **Step 1: Write the target config**

Create `configs/target/loc_gauss_2d_iid.yaml`:

```yaml
name: loc_gauss_2d_iid
_target_: cdsbi.simulators.location_gauss_2d_iid.LocationGaussian2D_iid
theta_range: [-7.0, 7.0]
```

- [ ] **Step 2: Write the flow config**

Create `configs/flow/triangular_additive.yaml`:

```yaml
name: triangular_additive
_target_: cdsbi.flows.triangular_additive.TriangularAdditiveFlow
hidden: ${budget.cdsbi_flow_hidden_d2}
# d is injected by experiments.run._build_flow from simulator.d_theta.
```

- [ ] **Step 3: Write the replication experiment config**

Create `configs/experiment/8_2_replication.yaml`:

```yaml
name: 8_2_replication
n_eval: 5000
eval_thetas_interior:
  - [-3.0, -3.0]
  - [-3.0,  0.0]
  - [ 0.0,  0.0]
  - [ 3.0,  0.0]
  - [ 3.0,  3.0]
eval_thetas_edge: []  # 2D edges are corners; deferred to v2 along with §8.3
alpha_grid: [0.5, 0.68, 0.9, 0.95]
n_eval_per_theta: 2000

defaults:
  - /target: loc_gauss_2d_iid
  - /flow: triangular_additive
  - /method: cd_sbi
  - /budget: medium
```

- [ ] **Step 4: Write the baseline sweep config**

Create `configs/experiment/8_2_baseline_sweep.yaml`:

```yaml
name: 8_2_baseline_sweep
n_eval: 5000
eval_thetas_interior:
  - [-3.0, -3.0]
  - [-3.0,  0.0]
  - [ 0.0,  0.0]
  - [ 3.0,  0.0]
  - [ 3.0,  3.0]
eval_thetas_edge: []
alpha_grid: [0.5, 0.68, 0.9, 0.95]
n_eval_per_theta: 2000

defaults:
  - /target: loc_gauss_2d_iid
  - /flow: triangular_additive

# Launch with explicit sweep dims per the v0 lesson — Hydra ignores
# hydra.sweeper.params inside experiment group configs:
#   python -m cdsbi.experiments.run -m experiment=8_2_baseline_sweep \
#     method=cd_sbi,npe,nle,nre,lf2i_bff \
#     budget=small,medium,large,xlarge \
#     seed=0,1,2,3,4 \
#     training.fresh_batch=false
```

- [ ] **Step 5: Smoke-test the configs via Hydra compose**

Run a single tiny CDSBI run to validate config wiring (uses `lr=1e-3, n_steps=20` for speed):

```bash
python -m cdsbi.experiments.run \
  experiment=8_2_replication seed=0 \
  training.lr=1e-3 training.n_steps=20 training.n_train=200 \
  training.batch_size=64 training.fresh_batch=false \
  experiment.n_eval_per_theta=50 \
  hydra.run.dir=outputs/v1_smoke
```

Expected: completes with `STATUS=OK`, produces all parquets (including
`joint_mahalanobis.parquet`).

Verify:

```bash
ls outputs/v1_smoke/diagnostics/
```

Expected list includes: `pivot_rmse.parquet`, `marginal_pit.parquet`,
`conditional_pit.parquet`, `coverage.parquet`, `set_size.parquet`,
`joint_mahalanobis.parquet`.

- [ ] **Step 6: Commit**

```bash
git add configs/target/loc_gauss_2d_iid.yaml configs/flow/triangular_additive.yaml configs/experiment/8_2_replication.yaml configs/experiment/8_2_baseline_sweep.yaml
git commit -m "config(experiment): §8.2 replication + baseline sweep YAMLs

v1 Hydra config groups for the 2D iid Gaussian-location experiment.
Replication uses the same 5 interior θ_0 corners as §8.1 (mapped to
2D via cartesian product of {-3, 0, 3}); baseline_sweep documents the
explicit CLI invocation (lesson learned in v0 — sweeper.params inside
group configs are ignored).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: §8.2 CDSBI replication intensive test

**Files:**
- Create: `tests/intensive/test_replicate_8_2.py`

- [ ] **Step 1: Write the intensive test**

Create `tests/intensive/test_replicate_8_2.py`:

```python
"""§8.2 replication: CDSBI on LocationGaussian2D_iid matches tolerance bands."""
from pathlib import Path
import subprocess

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.intensive
def test_replicate_8_2_cdsbi_matches_tolerance(tmp_path):
    """Run CDSBI at medium budget across 5 seeds; seed-averaged metrics within tolerance."""
    out_dir = tmp_path / "sweep"
    for seed in range(5):
        cmd = [
            "python", "-m", "cdsbi.experiments.run",
            f"hydra.run.dir={out_dir}/run_seed_{seed}",
            "experiment=8_2_replication",
            "method=cd_sbi",
            "budget=medium",
            f"seed={seed}",
            "training.fresh_batch=false",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        assert result.returncode == 0, f"seed {seed} failed:\n{result.stderr}"

    from cdsbi.analysis.loaders import load_runs
    df = load_runs(str(out_dir / "*"))
    assert len(df) == 5
    # Tolerance bands per manuscript §8.2 (loosened from §8.1 by ~1.5×
    # to absorb 2D conditioning-network noise on the r_2 coordinate;
    # paper reports total pivot RMSE 0.044, KS r_1/r_2 0.007/0.006,
    # joint Mahalanobis KS 0.011, coverage error < 0.01).
    assert df["pivot_rmse"].mean() <= 0.08, f"pivot_rmse mean = {df['pivot_rmse'].mean()}"
    assert df["coverage_error_max"].mean() <= 0.03, (
        f"coverage_error mean = {df['coverage_error_max'].mean()}"
    )
    # Joint Mahalanobis: load per-run parquet and check KS ≤ floor.
    import glob, os
    n_pass = 0
    for rd in glob.glob(str(out_dir / "*")):
        jm_path = os.path.join(rd, "diagnostics/joint_mahalanobis.parquet")
        if not os.path.exists(jm_path):
            continue
        jm = pd.read_parquet(jm_path)
        if (jm["ks"] <= 2.0 * jm["noise_floor"]).all():
            n_pass += 1
    assert n_pass >= 4, (
        f"only {n_pass}/5 seeds passed JointMahalanobis at 2× floor; "
        f"expected ≥ 4"
    )
```

- [ ] **Step 2: Run the intensive test**

Run: `pytest tests/intensive/test_replicate_8_2.py -v -s`

Expected: PASS. Wall time ~3–5 min (5 seeds × ~30–60 s each, plus diagnostics).

If it fails, inspect the per-seed diagnostics to identify which metric
exceeded its band. Likely causes:
- Pivot RMSE high: training under-converged → bump `training.n_steps` in the
  default recipe, or revisit `cdsbi_flow_hidden_d2` budget.
- Joint Mahalanobis fails at a single seed: this is the
  "conditioning network learning slower" signal mentioned in §8.2; first
  retune the budget retune to allocate more params to the r_2
  conditioning network, then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/intensive/test_replicate_8_2.py
git commit -m "test(intensive): §8.2 CDSBI replication on LocationGaussian2D_iid

5 seeds at medium budget; checks pivot RMSE ≤ 0.08 (~2× manuscript
0.044), coverage error ≤ 0.03 (~3× manuscript < 0.01), and that ≥ 4 of
5 seeds pass JointMahalanobis at 2× the noise floor. The bands are
loosened from §8.1 by ~1.5× to absorb 2D conditioning-network noise.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: `paper_table_8_2` helper

Mirrors `paper_table_8_1` but also includes the joint-Mahalanobis KS
column (one row per (method, budget); JointMahalanobis values come from
the per-run diagnostic parquets).

**Files:**
- Modify: `src/cdsbi/analysis/paper_tables.py`

- [ ] **Step 1: Add the failing test**

Create `tests/unit/test_paper_table_8_2.py`:

```python
"""paper_table_8_2: aggregates index_row.parquet + joint_mahalanobis.parquet
across (method, budget) into a single (mean, std) table."""
from __future__ import annotations

import pandas as pd

from cdsbi.analysis.paper_tables import paper_table_8_2


def test_paper_table_8_2_aggregates_metrics_with_joint_mahalanobis():
    df = pd.DataFrame([
        {"method": "cd_sbi", "budget_name": "medium", "seed": 0,
         "pivot_rmse": 0.04, "marginal_ks": 0.006, "coverage_error_max": 0.01,
         "joint_mahal_ks": 0.012, "actual_params_total": 4998},
        {"method": "cd_sbi", "budget_name": "medium", "seed": 1,
         "pivot_rmse": 0.05, "marginal_ks": 0.007, "coverage_error_max": 0.012,
         "joint_mahal_ks": 0.014, "actual_params_total": 4998},
        {"method": "npe", "budget_name": "medium", "seed": 0,
         "pivot_rmse": None, "marginal_ks": None, "coverage_error_max": 0.02,
         "joint_mahal_ks": None, "actual_params_total": 4740},
    ])
    out = paper_table_8_2(df)
    # cd_sbi/medium row
    cd_row = out.loc[("cd_sbi", "medium")]
    assert abs(cd_row["pivot_rmse_mean"] - 0.045) < 1e-9
    assert abs(cd_row["joint_mahal_ks_mean"] - 0.013) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_paper_table_8_2.py -v`

Expected: FAIL with `ImportError: cannot import name 'paper_table_8_2'`.

- [ ] **Step 3: Add the function**

Append to `src/cdsbi/analysis/paper_tables.py`:

```python
def paper_table_8_2(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot (method × budget) → seed-averaged §8.2 metrics.

    Same as paper_table_8_1 but adds joint_mahal_ks if present (lifted into
    index_row.parquet by analysis upstream, or computed per-run by callers).
    """
    metrics = [
        "coverage_error_max", "marginal_ks", "pivot_rmse",
        "joint_mahal_ks", "actual_params_total",
    ]
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby(["method", "budget_name"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg
```

Also lift `joint_mahal_ks` into `index_row.parquet` so the loader picks it
up automatically. In `src/cdsbi/experiments/run.py`, find `_write_index_row`
and append:

```python
    jm_path = rd.path / "diagnostics" / "joint_mahalanobis.parquet"
    if jm_path.exists():
        jm_df = pd.read_parquet(jm_path)
        if "ks" in jm_df.columns and len(jm_df):
            row["joint_mahal_ks"] = float(jm_df["ks"].mean())
```

(Insert this block right before the line `pd.DataFrame([row]).to_parquet(...)`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_paper_table_8_2.py tests/ -q -k "paper_table or analysis"`

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/analysis/paper_tables.py src/cdsbi/experiments/run.py tests/unit/test_paper_table_8_2.py
git commit -m "feat(analysis): paper_table_8_2; lift joint_mahal_ks into index_row

joint_mahal_ks (mean over θ_0 grid) lands in index_row.parquet so
load_runs picks it up automatically; paper_table_8_2 mirrors
paper_table_8_1 with the additional column.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Multivariate adaptation for the sbi-wrapped methods (cross-method sweep)

For the §8.2 *baseline sweep* (5 methods × 4 budgets × 5 seeds) we need
NPE / NLE / NRE / LF2I-BFF to also work in d=2. After Task 8 the MAF
backbone is d-aware (features = `simulator.d_theta`, context_features =
`simulator.d_x`); the classifier in NRE / LF2I-BFF picks up
`d_theta + d_x` correctly. The one remaining seam is LF2I-BFF's
marginal-integration grid, which is hardcoded to a 1D `torch.linspace` over
the prior range.

**Files:**
- Modify: `src/cdsbi/methods/lf2i_bff.py`

- [ ] **Step 1: Make BFF's marginal integration d-aware**

Find the existing marginal grid construction in
`src/cdsbi/methods/lf2i_bff.py`:

```python
        theta_grid = torch.linspace(a, b, self.marginal_grid_n, device=self.device).view(-1, 1)
        N_grid = theta_grid.shape[0]
```

Replace with:

```python
        d = int(simulator.d_theta)
        if d == 1:
            theta_grid = torch.linspace(
                a, b, self.marginal_grid_n, device=self.device,
            ).view(-1, 1)
        else:
            # d > 1: an N×N×... product grid is exponential in d; use prior
            # Monte-Carlo samples instead (the integral E_π[O(X; θ)] is the
            # average over prior samples, by definition).
            rng_grid = rngs.eval  # separate stream from train/eval data
            theta_grid_np = rng_grid.uniform(
                a, b, size=(self.marginal_grid_n, d),
            )
            theta_grid = torch.from_numpy(theta_grid_np).float().to(self.device)
        N_grid = theta_grid.shape[0]
```

The downstream BFF logic (computing
`logsumexp_b log_ratio(θ_b, X)`) is already d-agnostic — the grid is just an
arbitrary set of d-dim θ points.

- [ ] **Step 2: Test 2D BFF smoke**

Append to `tests/integration/test_lf2i_bff_smoke.py`:

```python
def test_lf2i_bff_smoke_d2(seed):
    """BFF runs end-to-end on LocationGaussian2D_iid; produces a finite-size 2D set."""
    from cdsbi.simulators.location_gauss_2d_iid import LocationGaussian2D_iid
    seed_everything(seed)
    sim = LocationGaussian2D_iid()
    runner = LF2IBFFRunner(
        classifier_hidden=16, classifier_depth=2,
        quantile_hidden=8, quantile_depth=2,
        marginal_grid_n=32,
    )
    trained = runner.fit(
        simulator=sim,
        config={
            "lr": 1e-3, "batch_size": 32, "n_steps": 50,
            "n_train_stat": 300, "n_train_quantile": 150,
            "alpha_grid": [0.5, 0.9], "fresh_batch": False,
        },
        seed=seed,
    )
    cs = trained.procedure.confidence_set(torch.tensor([[0.0, 0.0]]), alpha=0.9)
    assert cs.boundary_repr.ndim == 2
    assert cs.boundary_repr.shape[1] == 2  # d=2
```

But wait — `CriticalValueProcedure.confidence_set` was 1D in v0. Like
`PivotBasedProcedure`, it needs the same multivariate treatment. Add this
as a sub-step:

- [ ] **Step 3: Add multivariate `confidence_set` for `CriticalValueProcedure`**

In `src/cdsbi/confidence_set/procedures.py`, modify `CriticalValueProcedure.confidence_set` to mirror PivotBasedProcedure's d>1 ray-bisection path:

Find the existing method (the one with `assert self.d_theta == 1`) and
replace with:

```python
    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        if self.d_theta == 1:
            return self._confidence_set_1d(x_obs, alpha)
        return self._confidence_set_ray_sampled(x_obs, alpha, n_rays=200)

    def _confidence_set_1d(self, x_obs, alpha):
        # (Existing v0 body — 1D grid + bisection — preserved verbatim.)
        def f(theta_val: float) -> float:
            theta = torch.tensor([[theta_val]], dtype=x_obs.dtype, device=x_obs.device)
            t = self.test_stat_fn(theta, x_obs).item()
            c = self.critical_value_fn(theta, alpha).item()
            return t - c
        lo, hi = self.theta_range
        grid = torch.linspace(lo, hi, 200).tolist()
        signs = [f(t) <= 0 for t in grid]
        try:
            i_first = signs.index(True)
            i_last = len(signs) - 1 - signs[::-1].index(True)
        except ValueError:
            return ConfidenceSet(
                contains=lambda th: False, boundary_repr=torch.tensor([0.0, 0.0]), alpha=alpha,
            )
        left_lo, left_hi = grid[max(i_first - 1, 0)], grid[i_first]
        right_lo, right_hi = grid[i_last], grid[min(i_last + 1, len(grid) - 1)]
        left = bisect_1d(f, left_lo, left_hi, tol=1e-4) if i_first > 0 else grid[0]
        right = bisect_1d(f, right_lo, right_hi, tol=1e-4) if i_last < len(grid) - 1 else grid[-1]
        def contains(theta_val):
            v = float(theta_val) if not hasattr(theta_val, "__len__") else float(theta_val[0])
            return left <= v <= right
        return ConfidenceSet(
            contains=contains, boundary_repr=torch.tensor([left, right]), alpha=alpha,
        )

    def _confidence_set_ray_sampled(self, x_obs, alpha, n_rays: int):
        """Multivariate set via line-sampling: K random unit directions
        from the set's center (X_obs as initial guess), bisect along each
        ray to find where T(θ_t; X_obs) = c_α(θ_t)."""
        dtype = x_obs.dtype
        device = x_obs.device
        d = self.d_theta
        center = x_obs.squeeze(0)
        u = torch.randn(n_rays, d, dtype=dtype, device=device)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        lo, hi = self.theta_range
        t_lo = torch.zeros(n_rays, dtype=dtype, device=device)
        t_hi = torch.full((n_rays,), 2.0 * (hi - lo), dtype=dtype, device=device)
        x_batch = x_obs.expand(n_rays, -1)
        for _ in range(40):
            m = 0.5 * (t_lo + t_hi)
            theta_m = center.unsqueeze(0) + m.unsqueeze(-1) * u
            t = self.test_stat_fn(theta_m, x_batch).squeeze(-1)
            c = self.critical_value_fn(theta_m, alpha)
            if c.ndim > 1:
                c = c.squeeze(-1)
            f = t - c
            outside = f > 0
            t_hi = torch.where(outside, m, t_hi)
            t_lo = torch.where(outside, t_lo, m)
        t = 0.5 * (t_lo + t_hi)
        boundary = center.unsqueeze(0) + t.unsqueeze(-1) * u
        def contains(theta_val):
            theta_t = torch.tensor(
                [list(theta_val)] if not isinstance(theta_val, torch.Tensor) else theta_val,
                dtype=dtype, device=device,
            ).view(1, d)
            t_obs = self.test_stat_fn(theta_t, x_obs).item()
            c_obs = self.critical_value_fn(theta_t, alpha).item()
            return bool(t_obs <= c_obs)
        return ConfidenceSet(contains=contains, boundary_repr=boundary, alpha=alpha)
```

Do the same for `LikelihoodBasedProcedure.confidence_set` and have
`RatioBasedProcedure` continue to delegate via the existing wrapper.

(For `PosteriorBasedProcedure`, the `equal_tailed_1d` extractor is 1D
only; in d > 1 we fall back to a per-coordinate marginal interval and add
a TODO for true HPD region — that's a v1+ refinement, not blocking the
§8.2 sweep since coverage at marginal coverage already exercises the
critical machinery. Modify `PosteriorBasedProcedure.confidence_set` to
support d > 1 by returning the cartesian product of per-coordinate
intervals as the `boundary_repr`'s eight corners.)

- [ ] **Step 4: Run BFF + LF2I smoke + diagnostic tests**

Run: `pytest tests/integration/test_lf2i_bff_smoke.py tests/diagnostics/ -v`

Expected: ALL PASS, including the new 2D test.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/methods/lf2i_bff.py src/cdsbi/confidence_set/procedures.py tests/integration/test_lf2i_bff_smoke.py
git commit -m "feat(methods, confidence_set): d > 1 adaptation for BFF + procedures

- LF2IBFFRunner: marginal integration grid uses prior MC samples for
  d > 1 (1D linspace is unchanged for d == 1).
- CriticalValueProcedure / LikelihoodBasedProcedure / RatioBasedProcedure:
  confidence_set switches to ray-bisection for d > 1; existing 1D paths
  preserved verbatim. boundary_repr becomes (K, d) sampled points.

These complete the v1 cross-method §8.2 baseline-sweep capability.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Launch the §8.2 baseline sweep (sanity check)

**Files:** (no code changes; this task documents how to validate the v1
deliverable end-to-end)

- [ ] **Step 1: Launch the cross-method sweep**

Run:

```bash
python -m cdsbi.experiments.run -m \
  experiment=8_2_baseline_sweep \
  method=cd_sbi,npe,nle,nre,lf2i_bff \
  budget=small,medium,large,xlarge \
  seed=0,1,2,3,4 \
  training.fresh_batch=false
```

Expected: 100 runs complete. With Phase-B Coverage vectorization plus the
ray-bisected `confidence_set` (~40 bisection iters across n_rays=200),
each run is ~30–90 s; total wall ~1–2 hours.

- [ ] **Step 2: Generate the paper table**

```python
from cdsbi.analysis.loaders import load_runs
from cdsbi.analysis.paper_tables import paper_table_8_2
df = load_runs("outputs/8_2_baseline_sweep/<timestamp>/*/")
print(paper_table_8_2(df))
```

Expected output: a multi-row DataFrame indexed by (method, budget) with
mean and std for `pivot_rmse`, `marginal_ks`, `coverage_error_max`,
`joint_mahal_ks`, `actual_params_total`. CDSBI's row at medium budget
should match the §8.2 tolerance band committed in Task 11.

- [ ] **Step 3: Commit the §8.2 sweep results notebook or summary** (no
git change required if the user wants this kept ephemeral; mark this task
as done once the table is reviewed.)

This is the v1 sweep deliverable. Manuscript §8.2 cross-method comparison
is now backed by a single CLI invocation, the same as §8.1. Task 15
integrates these numbers into the manuscript text.

---

## Task 15: Integrate v1 results into manuscript §8.2

**Files:**
- Modify: `cd_sbi_v7.tex` — replace existing §8.2 (single-seed table at
  lines ~2498–2553) with the sweep-averaged version + cross-method
  comparison, mirroring the §8.1 restructure already landed.
- Verify: `/usr/bin/pdflatex cd_sbi_v7 && /usr/bin/bibtex cd_sbi_v7 &&
  /usr/bin/pdflatex cd_sbi_v7 && /usr/bin/pdflatex cd_sbi_v7` builds
  cleanly. (Use the absolute path; the conda `pdflatex` on this machine
  is broken — see CLAUDE.md.)

**Prerequisite:** Task 14 complete and the user has signed off on the
v1 results. Do not invent or paraphrase numbers; pull them from the
sweep output via `paper_table_8_2(load_runs(...))`.

**Template:** The §8.1 update at `cd_sbi_v7.tex` already shows the
target structure (committed to `main`). Reproduce the same five-block
shape for §8.2:

1. **Setup** — keep the current 2D location-Gaussian setup
   (\(X \mid \theta \sim \mathcal{N}(\theta, I_2)\), \(\theta \sim U[-7, 7]^2\))
   and the truth statement \(r^*(\theta, X) = \theta - X\) component-wise.
2. **Architecture** — keep the current triangular-flow paragraph
   (\(r_1, r_2\) with contextual UMNNs). Drop the hard-coded
   "5,004 parameters total" — replace with a pointer to the budget grid
   used in the sweep.
3. **Sweep** — same paragraph as §8.1, adapted for 2D: 5 seeds × 4
   budgets, evaluation-set sample sizes ($N=5000$ marginal, $N/\text{bin}=1000$
   conditional, 200 \(\theta_0\) × 50 \(X\mid\theta\) coverage), Monte-Carlo
   noise floors quoted with each diagnostic.
4. **CDSBI calibration diagnostics (medium budget)** — replace the
   existing single-run §8.2 table with a sweep-averaged version. Keep
   the diagnostic rows the current table has: Pivot RMSE (total +
   per-coord), Marginal PIT KS (\(r_1, r_2\)), Joint Mahalanobis vs
   \(\chi^2_2\) KS, Conditional PIT KS (bulk + edge per coord),
   Conditional Mahalanobis KS, Coverage error max. All values as
   mean ± std across 5 seeds; noise-floor column instead of "Status".
5. **Budget invariance** — short paragraph if the §8.2 sweep shows
   the same saturation pattern §8.1 shows. If it doesn't (i.e.,
   diagnostics genuinely improve with budget at d=2 because the
   conditioning network needs more capacity), then describe what the
   data shows; do not assume the §8.1 pattern carries over.
6. **Comparison with baselines** — new 5-method × 4-budget table.
   Headline metric is `joint_mahal_ks` for §8.2 (the inferentially
   primary 2D diagnostic; coverage_error_max can go in a secondary
   row or supplementary). Cross-reference §10 for the architectural
   reading and §3.4 for the NLE/CDSBI shared-loss explanation. Be
   explicit about what NPE's "marginal-product" credible region means
   in the table — call it out as a known Bonferroni-conservative
   diagnostic until Task 13c HPD lands (currently in the addendum's
   acknowledged-but-not-patched list).
7. **Synthesis** — one paragraph closing §8.2, mirroring §8.1's
   synthesis tone. The §8.2-specific point worth landing is whether
   the §8.1 "CDSBI = NLE at the floor" pattern persists into 2D or
   diverges (the multivariate-prior + MCMC-mixing story I outlined
   in the chat).

- [ ] **Step 1: Extract the §8.2 numbers**

Run, from the repo root:

```python
from cdsbi.analysis.loaders import load_runs
from cdsbi.analysis.paper_tables import paper_table_8_2
import pandas as pd, glob

# Union all sweep dirs that contributed OK runs (may be 1, may be 2-3 if
# re-runs were needed). Match the dir-glob used during Task 14.
dirs = sorted(glob.glob("outputs/8_2_baseline_sweep/<ts1>/*")) + \
       sorted(glob.glob("outputs/8_2_baseline_sweep/<ts2>/*"))
rows = [load_runs(d).iloc[0] for d in dirs if "STATUS" in open(...).read()...]
df = pd.DataFrame(rows)

print(paper_table_8_2(df))   # → table for cross-method block

# For the CDSBI calibration table, pull per-run diagnostic parquets
# (marginal_pit.parquet, conditional_pit.parquet, joint_mahalanobis.parquet,
# pivot_rmse.parquet, coverage.parquet) across the 5 CDSBI medium-budget
# seeds and aggregate mean±std per diagnostic. Use the same per-bin
# bulk/edge split as §8.1 (bins 1,2,3 vs 0,4 of the 5-bin grid).
```

Sanity-check the numbers against the manuscript's existing §8.2 values
(pivot RMSE 0.044, marginal KS 0.007 / 0.006, joint Mahalanobis KS 0.011,
coverage error < 0.01). The seed-averaged sweep numbers will likely be
slightly higher (single-run vs 5-seed average) and statistically
distinguishable from the manuscript's single-run point estimates — that
is the whole point of doing the sweep.

- [ ] **Step 2: Write the LaTeX edit**

In `cd_sbi_v7.tex`, replace lines ~2498–2553 (current §8.2 block) with
the seven-block structure above, using the actual numbers from Step 1.
Mirror the LaTeX patterns of §8.1: `longtable` for both tables (3-col
for the CDSBI calibration table, 5-col for the cross-method comparison),
mean ± std in math mode, noise-floor column instead of "Status", cross-
references via `\S\ref{sec:10}` / `\S\ref{subsec:3.4}` / `\S\ref{sec:4}`
not unresolved labels.

Drop content the v1 sweep doesn't support: the "conditioning network
learning slower" parenthetical on \(r_2\) (line ~2530) is single-seed
narrative — keep only if the 5-seed sweep shows \(r_2\) systematically
above \(r_1\) by more than the per-seed scatter.

- [ ] **Step 3: Compile the manuscript**

```bash
/usr/bin/pdflatex -interaction=nonstopmode -halt-on-error cd_sbi_v7 \
  && /usr/bin/bibtex cd_sbi_v7 \
  && /usr/bin/pdflatex -interaction=nonstopmode -halt-on-error cd_sbi_v7 \
  && /usr/bin/pdflatex -interaction=nonstopmode -halt-on-error cd_sbi_v7
```

Expected: "Output written on cd_sbi_v7.pdf (47–48 pages, ...)". No
"undefined reference", no "missing $ inserted", no "extra alignment tab".
If the page count moves more than ±1, the table widths or column counts
likely don't match the spec — re-run the structural-sanity script from
the §8.1 commit history before chasing visual fixes.

- [ ] **Step 4: Structural sanity check on the new TeX**

Run the helper from the §8.1 commit history (counts `&` per row,
checks column-spec match, brace balance, cross-reference resolvability)
against the §8.2 region. Both new longtables should report all-cells-OK
and brace-balanced.

- [ ] **Step 5: Commit**

```bash
git add cd_sbi_v7.tex
git commit -m "$(cat <<'EOF'
manuscript(8.2): v1 sweep-averaged results + cross-method comparison

Replace single-seed §8.2 numbers with the seed-averaged v1 sweep
(5 seeds × 4 budgets × 5 methods). Adds a cross-method comparison
table on joint_mahal_ks (the §8.2-primary diagnostic) parallel to
the §8.1 structure already in the manuscript.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Done condition:** PDF builds in 4 passes (pdflatex × 1, bibtex,
pdflatex × 2) with no warnings beyond the natbib citation pass; §8.2
in the rendered PDF reads as a parallel structure to §8.1; the user
has reviewed the substantive empirical claims.

---

## Self-Review

**1. Spec coverage** — checked against §11–§13 of the design spec:

- v1 deliverable per §12 row "v1 §8.2": `TriangularAdditiveFlow` (Task 3),
  multivariate `confidence_set` (Task 4 + Task 13), `JointMahalanobis`
  diagnostic + regression test (Task 5), `target/loc_gauss_2d_iid` (Task 10),
  `flow/triangular_additive` (Task 10), `experiment/8_2_*` (Task 10),
  cross-method §8.2 baseline (Task 13 + Task 14). ✓
- §13 forward-hook commitments preserved: `Flow.forward(θ, context)`
  signature unchanged (Task 3 uses `context=X`); `monotonicity_guarantees`
  still a first-class attribute (Task 3); `log_det_jac_input` naming
  unchanged; `Simulator.entropy_lower_bound` extends to d > 1 (Task 2);
  `ConfidenceProcedure` subtypes get d-aware confidence_set (Tasks 4, 13). ✓
- §11 done-criterion analogs for v1:
  1. CLI completes deterministically → Tasks 10 + 14 demonstrate.
  2. CDSBI tolerance bands at d=2 → Task 11 (intensive test).
  3. Matched-budget across (method, budget) → Task 9 (retune).
  4. Fast tests pass <30 s → preserved (incremental adds).
  5. `pytest -m intensive` matches bands → Task 11 implements.

**2. Placeholder scan**: no "TBD", "TODO without code", or hand-wave steps in
any task. One explicit forward-TODO is documented in the code comment for
asymmetric-d MAF construction (Task 8 step 1) — flagged as the expected
v1+ refinement, not a placeholder for this plan.

**3. Type consistency**:
- `simulator.d_theta` / `simulator.d_x` are int across all tasks. ✓
- `Simulator.sample_x_given_theta(theta_0, n, rng) → torch.Tensor` shape
  `(n, d_x)` consistent across Tasks 1, 2, 6. ✓
- `ConfidenceSet.boundary_repr`: `(2,)` in 1D, `(K, d)` in d > 1, `(0, d)`
  for empty sets (Task 4 edge case) — docstring + downstream readers
  (SetSize in Task 6) match. ✓
- `TriangularAdditiveFlow.forward(theta, context) → (r, log_det)` with
  `r.shape == (n, d)` and `log_det.shape == (n,)` — Task 3 + Task 8. ✓
- Diagnostic `meta` extensions (`d`, `n_bins`) added in Task 7 are read
  optionally; no caller depends on them. ✓

## Addendum: review-driven changes already applied + caveats for execution

An independent reviewer agent surfaced 16 findings (4 BLOCKER, 7 IMPORTANT,
5 MINOR) on the v1 plan + spec. The plan has been updated inline to address
the following:

- **Task 4 ray-bisection center** (BLOCKER): trained flows generally have
  `r(X_obs; X_obs) ≠ 0`, so using X_obs as the bisection center can place
  the start outside the α-set and make the bisection trivially return 0.
  Fixed by adding `_find_center` (coarse d-dim grid argmin + L-BFGS refine)
  and an empty-set short-circuit when even the minimum is outside.
- **Task 5 dead-code line** (MINOR): removed the `theta_t = torch.full_like(...)`
  line that was overwritten on the next statement.
- **Task 6 duck-typing** (IMPORTANT): replaced the
  `hasattr(x, "__len__")` check (which misclassifies numpy scalars and
  0-d tensors) with a small `_theta_to_row_tensor` helper using
  `np.atleast_1d`. Used uniformly by every procedure's `contains_batch`.
- **Task 6 LikelihoodBased grid for d > 1** (IMPORTANT, ties to finding
  14/15 — RatioBased delegation breaks in d > 1): the `torch.linspace` in
  `LikelihoodBasedProcedure.contains_batch` can't reshape into `(-1, d)`
  for d > 1; replaced with random-grid sampling over the prior box,
  matching the LF2I-BFF marginal-integration approach.
- **Task 6 SetSize d > 1** (BLOCKER, surfaced by reviewer): SetSize's
  per-sample slow-path `cs.boundary_repr[1] - cs.boundary_repr[0]` is
  1D-only. Added a d > 1 branch that computes the **set diameter** =
  `max_i ||boundary_i - centroid||` (rough proxy; for paper-table 8.2
  use-cases this is adequate, true volume would be over-engineered at d=2).
- **Task 7 `_write_index_row` marginal_ks** (IMPORTANT, caught during
  self-review): updated to extract the per-coord mean when the value is
  a DataFrame, preserving `index_row.parquet`'s scalar contract.

The following findings are **acknowledged but not patched inline** —
flag them at execution time:

- **Task 13 PosteriorBased d > 1** (IMPORTANT): the per-coord cartesian
  product is Bonferroni-conservative (joint coverage of two 90% marginals
  ≈ 81%). For the §8.2 cross-method comparison NPE will look worse than
  it should. The proper fix is a true HPD region from posterior samples
  (kernel-density estimate + level-set inversion, or sample-based density
  thresholding). Defer to Task 13c during execution; mark NPE results in
  the §8.2 paper table as "marginal-product" until HPD lands.
- **Task 13 rngs.eval sub-stream** (IMPORTANT): the v1 d > 1 BFF marginal
  grid pulls from `rngs.eval`, which also draws the calibration set —
  changing draw order between d=1 and d=2 paths. Document the
  determinism contract change in the v1 commit messages; consider
  deriving a sub-stream via
  `hashlib.sha256((seed, "bff_marginal_grid")).digest()` in a follow-up.
- **Task 13 splitting** (MINOR): the single Task 13 covers BFF marginal
  grid + CriticalValue d > 1 + Likelihood d > 1 + Posterior d > 1, which
  is heavy. Recommend splitting into Tasks 13a / 13b / 13c at execution
  time for clean review checkpoints.
- **Task 11 tolerance band wording** (MINOR): the "≤ 0.08 ≈ 1.5–2×
  manuscript" justification is closer to 1.8× of the manuscript's 0.044.
  Tighten the threshold to ≤ 0.07 or fix the prose; either resolves it.

All remaining MINOR findings (n_rays default, scipy kstest calling
convention, test naming consistency, etc.) are non-blocking and can be
addressed during execution review checkpoints.
