# CD-SBI (μ, σ²) — M0: Stage-A Core (target + oracle + non-additive flow)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage-A core for the unknown-(μ, σ²) Gaussian target: the simulator (with closed-form joint pivot), the oracle sufficient-statistic conditioner, the non-additive autoregressive `TriangularDoublyMonotoneFlow`, wire them into the run CLI, and confirm a single CDSBI run recovers the closed-form pivot.

**Architecture:** New simulator `NormalUnknownMeanVar` (θ=(log σ, μ), n_iid=10, exact `r_star`). New oracle `SufficientStatConditioner` (X→(s², X̄), frozen, constant log-det). New flow `TriangularDoublyMonotoneFlow(d)` — autoregressive/triangular, each coordinate doubly-monotone in (θ_k, feat_k) via Gauss–Legendre quadrature and conditioned on (θ_{<k}, feat_{<k}); lower-triangular feature-Jacobian gives `log_det = Σ_k log|∂r_k/∂feat_k|` in closed form. Generalizes both `TriangularAdditiveFlow` (additive special case) and `DoublyMonotoneUMNN` (d=1 special case).

**Tech Stack:** PyTorch (Gauss–Legendre quadrature, closed-form Jacobians), Hydra, pytest, numpy/scipy (closed-form pivot via χ²/Φ).

**Spec:** `docs/superpowers/specs/2026-05-29-cd-sbi-unknown-mean-variance-design.md` (Stage A / M0).

---

## Conventions locked for M0

- **Parameter order `θ = (log σ, μ)`** (index 0 = log σ, index 1 = μ) — the forced KR order. **Feature order `(s², X̄)`** paired to it.
- **Closed-form truth** (n = n_iid): `r*_σ = Φ⁻¹(F_{χ²_{n−1}}((n−1)s²/σ²))`, `r*_μ = √n(X̄−μ)/σ`, with `σ = exp(log σ)`. `(r*_σ, r*_μ) ~ 𝒩(0, I₂)` at θ₀.
- **Flow `forward(theta, context)`** signature (matches `Flow` protocol): `context` is the conditioner output (the features), shape `(n, d)`; returns `(r, log_det_jac_input)` with `r` shape `(n, d)`, `log_det` shape `(n,)`.
- **Conditioner `encode(X) → (features, log_det_contrib)`** (matches `Conditioner` protocol); `log_det_contrib` shape `(n,)`.

---

## File structure

```
NEW
  src/cdsbi/simulators/normal_unknown_mean_var.py   # NormalUnknownMeanVar
  src/cdsbi/conditioners/sufficient_stat.py          # SufficientStatConditioner (oracle)
  src/cdsbi/flows/triangular_doubly_monotone.py      # _CondMonotoneScalarUMNN + TriangularDoublyMonotoneFlow
  configs/target/normal_mu_sigma.yaml
  configs/flow/triangular_doubly_monotone.yaml
  configs/conditioner/sufficient_stat.yaml
  configs/experiment/mu_sigma_replication.yaml
  tests/unit/test_normal_unknown_mean_var.py
  tests/unit/test_sufficient_stat_conditioner.py
  tests/unit/test_triangular_doubly_monotone_flow.py
  tests/integration/test_mu_sigma_smoke.py

MODIFY
  src/cdsbi/experiments/run.py    # _build_flow: triangular_doubly_monotone; conditioner build for the new target
```

No changes to existing simulators/flows/conditioners. `CDSBIRunner.fit`'s conditioner-param training is **M2** (the oracle conditioner here is frozen/zero-param, so M0 needs no fit change).

---

## Task 1: `NormalUnknownMeanVar` simulator — sampling

**Files:**
- Create: `src/cdsbi/simulators/normal_unknown_mean_var.py`
- Test: `tests/unit/test_normal_unknown_mean_var.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_normal_unknown_mean_var.py`:

```python
"""NormalUnknownMeanVar: X = (X_1..X_n_iid) iid N(μ, σ²); θ = (log σ, μ)."""
from __future__ import annotations

import numpy as np
import torch


def test_shapes_and_priors():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    sim = NormalUnknownMeanVar()
    assert sim.d_theta == 2 and sim.d_x == 10 and sim.n_iid == 10
    rng = np.random.default_rng(0)
    theta, x = sim.sample(20000, rng)
    assert theta.shape == (20000, 2) and x.shape == (20000, 10)
    log_sigma, mu = theta[:, 0], theta[:, 1]
    # priors: log σ ∈ [log 0.3, log 3], μ ∈ [-5, 5]
    assert float(log_sigma.min()) >= np.log(0.3) - 1e-4
    assert float(log_sigma.max()) <= np.log(3.0) + 1e-4
    assert float(mu.min()) >= -5.0 - 1e-3 and float(mu.max()) <= 5.0 + 1e-3


def test_conditional_moments():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    sim = NormalUnknownMeanVar()
    rng = np.random.default_rng(1)
    # θ0 = (log σ = log 2, μ = 1.0) → σ = 2
    x = sim.sample_x_given_theta([np.log(2.0), 1.0], 50000, rng)
    assert x.shape == (50000, 10)
    assert abs(float(x.mean()) - 1.0) < 0.02            # E[X] = μ
    assert abs(float(x.std()) - 2.0) < 0.03             # sd = σ
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_normal_unknown_mean_var.py -v`
Expected: FAIL — `ModuleNotFoundError: ...normal_unknown_mean_var`.

- [ ] **Step 3: Write the implementation (sampling only — `r_star`/`log_prob`/`entropy` in Task 2)**

`src/cdsbi/simulators/normal_unknown_mean_var.py`:

```python
"""NormalUnknownMeanVar: X = (X_1, …, X_{n_iid}) iid N(μ, σ²).

θ = (log σ, μ) — the forced KR order (scale first). The first target with a
scale/nuisance parameter. Exact closed-form joint pivot (Basu independence of
X̄ and s²):
    r*_σ(θ, X) = Φ⁻¹(F_{χ²_{n-1}}((n-1) s² / σ²))     # uses (σ², s²)
    r*_μ(θ, X) = √n (X̄ − μ) / σ                        # uses (μ, σ; X̄)
with σ = exp(log σ); (r*_σ, r*_μ) ~ N(0, I₂) at the true θ₀.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.stats import chi2, norm


@dataclass
class NormalUnknownMeanVar:
    n_iid: int = 10
    mu_range: Tuple[float, float] = (-5.0, 5.0)
    log_sigma_range: Tuple[float, float] = (math.log(0.3), math.log(3.0))
    d_theta: int = 2

    @property
    def d_x(self) -> int:
        return self.n_iid

    def _draw_theta(self, n: int, rng: np.random.Generator) -> np.ndarray:
        log_sigma = rng.uniform(*self.log_sigma_range, size=(n, 1))
        mu = rng.uniform(*self.mu_range, size=(n, 1))
        return np.concatenate([log_sigma, mu], axis=1)  # (n, 2): [log σ, μ]

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_np = self._draw_theta(n, rng)
        sigma = np.exp(theta_np[:, 0:1])           # (n, 1)
        mu = theta_np[:, 1:2]                       # (n, 1)
        x_np = rng.normal(loc=mu, scale=sigma, size=(n, self.n_iid))
        return torch.from_numpy(theta_np).float(), torch.from_numpy(x_np).float()

    def sample_x_given_theta(self, theta_0, n: int, rng: np.random.Generator) -> torch.Tensor:
        theta_vec = np.atleast_1d(np.asarray(theta_0, dtype=np.float64))
        assert theta_vec.shape == (self.d_theta,), (
            f"theta_0 has shape {theta_vec.shape}, expected ({self.d_theta},)"
        )
        sigma = float(np.exp(theta_vec[0]))
        mu = float(theta_vec[1])
        x_np = rng.normal(loc=mu, scale=sigma, size=(n, self.n_iid))
        return torch.from_numpy(x_np).float()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_normal_unknown_mean_var.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/simulators/normal_unknown_mean_var.py tests/unit/test_normal_unknown_mean_var.py
git commit -m "feat(sim): NormalUnknownMeanVar sampling (θ=(log σ, μ), n_iid=10)"
```

---

## Task 2: simulator `r_star` + `log_prob` + `entropy_lower_bound`

**Files:**
- Modify: `src/cdsbi/simulators/normal_unknown_mean_var.py`
- Test: `tests/unit/test_normal_unknown_mean_var.py` (append)

`r_star` accepts raw X `(n, n_iid)`, computes `(X̄, s²)` internally, returns `(n, 2)` in order `(r_σ, r_μ)`. The key validation: `r*(θ₀, X)` for `X ~ p(·|θ₀)` is `~ 𝒩(0, I₂)`.

- [ ] **Step 1: Append the failing tests**

```python
def test_r_star_is_standard_normal_at_truth():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    sim = NormalUnknownMeanVar()
    rng = np.random.default_rng(2)
    theta0 = [np.log(1.5), 0.5]                  # σ = 1.5, μ = 0.5
    x = sim.sample_x_given_theta(theta0, 40000, rng)
    theta = torch.tensor(theta0, dtype=torch.float32).expand(40000, 2)
    r = sim.r_star(theta, x)
    assert r.shape == (40000, 2)
    # each coordinate ~ N(0,1); jointly independent → mean≈0, std≈1, corr≈0
    r_np = r.numpy()
    assert np.allclose(r_np.mean(axis=0), 0.0, atol=0.03)
    assert np.allclose(r_np.std(axis=0), 1.0, atol=0.03)
    assert abs(float(np.corrcoef(r_np.T)[0, 1])) < 0.03


def test_r_star_monotonicity_signs():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    sim = NormalUnknownMeanVar()
    rng = np.random.default_rng(3)
    x = sim.sample_x_given_theta([0.0, 0.0], 8, rng)         # one X (8 used loosely)
    x = sim.sample_x_given_theta([0.0, 0.0], 1, rng)         # shape (1, 10)
    base = torch.tensor([[0.0, 0.0]])
    r0 = sim.r_star(base, x)
    # r_μ ↓ in μ (index 1): increasing μ lowers r_μ
    r_mu_up = sim.r_star(torch.tensor([[0.0, 0.5]]), x)
    assert float(r_mu_up[0, 1]) < float(r0[0, 1])
    # r_σ ↓ in log σ (index 0): increasing log σ lowers r_σ
    r_sig_up = sim.r_star(torch.tensor([[0.5, 0.0]]), x)
    assert float(r_sig_up[0, 0]) < float(r0[0, 0])


def test_entropy_lower_bound_runs():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    val = NormalUnknownMeanVar().entropy_lower_bound(n_mc=20000)
    assert np.isfinite(val)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_normal_unknown_mean_var.py -v`
Expected: FAIL — `AttributeError: 'NormalUnknownMeanVar' object has no attribute 'r_star'`.

- [ ] **Step 3: Add `r_star`, `log_prob`, `entropy_lower_bound`**

Append these methods to the dataclass (after `sample_x_given_theta`). Note `register_buffer`-style state isn't needed (pure functions); they convert to numpy for the χ²/Φ closed forms (only used by diagnostics, never inside autograd):

```python
    def _suff_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(X̄, s²) with s² the unbiased sample variance (ddof=1) over the n_iid axis."""
        xbar = x.mean(dim=-1, keepdim=True)                       # (n, 1)
        s2 = x.var(dim=-1, unbiased=True, keepdim=True)           # (n, 1), ddof=1
        return xbar, s2

    def r_star(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Closed-form joint pivot (r_σ, r_μ), shape (n, 2). θ = (log σ, μ)."""
        xbar, s2 = self._suff_stats(x)
        log_sigma = theta[:, 0:1]
        mu = theta[:, 1:2]
        sigma = torch.exp(log_sigma)
        n = self.n_iid
        # r_σ via χ²_{n-1}: w = (n-1) s² / σ²
        w = ((n - 1) * s2 / (sigma ** 2)).detach().cpu().numpy()
        u = chi2.cdf(w, df=n - 1)
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        r_sigma = torch.from_numpy(norm.ppf(u)).float().to(theta.device)  # (n, 1)
        # r_μ = √n (X̄ − μ) / σ  (closed form, differentiable)
        r_mu = math.sqrt(n) * (xbar - mu) / sigma                          # (n, 1)
        return torch.cat([r_sigma, r_mu], dim=-1)                          # (n, 2)

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Σ_i log N(X_i; μ, σ²)."""
        log_sigma = theta[:, 0:1]
        mu = theta[:, 1:2]
        sigma = torch.exp(log_sigma)
        z = (x - mu) / sigma
        per_obs = -0.5 * z ** 2 - log_sigma - 0.5 * math.log(2 * math.pi)
        return per_obs.sum(dim=-1)

    def entropy_lower_bound(self, n_mc: int = 50000, seed: int = 42) -> float:
        """MC estimate of E[NF-MLE loss at r*] on the (θ, (X̄,s²)) scale.

        The trainer's loss with the SufficientStatConditioner is
            L = ½‖r‖² + d log(2π)/2 − log|∂r/∂features| − log_det_contrib.
        At r = r*, E[L] is the conditional-entropy floor any valid surrogate
        must satisfy (Theorem 3.2). We evaluate the feature-Jacobian of r* in
        closed form:
            ∂r_σ/∂s² = (n-1)/σ² · f_{χ²_{n-1}}(w) / φ(r_σ)
            ∂r_μ/∂X̄ = √n / σ
        (the feature-Jacobian is lower-triangular ⇒ log|det| = sum of these
        diagonal terms). log_det_contrib is the conditioner's constant
        (Task 3); we use the SAME constant here so the floor is comparable.
        """
        rng = np.random.default_rng(seed)
        theta_np = self._draw_theta(n_mc, rng)
        sigma = np.exp(theta_np[:, 0:1]); mu = theta_np[:, 1:2]
        x = rng.normal(loc=mu, scale=sigma, size=(n_mc, self.n_iid))
        xbar = x.mean(axis=-1, keepdims=True)
        s2 = x.var(axis=-1, ddof=1, keepdims=True)
        n = self.n_iid
        w = (n - 1) * s2 / sigma ** 2
        u = np.clip(chi2.cdf(w, df=n - 1), 1e-12, 1 - 1e-12)
        r_sigma = norm.ppf(u)
        r_mu = math.sqrt(n) * (xbar - mu) / sigma
        # feature-Jacobian diagonal terms
        dr_sigma_ds2 = (n - 1) / sigma ** 2 * chi2.pdf(w, df=n - 1) / np.clip(norm.pdf(r_sigma), 1e-30, None)
        dr_mu_dxbar = math.sqrt(n) / sigma
        log_det_feat = np.log(np.clip(dr_sigma_ds2, 1e-30, None)) + np.log(dr_mu_dxbar)
        log_det_contrib = _SUFFICIENT_STAT_LOG_DET_CONST  # shared module constant (Task 3)
        loss = (0.5 * (r_sigma ** 2 + r_mu ** 2) + math.log(2 * math.pi)
                - log_det_feat - log_det_contrib)
        return float(loss.mean())
```

And add the shared constant near the top of the module (used by both the simulator's floor and the conditioner in Task 3 — define it here, import it there):

```python
# θ-independent volume constant for the (X̄, s²) sufficient-statistic reduction.
# By sufficiency p(X | X̄, s²) is θ-free, so this only shifts the loss scale,
# not the argmin. Set to 0.0 (the reduction's constant is folded into the
# entropy-floor reference, not the optimization). Kept as a named constant so
# the floor calc (here) and the conditioner (Task 3) stay in lockstep.
_SUFFICIENT_STAT_LOG_DET_CONST = 0.0
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/unit/test_normal_unknown_mean_var.py -v`
Expected: PASS (5 passed). The `test_r_star_is_standard_normal_at_truth` is the load-bearing one — confirms the exact pivot.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/simulators/normal_unknown_mean_var.py tests/unit/test_normal_unknown_mean_var.py
git commit -m "feat(sim): NormalUnknownMeanVar closed-form r* + log_prob + entropy floor"
```

---

## Task 3: `SufficientStatConditioner` (oracle X → (s², X̄))

**Files:**
- Create: `src/cdsbi/conditioners/sufficient_stat.py`
- Test: `tests/unit/test_sufficient_stat_conditioner.py`

Outputs features in **(s², X̄)** order (index 0 = s² paired to log σ, index 1 = X̄ paired to μ). Frozen, zero-parameter; `log_det_contrib` is the shared θ-independent constant.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_sufficient_stat_conditioner.py`:

```python
"""SufficientStatConditioner: oracle X → (s², X̄), frozen, constant log-det."""
from __future__ import annotations

import torch


def test_encode_features_and_order():
    from cdsbi.conditioners.sufficient_stat import SufficientStatConditioner
    cond = SufficientStatConditioner(n_iid=10)
    # X with known mean/var per row
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0]])  # mean=2, s²=ddof1
    feats, log_det = cond.encode(x)
    assert feats.shape == (1, 2)
    s2_expected = float(x.var(dim=-1, unbiased=True))
    assert abs(float(feats[0, 0]) - s2_expected) < 1e-5     # index 0 = s²
    assert abs(float(feats[0, 1]) - 2.0) < 1e-6             # index 1 = X̄
    assert log_det.shape == (1,)


def test_zero_params():
    from cdsbi.conditioners.sufficient_stat import SufficientStatConditioner
    assert SufficientStatConditioner(n_iid=10).n_params() == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_sufficient_stat_conditioner.py -v`
Expected: FAIL — `ModuleNotFoundError: ...sufficient_stat`.

- [ ] **Step 3: Write the implementation**

`src/cdsbi/conditioners/sufficient_stat.py`:

```python
"""SufficientStatConditioner: oracle reduction X → (s², X̄) for the (μ, σ²) target.

Features are ordered (s², X̄) to pair with θ = (log σ, μ) in the autoregressive
flow. Zero-parameter (the oracle), θ-independent constant log-det (sufficiency).
"""
from __future__ import annotations

from typing import Tuple

import torch

from cdsbi.simulators.normal_unknown_mean_var import _SUFFICIENT_STAT_LOG_DET_CONST


class SufficientStatConditioner:
    def __init__(self, n_iid: int):
        self.n_iid = n_iid

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xbar = x.mean(dim=-1, keepdim=True)                       # (n, 1)
        s2 = x.var(dim=-1, unbiased=True, keepdim=True)           # (n, 1)
        feats = torch.cat([s2, xbar], dim=-1)                     # (n, 2): [s², X̄]
        log_det = torch.full(
            (x.shape[0],), _SUFFICIENT_STAT_LOG_DET_CONST, dtype=x.dtype, device=x.device,
        )
        return feats, log_det

    def n_params(self) -> int:
        return 0
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/unit/test_sufficient_stat_conditioner.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/conditioners/sufficient_stat.py tests/unit/test_sufficient_stat_conditioner.py
git commit -m "feat(cond): SufficientStatConditioner oracle X→(s², X̄)"
```

---

## Task 4: `_CondMonotoneScalarUMNN` (context-conditioned monotone scalar)

**Files:**
- Create: `src/cdsbi/flows/triangular_doubly_monotone.py`
- Test: `tests/unit/test_triangular_doubly_monotone_flow.py`

The building block: a monotone-increasing-in-`z` scalar map `f(z; ctx) = bias(ctx) + ∫_{z_ref}^{z} softplus(MLP([t, ctx])) dt`, with `derivative(z; ctx) = softplus(MLP([z, ctx]))`. Generalizes `doubly_monotone._MonotoneScalarUMNN` by conditioning on a context vector (empty `ctx` ⇒ the unconditioned case).

- [ ] **Step 1: Write the failing test**

`tests/unit/test_triangular_doubly_monotone_flow.py`:

```python
"""TriangularDoublyMonotoneFlow + its conditioned monotone-scalar building block."""
from __future__ import annotations

import torch


def test_cond_monotone_scalar_increasing_and_derivative_matches_autograd():
    from cdsbi.flows.triangular_doubly_monotone import _CondMonotoneScalarUMNN
    torch.manual_seed(0)
    f = _CondMonotoneScalarUMNN(context_dim=3, hidden=16, depth=2, z_ref=0.0)
    n = 64
    ctx = torch.randn(n, 3)
    z = torch.linspace(-2, 2, n).unsqueeze(-1)
    val = f(z, ctx)
    assert val.shape == (n, 1)
    # strictly increasing in z (compare sorted z)
    z_sorted, idx = torch.sort(z.squeeze(-1))
    v_sorted = f(z_sorted.unsqueeze(-1), ctx[idx])
    # not a valid monotonicity check across different ctx; instead test at fixed ctx:
    ctx0 = torch.randn(1, 3).expand(n, 3)
    zz = torch.linspace(-3, 3, n).unsqueeze(-1)
    vv = f(zz, ctx0).squeeze(-1)
    assert torch.all(vv[1:] - vv[:-1] > 0)               # increasing in z at fixed ctx
    # analytic derivative matches autograd
    z_req = torch.tensor([[0.7]], requires_grad=True)
    c1 = torch.randn(1, 3)
    out = f(z_req, c1)
    (grad,) = torch.autograd.grad(out.sum(), z_req)
    assert torch.allclose(grad, f.derivative(z_req, c1), atol=1e-4)


def test_cond_monotone_scalar_empty_context():
    from cdsbi.flows.triangular_doubly_monotone import _CondMonotoneScalarUMNN
    f = _CondMonotoneScalarUMNN(context_dim=0, hidden=8, depth=2, z_ref=0.0)
    z = torch.linspace(-2, 2, 32).unsqueeze(-1)
    v = f(z, None).squeeze(-1)
    assert torch.all(v[1:] - v[:-1] > 0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_triangular_doubly_monotone_flow.py::test_cond_monotone_scalar_empty_context -v`
Expected: FAIL — `ModuleNotFoundError: ...triangular_doubly_monotone`.

- [ ] **Step 3: Write the building block**

`src/cdsbi/flows/triangular_doubly_monotone.py` (this file grows over Tasks 4–5; start it now):

```python
"""TriangularDoublyMonotoneFlow — non-additive autoregressive doubly-monotone flow.

Coordinate k: r_k = b_k(feat_k; ctx_k) + ∫_{θ_ref}^{θ_k} softplus(α_k(t; ctx_k)
+ β_k(feat_k; ctx_k)) dt,  with ctx_k = (θ_{<k}, feat_{<k}). Each coordinate is
R1 (∂r_k/∂θ_k = softplus(·) > 0) and R2 (∂r_k/∂feat_k > 0) by construction. The
feature-Jacobian is lower-triangular (r_k depends only on feat_{≤k}), so
log|det ∂r/∂feat| = Σ_k log(∂r_k/∂feat_k), computed in closed form by the same
quadrature as the forward pass.

Generalizes TriangularAdditiveFlow (additive special case) and
DoublyMonotoneUMNN (d=1 special case).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cdsbi.flows.base import Flow, Guarantee

_NODES_NP, _WEIGHTS_NP = np.polynomial.legendre.leggauss(12)


def _tanh_mlp(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Sequential:
    assert depth >= 1
    layers: list = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class _CondMonotoneScalarUMNN(nn.Module):
    """Monotone-increasing-in-z scalar map conditioned on a context vector:
        f(z; ctx) = bias(ctx) + ∫_{z_ref}^{z} softplus(MLP([t, ctx])) dt
    derivative(z; ctx) = softplus(MLP([z, ctx])).  context_dim=0 ⇒ unconditioned.
    """

    def __init__(self, context_dim: int, hidden: int = 16, depth: int = 2, z_ref: float = 0.0):
        super().__init__()
        self.context_dim = context_dim
        self.z_ref = z_ref
        self.mlp = _tanh_mlp(in_dim=1 + context_dim, hidden=hidden, out_dim=1, depth=depth)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        if context_dim == 0:
            self.bias_param = nn.Parameter(torch.zeros(1))
            self.bias_net = None
        else:
            self.bias_param = None
            self.bias_net = nn.Linear(context_dim, 1)
        self.register_buffer("_nodes", torch.tensor(_NODES_NP, dtype=torch.float32))
        self.register_buffer("_weights", torch.tensor(_WEIGHTS_NP, dtype=torch.float32))

    def _bias(self, ctx: Optional[torch.Tensor], n: int, device, dtype) -> torch.Tensor:
        if self.context_dim == 0 or ctx is None:
            return self.bias_param.to(device=device, dtype=dtype).view(1, 1).expand(n, 1)
        return self.bias_net(ctx)                              # (n, 1)

    def _integrand(self, t: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        """softplus(MLP([t, ctx])) + 1e-3, strictly positive. t: (m, 1); ctx: (m, c) or None."""
        if self.context_dim == 0 or ctx is None:
            inp = t
        else:
            inp = torch.cat([t, ctx], dim=-1)
        return F.softplus(self.mlp(inp)) + 1e-3

    def derivative(self, z: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        """f'(z; ctx) = softplus(MLP([z, ctx])) + 1e-3. z: (n, 1)."""
        return self._integrand(z, ctx)

    def forward(self, z: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        n = z.shape[0]
        a = self.z_ref
        u = self._nodes.view(1, -1, 1).expand(n, -1, 1)         # (n, K, 1)
        z_exp = z.view(n, 1, 1).expand(-1, u.size(1), -1)
        t = a + 0.5 * (z_exp - a) * (u + 1.0)                   # (n, K, 1)
        K = u.size(1)
        if self.context_dim == 0 or ctx is None:
            ctx_rep = None
        else:
            ctx_rep = ctx.unsqueeze(1).expand(-1, K, -1).reshape(n * K, -1)
        integrand = self._integrand(t.reshape(n * K, 1), ctx_rep).view(n, K, 1)
        weights = self._weights.view(1, -1, 1)
        integral = 0.5 * (z - a) * (weights * integrand).sum(dim=1)   # (n, 1)
        return self._bias(ctx, n, z.device, z.dtype) + integral

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run to verify both block tests pass**

Run: `pytest tests/unit/test_triangular_doubly_monotone_flow.py -k cond_monotone -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/flows/triangular_doubly_monotone.py tests/unit/test_triangular_doubly_monotone_flow.py
git commit -m "feat(flows): _CondMonotoneScalarUMNN (context-conditioned monotone scalar)"
```

---

## Task 5: `TriangularDoublyMonotoneFlow` assembly

**Files:**
- Modify: `src/cdsbi/flows/triangular_doubly_monotone.py` (append the flow class)
- Test: `tests/unit/test_triangular_doubly_monotone_flow.py` (append)

Per coordinate `k`: `α_k` is a context-MLP of `(θ_k via the integrand t, ctx_k)`; `β_k` and `b_k` are `_CondMonotoneScalarUMNN`s of `feat_k` conditioned on `ctx_k`. `ctx_k = concat(θ_{<k}, feat_{<k})` (dim `2k`). Forward returns `r` (n, d) and `log_det = Σ_k log(∂r_k/∂feat_k)`.

- [ ] **Step 1: Append the failing tests**

```python
def test_flow_advertises_R1_R2_and_shapes():
    from cdsbi.flows.triangular_doubly_monotone import TriangularDoublyMonotoneFlow
    from cdsbi.flows.base import Guarantee
    flow = TriangularDoublyMonotoneFlow(d=2, hidden=16)
    assert flow.monotonicity_guarantees == frozenset({Guarantee.R1, Guarantee.R2})
    theta = torch.randn(32, 2)
    feats = torch.randn(32, 2)
    r, log_det = flow.forward(theta, context=feats)
    assert r.shape == (32, 2) and log_det.shape == (32,)


def test_flow_monotone_in_theta_per_coord():
    from cdsbi.flows.triangular_doubly_monotone import TriangularDoublyMonotoneFlow
    torch.manual_seed(1)
    flow = TriangularDoublyMonotoneFlow(d=2, hidden=16)
    feats = torch.randn(1, 2).expand(40, 2).contiguous()
    # sweep θ_1 (μ) at fixed θ_0, fixed feats → r_1 strictly increasing in θ_1
    th = torch.zeros(40, 2); th[:, 1] = torch.linspace(-3, 3, 40)
    r, _ = flow.forward(th, context=feats)
    assert torch.all(r[1:, 1] - r[:-1, 1] > 0)
    # sweep θ_0 (log σ) → r_0 strictly increasing in θ_0
    th = torch.zeros(40, 2); th[:, 0] = torch.linspace(-3, 3, 40)
    r, _ = flow.forward(th, context=feats)
    assert torch.all(r[1:, 0] - r[:-1, 0] > 0)


def test_flow_logdet_matches_autograd_feature_jacobian():
    from cdsbi.flows.triangular_doubly_monotone import TriangularDoublyMonotoneFlow
    torch.manual_seed(2)
    flow = TriangularDoublyMonotoneFlow(d=2, hidden=16)
    theta = torch.randn(1, 2)
    feats = torch.randn(1, 2, requires_grad=True)
    r, log_det = flow.forward(theta, context=feats)
    # lower-triangular feature-Jacobian: build it via autograd, take log|det|
    J = torch.zeros(2, 2)
    for i in range(2):
        (gi,) = torch.autograd.grad(r[0, i], feats, retain_graph=True)
        J[i] = gi[0]
    logdet_ref = torch.log(torch.abs(torch.det(J)))
    assert torch.allclose(log_det[0], logdet_ref, atol=1e-3)
    # confirm triangular: ∂r_0/∂feat_1 ≈ 0
    assert abs(float(J[0, 1])) < 1e-5


def test_flow_monotone_in_features_per_coord():
    from cdsbi.flows.triangular_doubly_monotone import TriangularDoublyMonotoneFlow
    torch.manual_seed(3)
    flow = TriangularDoublyMonotoneFlow(d=2, hidden=16)
    theta = torch.randn(1, 2).expand(40, 2).contiguous()
    # r_0 increasing in feat_0
    fe = torch.zeros(40, 2); fe[:, 0] = torch.linspace(-3, 3, 40)
    r, _ = flow.forward(theta, context=fe)
    assert torch.all(r[1:, 0] - r[:-1, 0] > 0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_triangular_doubly_monotone_flow.py -k flow -v`
Expected: FAIL — `AttributeError`/`ImportError` for `TriangularDoublyMonotoneFlow`.

- [ ] **Step 3: Append the flow class**

```python
class TriangularDoublyMonotoneFlow(nn.Module, Flow):
    monotonicity_guarantees = frozenset({Guarantee.R1, Guarantee.R2})

    def __init__(self, d: int, hidden: int = 16, depth: int = 2,
                 theta_ref: float = 0.0):
        super().__init__()
        if d < 1:
            raise ValueError(f"d must be ≥ 1, got {d}")
        self.d = d
        self.depth = depth
        self.theta_ref = theta_ref
        # Per-coordinate nets. ctx_k dim = 2k (θ_{<k}, feat_{<k}).
        self._alpha = nn.ModuleList()   # α_k: MLP([t, ctx_k]) → 1  (t is the θ_k integration var)
        self._beta = nn.ModuleList()    # β_k: monotone-increasing in feat_k, cond on ctx_k
        self._b = nn.ModuleList()       # b_k: monotone-increasing in feat_k, cond on ctx_k
        for k in range(d):
            ctx_dim = 2 * k
            self._alpha.append(_tanh_mlp(in_dim=1 + ctx_dim, hidden=hidden, out_dim=1, depth=depth))
            self._beta.append(_CondMonotoneScalarUMNN(context_dim=ctx_dim, hidden=hidden, depth=depth))
            self._b.append(_CondMonotoneScalarUMNN(context_dim=ctx_dim, hidden=hidden, depth=depth))
        self.register_buffer("_nodes", torch.tensor(_NODES_NP, dtype=torch.float32))
        self.register_buffer("_weights", torch.tensor(_WEIGHTS_NP, dtype=torch.float32))

    def _alpha_eval(self, k: int, t: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        """α_k(t; ctx_k). t: (m,1); ctx: (m, 2k) or None → (m,1)."""
        if ctx is None or ctx.shape[-1] == 0:
            inp = t
        else:
            inp = torch.cat([t, ctx], dim=-1)
        return self._alpha[k](inp)

    def forward(self, theta: torch.Tensor, context: Optional[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert context is not None and context.shape[-1] == self.d, (
            f"expected context (features) of width d={self.d}, got {None if context is None else context.shape}"
        )
        feats = context
        n = theta.shape[0]
        a = self.theta_ref
        K = self._nodes.shape[0]
        u = self._nodes.view(1, -1, 1).expand(n, -1, 1)      # (n, K, 1)
        weights = self._weights.view(1, -1, 1)
        r_cols: List[torch.Tensor] = []
        logdet_terms: List[torch.Tensor] = []
        for k in range(self.d):
            theta_k = theta[:, k:k + 1]
            feat_k = feats[:, k:k + 1]
            ctx_k = None if k == 0 else torch.cat([theta[:, :k], feats[:, :k]], dim=-1)  # (n, 2k)
            # β_k(feat_k; ctx_k), β'_k, b_k, b'_k
            beta = self._beta[k](feat_k, ctx_k)                       # (n, 1)
            beta_prime = self._beta[k].derivative(feat_k, ctx_k)      # (n, 1)
            b_val = self._b[k](feat_k, ctx_k)                         # (n, 1)
            b_prime = self._b[k].derivative(feat_k, ctx_k)            # (n, 1)
            # θ-integral: ∫_{a}^{θ_k} softplus(α_k(t;ctx_k) + β_k) dt via quadrature
            b_node = theta_k.view(n, 1, 1).expand(-1, K, -1)
            t = a + 0.5 * (b_node - a) * (u + 1.0)                    # (n, K, 1)
            if ctx_k is None:
                ctx_rep = None
            else:
                ctx_rep = ctx_k.unsqueeze(1).expand(-1, K, -1).reshape(n * K, -1)
            alpha_t = self._alpha_eval(k, t.reshape(n * K, 1), ctx_rep).view(n, K, 1)
            beta_b = beta.unsqueeze(1).expand(-1, K, -1)              # (n, K, 1)
            integrand = F.softplus(alpha_t + beta_b) + 1e-3
            integral = 0.5 * (theta_k - a) * (weights * integrand).sum(dim=1)   # (n, 1)
            r_k = b_val + integral                                   # (n, 1)
            # ∂r_k/∂feat_k = b'_k + β'_k · ∫ σ(α_k + β_k) dt
            sig = torch.sigmoid(alpha_t + beta_b)                    # (n, K, 1)
            sig_integral = 0.5 * (theta_k - a) * (weights * sig).sum(dim=1)     # (n, 1)
            dr_dfeat = b_prime + beta_prime * sig_integral           # (n, 1)
            r_cols.append(r_k)
            logdet_terms.append(torch.log(dr_dfeat.clamp_min(1e-12)).squeeze(-1))
        r = torch.cat(r_cols, dim=-1)                                # (n, d)
        log_det = torch.stack(logdet_terms, dim=-1).sum(dim=-1)      # (n,)
        return r, log_det

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run to verify all flow tests pass**

Run: `pytest tests/unit/test_triangular_doubly_monotone_flow.py -v`
Expected: PASS (6 passed). The `logdet_matches_autograd` + `monotone_in_features` tests confirm R2 and the closed-form Jacobian; `monotone_in_theta` confirms R1; the triangular check confirms `∂r_0/∂feat_1 = 0`.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/flows/triangular_doubly_monotone.py tests/unit/test_triangular_doubly_monotone_flow.py
git commit -m "feat(flows): TriangularDoublyMonotoneFlow (non-additive autoregressive)"
```

---

## Task 6: Hydra configs

**Files:** Create `configs/target/normal_mu_sigma.yaml`, `configs/flow/triangular_doubly_monotone.yaml`, `configs/conditioner/sufficient_stat.yaml`, `configs/experiment/mu_sigma_replication.yaml`.

- [ ] **Step 1: Write the configs**

`configs/target/normal_mu_sigma.yaml`:
```yaml
name: normal_mu_sigma
_target_: cdsbi.simulators.normal_unknown_mean_var.NormalUnknownMeanVar
n_iid: 10
```

`configs/flow/triangular_doubly_monotone.yaml`:
```yaml
name: triangular_doubly_monotone
_target_: cdsbi.flows.triangular_doubly_monotone.TriangularDoublyMonotoneFlow
hidden: ${budget.cdsbi_flow_hidden}
depth: 2
```

`configs/conditioner/sufficient_stat.yaml`:
```yaml
name: sufficient_stat
_target_: cdsbi.conditioners.sufficient_stat.SufficientStatConditioner
n_iid: 10
```

`configs/experiment/mu_sigma_replication.yaml`:
```yaml
# @package _global_
defaults:
  - override /target: normal_mu_sigma
  - override /flow: triangular_doubly_monotone
  - override /conditioner: sufficient_stat
  - override /method: cd_sbi
  - override /budget: medium

experiment:
  name: mu_sigma_replication
  n_eval: 6000
  eval_thetas_interior:
    - [-0.69, -3.0]   # log σ ≈ log 0.5, μ = -3
    - [-0.69,  3.0]
    - [ 0.0,   0.0]   # σ = 1, μ = 0
    - [ 0.69, -3.0]   # σ ≈ 2
    - [ 0.69,  3.0]
  eval_thetas_edge: []
  alpha_grid: [0.5, 0.68, 0.9, 0.95]
  n_eval_per_theta: 2000
  joint_mahalanobis_n_per_theta: 2000
```

- [ ] **Step 2: Verify Hydra composition**

Run: `python -m cdsbi.experiments.run experiment=mu_sigma_replication --cfg job 2>&1 | grep -E "name:|n_iid|hidden" | head`
Expected: shows `name: normal_mu_sigma`, `name: triangular_doubly_monotone`, `name: sufficient_stat`, `name: cd_sbi`. If it errors on an unknown group/key, fix the override path before proceeding.

- [ ] **Step 3: Commit**

```bash
git add configs/target/normal_mu_sigma.yaml configs/flow/triangular_doubly_monotone.yaml configs/conditioner/sufficient_stat.yaml configs/experiment/mu_sigma_replication.yaml
git commit -m "config: (μ,σ²) target + triangular_doubly_monotone flow + sufficient_stat conditioner"
```

---

## Task 7: Wire flow + conditioner into `run.py`

**Files:** Modify `src/cdsbi/experiments/run.py`.

The flow needs `d` injected from the simulator (like `TriangularAdditiveFlow`); the conditioner is instantiated from `cfg.conditioner` (currently `run` may hardcode Identity or build from a fixed path — verify and extend).

- [ ] **Step 1: Read the current flow/conditioner build paths**

Run: `grep -n "TriangularAdditiveFlow\|conditioner =\|Identity()\|MLPConditioner\|_build_flow\|cfg.target.name" src/cdsbi/experiments/run.py | head -30`
Verified structure (for reference): the CDSBI conditioner is built inside `_build_method` (~L180) by a **hardcoded target-name dispatch** — `if cfg.target.name == "exp_rate": MLPConditioner(frozen_sum) else: Identity()`. It does **not** read `cfg.conditioner`. (A second block ~L478 wraps non-CDSBI methods with a `ReducedSimulator` for exp_rate only — that's a baselines concern, **M4**, leave it untouched in M0.)

- [ ] **Step 2: Add the triangular_doubly_monotone fast-path to `_build_flow`**

In `_build_flow`, the v3 fast-path handles `{doubly_monotone, joint_umnn, joint_umnn_1d}`. Add `triangular_doubly_monotone` to a path that injects `d` from the simulator (it needs `d`, like `TriangularAdditiveFlow`). Locate the fast-path block:
```python
    v3_flow_names = {"doubly_monotone", "joint_umnn", "joint_umnn_1d"}
    if hydra_flow_name in v3_flow_names and (
        method_flow_label is None or method_flow_label == hydra_flow_name
    ):
        flow_dict = OmegaConf.to_container(cfg.flow, resolve=True)
        target = flow_dict.pop("_target_")
        flow_dict.pop("name", None)
        return _instantiate(target, **flow_dict)
```
Insert, just before it, a dedicated branch (it needs `d`):
```python
    if hydra_flow_name == "triangular_doubly_monotone" and (
        method_flow_label is None or method_flow_label == hydra_flow_name
    ):
        flow_dict = OmegaConf.to_container(cfg.flow, resolve=True)
        target = flow_dict.pop("_target_")
        flow_dict.pop("name", None)
        flow_dict.setdefault("d", int(simulator.d_theta))
        return _instantiate(target, **flow_dict)
```

- [ ] **Step 3: Make the CDSBI conditioner honor `cfg.conditioner`**

In `_build_method`, replace the hardcoded `if cfg.target.name == "exp_rate": … else: Identity()` block with one that **prefers an explicit non-identity `cfg.conditioner`** (so `sufficient_stat` — and M2's future learned summary — is built generically), falling back to exp_rate's frozen-sum and the Identity default:
```python
        # Conditioner dispatch: an explicit non-identity cfg.conditioner
        # (sufficient_stat now, learned summaries in M2) is built generically;
        # else exp_rate's frozen-sum reduction; else Identity passthrough.
        cond_name = OmegaConf.select(cfg, "conditioner.name", default="identity")
        if cond_name not in ("identity", None):
            cond_cfg = OmegaConf.to_container(cfg.conditioner, resolve=True)
            cond_target = cond_cfg.pop("_target_")
            cond_cfg.pop("name", None)
            conditioner = _instantiate(cond_target, **cond_cfg)
        elif cfg.target.name == "exp_rate":
            from cdsbi.conditioners.mlp import MLPConditioner
            conditioner = MLPConditioner(
                input_dim=int(simulator.d_x), output_dim=1, mode="frozen_sum",
            )
        else:
            from cdsbi.conditioners.identity import Identity
            conditioner = Identity()
```
The `sufficient_stat` config resolves to `_instantiate(SufficientStatConditioner, n_iid=10)`. The global default stays `conditioner: identity` (config.yaml), so existing experiments are unaffected; only `mu_sigma_replication` (which overrides `/conditioner: sufficient_stat`) gets the new path.

- [ ] **Step 4: Smoke-check a 50-step run composes and trains**

Run:
```bash
python -m cdsbi.experiments.run experiment=mu_sigma_replication seed=0 \
  training.n_train=2000 'training.n_steps=50' \
  method.flow=triangular_doubly_monotone run_dir=outputs/_m0_smoke 2>&1 | tail -5
```
Expected: completes without error, writes `outputs/_m0_smoke/...`, STATUS OK. (`method.flow=triangular_doubly_monotone` per the CLAUDE.md flow-dispatch gotcha so the fast-path fires for cd_sbi.) Confirm `model.pt`'s `arch_metadata.flow_class == "TriangularDoublyMonotoneFlow"`:
```bash
python -c "import torch,glob; ck=torch.load(glob.glob('outputs/_m0_smoke/**/model.pt',recursive=True)[0],map_location='cpu',weights_only=False); print(ck['arch_metadata']['flow_class'])"
```
Expected: `TriangularDoublyMonotoneFlow`.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/experiments/run.py
git commit -m "feat(run): wire TriangularDoublyMonotoneFlow + sufficient_stat conditioner"
```

---

## Task 8: Single-run validation smoke (recovers r*)

**Files:** Create `tests/integration/test_mu_sigma_smoke.py`.

A short full training run (small budget) that asserts the trained pivot is in a loose band around the closed-form `r*` and the marginal PIT is roughly uniform — the end-to-end Stage-A correctness check. Marked `intensive` (it trains).

- [ ] **Step 1: Write the test**

`tests/integration/test_mu_sigma_smoke.py`:

```python
"""Stage-A smoke: CDSBI on (μ,σ²) with oracle summary recovers r* in a loose band."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
from cdsbi.conditioners.sufficient_stat import SufficientStatConditioner
from cdsbi.flows.triangular_doubly_monotone import TriangularDoublyMonotoneFlow
from cdsbi.losses.nfmle import NFMLELoss
from cdsbi.methods.cd_sbi import CDSBIRunner


@pytest.mark.intensive
def test_stage_a_recovers_pivot():
    torch.manual_seed(0)
    sim = NormalUnknownMeanVar()
    flow = TriangularDoublyMonotoneFlow(d=2, hidden=32, depth=2)
    cond = SufficientStatConditioner(n_iid=sim.n_iid)
    runner = CDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss())
    config = {"lr": 3e-3, "batch_size": 256, "n_steps": 3000, "n_train": 10000,
              "optimizer": "adamw", "fresh_batch": False}
    trained = runner.fit(simulator=sim, config=config, seed=0)

    # Evaluate trained pivot vs closed-form r* on a fresh eval set.
    rng = np.random.default_rng(123)
    theta, x = sim.sample(4000, rng)
    with torch.no_grad():
        r_hat = trained.procedure.pivot(theta, x)
        r_true = sim.r_star(theta, x)
    rmse = float(((r_hat - r_true) ** 2).mean().sqrt())
    assert rmse < 0.25, f"pivot RMSE {rmse:.3f} too high — Stage A did not recover r*"
    # marginal PIT roughly uniform: Φ(r_hat) mean ≈ 0.5, std ≈ 1/√12 ≈ 0.289
    from scipy.stats import norm as _norm
    u = _norm.cdf(r_hat.numpy())
    assert abs(u.mean() - 0.5) < 0.05
```

- [ ] **Step 2: Run it (opt-in intensive — it trains; minutes on GPU)**

Run: `pytest tests/integration/test_mu_sigma_smoke.py -m intensive -v`
Expected: PASS. If `rmse` is borderline, bump `n_steps` to 5000 (the flow can represent `r*` exactly; under-training, not capacity, is the only failure mode here). If it FAILS structurally (not just borderline), STOP and report — it indicates a flow/wiring bug, not a tuning issue.

- [ ] **Step 3: Confirm the fast suite still passes**

Run: `pytest -q`
Expected: all prior + the new unit tests pass; the intensive smoke is deselected by default.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_mu_sigma_smoke.py
git commit -m "test(intensive): Stage-A (μ,σ²) smoke — recovers closed-form r*"
```

---

## Self-review notes

- **Spec coverage (M0):** target sampling (Task 1) + closed-form `r*`/`log_prob`/entropy (Task 2) ✓; oracle `SufficientStatConditioner` with (s², X̄) order + constant log-det (Task 3) ✓; the non-additive autoregressive `TriangularDoublyMonotoneFlow` built up from the conditioned monotone scalar (Tasks 4–5) ✓; configs (Task 6) ✓; run-wiring incl. the flow-dispatch gotcha + `d` injection (Task 7) ✓; end-to-end Stage-A recovery smoke (Task 8) ✓. M1 (MarginalCDRecovery, 2-D coverage grid, full replication), M2–M4 are out of scope here.
- **Placeholder scan:** every step has complete runnable code + exact commands; no TBD.
- **Type consistency:** `θ=(log σ, μ)` index order and `features=(s², X̄)` order are consistent across simulator, conditioner, flow tests, and configs. `_SUFFICIENT_STAT_LOG_DET_CONST` is defined once in the simulator module and imported by the conditioner (single source). Flow `forward(theta, context)` and conditioner `encode(x)` match the `Flow`/`Conditioner` protocols. `entropy_lower_bound` uses the same log-det constant as the conditioner.
- **Known follow-on (M2):** `CDSBIRunner.fit` optimizes only `self.flow.parameters()`; M0's oracle conditioner is zero-param so this is fine, but M2's learned `DeepSetsConditioner` requires the optimizer extension — flagged in the spec, not needed here.
- **Risk:** the `triangular_doubly_monotone` flow-dispatch in `run.py` mirrors the `cb1e08e` guard; Task 7 Step 4 verifies `flow_class == TriangularDoublyMonotoneFlow` to catch the AdditiveFlow1D fallback that bit the F2 milestone.
