# CD-SBI bivariate-normal (μ, Σ) — N1: Stage-A core

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage-A core for the bivariate-normal unknown-(μ, Σ) target (d_theta=5): the `NormalBivariateUnknownCov` simulator with the closed-form **Bartlett** joint pivot, the oracle `BartlettSummaryConditioner`, wire the existing d=5 `SingleIndexMonotoneFlow` in, and confirm a single CDSBI run recovers the pivot.

**Architecture:** New simulator `NormalBivariateUnknownCov` (θ = (ℓ₁₁, ℓ₂₂, L₂₁, μ₁, μ₂) log-Cholesky, `n_iid=10`, data flattened to ℝ²⁰, exact Bartlett `r_star`). New oracle `BartlettSummaryConditioner` (X→(log D₁₁, log D₂₂, D₂₁, X̄₁, X̄₂), frozen, constant log-det). The `SingleIndexMonotoneFlow` is already arbitrary-d; run.py already injects `d`/`theta_signs`/`feat_signs` from the simulator for `single_index_monotone`, so wiring is config-only. **N0 prototype validated this exact construction (RMSE 0.098).**

**Tech Stack:** PyTorch (vectorized Cholesky/Bartlett), numpy/scipy (`chi2`, `norm`), Hydra, pytest.

**Spec:** `docs/superpowers/specs/2026-05-30-cd-sbi-bivariate-normal-mu-cov-design.md` (§2 conventions, §3 components). **Deferred to N2:** `entropy_lower_bound` (the NF-MLE feature-scale floor — needs the 5×5 triangular feature-Jacobian; validate numerically against the trained loss in N2, where `FloorIntegrity` consumes it). N1 includes `data_entropy_lower_bound` (analytic, used by Stage-B in N3).

> **Guard-rail (deferral landmine):** `FloorIntegrity` maps `NFMLELoss → simulator.entropy_lower_bound()`. The N1 smoke (Task 5) and the Task-4 verify use the **direct API** (no `_run_diagnostics`), so they're safe. But do **NOT** run the full `python -m cdsbi.experiments.run experiment=mu_cov_replication` until N2 adds `entropy_lower_bound` — the diagnostics pass would `AttributeError`. N1 is validated via the direct-API smoke only.

---

## Conventions locked for N1 (from the validated N0 prototype)

- **θ order = `(ℓ₁₁, ℓ₂₂, L₂₁, μ₁, μ₂)`** (log-Cholesky: `Σ = L Lᵀ`, `L = [[exp ℓ₁₁, 0],[L₂₁, exp ℓ₂₂]]`). `d_theta = 5`.
- **Data X flattened to `(n, 20)`**; recover observations via `x.reshape(n, n_iid, 2)`. `d_x = n_iid·2 = 20`.
- **Features (oracle summary) = `(log D₁₁, log D₂₂, D₂₁, X̄₁, X̄₂)`** where `D` = lower-Cholesky of the scatter `A = Σᵢ(Xᵢ−X̄)(Xᵢ−X̄)ᵀ`.
- **Closed-form `r*` (increasing-θ convention), `~ N(0, I₅)` at truth** — `C₁₁=exp ℓ₁₁`, `C₂₂=exp ℓ₂₂`, `T₁₁=D₁₁/C₁₁`, `T₂₂=D₂₂/C₂₂`, `T₂₁=(D₂₁−(L₂₁/C₁₁)D₁₁)/C₂₂`:
  - `r₁ = Φ⁻¹(1 − F_{χ²_{n−1}}(T₁₁²))`
  - `r₂ = Φ⁻¹(1 − F_{χ²_{n−2}}(T₂₂²))`
  - `r₃ = T₂₁`
  - `r₄ = √n (μ₁ − X̄₁)/C₁₁`
  - `r₅ = √n (−(L₂₁/(C₁₁C₂₂))(μ₁ − X̄₁) + (μ₂ − X̄₂)/C₂₂)`
- **Signs:** `theta_signs = (+1,+1,−1,+1,+1)`, `feat_signs = (−1,−1,+1,−1,−1)`.
- **Prior:** `μ ~ U(−3,3)²`, `ℓᵢᵢ ~ U(log 0.4, log 2.5)`, `L₂₁ ~ U(−1.5, 1.5)`.

---

## File structure

```
NEW
  src/cdsbi/simulators/normal_bivariate_unknown_cov.py   # NormalBivariateUnknownCov
  src/cdsbi/conditioners/bartlett_summary.py             # BartlettSummaryConditioner (oracle)
  configs/target/normal_mu_cov.yaml
  configs/conditioner/bartlett_summary.yaml
  configs/experiment/mu_cov_replication.yaml
  tests/unit/test_normal_bivariate_unknown_cov.py
  tests/unit/test_bartlett_summary_conditioner.py
  tests/integration/test_mu_cov_smoke.py

MODIFY
  (none expected — flow wiring is config-only; verify in Task 4)
```

No change to `SingleIndexMonotoneFlow`, `run.py`, or existing diagnostics.

---

## Task 1: `NormalBivariateUnknownCov` — sampling + params + signs

**Files:**
- Create: `src/cdsbi/simulators/normal_bivariate_unknown_cov.py`
- Test: `tests/unit/test_normal_bivariate_unknown_cov.py`

READ `src/cdsbi/simulators/normal_unknown_mean_var.py` for the dataclass/`sample`/`_draw_theta` conventions.

- [ ] **Step 1: Write the failing tests**

```python
"""NormalBivariateUnknownCov: sampling, shapes, prior, signs."""
import numpy as np
import torch


def test_sample_shapes_and_dims():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    sim = NormalBivariateUnknownCov()
    assert sim.d_theta == 5 and sim.d_x == 20
    theta, x = sim.sample(64, np.random.default_rng(0))
    assert theta.shape == (64, 5) and x.shape == (64, 20)


def test_sample_x_given_theta_is_bivariate_normal():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    sim = NormalBivariateUnknownCov()
    theta0 = (0.2, -0.1, 0.5, 1.0, -2.0)        # (ℓ11, ℓ22, L21, μ1, μ2)
    x = sim.sample_x_given_theta(theta0, 20000, np.random.default_rng(1))
    assert x.shape == (20000, 20)
    obs = x.reshape(20000, sim.n_iid, 2).reshape(-1, 2).numpy()   # all observations
    import numpy as np
    L = np.array([[np.exp(0.2), 0.0], [0.5, np.exp(-0.1)]])
    Sig = L @ L.T
    assert np.allclose(obs.mean(0), [1.0, -2.0], atol=0.05)
    assert np.allclose(np.cov(obs.T), Sig, atol=0.08)


def test_signs():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    sim = NormalBivariateUnknownCov()
    assert tuple(sim.theta_signs) == (1.0, 1.0, -1.0, 1.0, 1.0)
    assert tuple(sim.feat_signs) == (-1.0, -1.0, 1.0, -1.0, -1.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_normal_bivariate_unknown_cov.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement sampling + params (r_star/oracle_summary/entropies in Task 2)**

```python
"""NormalBivariateUnknownCov: bivariate normal with unknown mean AND covariance.

θ = (ℓ₁₁, ℓ₂₂, L₂₁, μ₁, μ₂) log-Cholesky (Σ = L Lᵀ, L lower-triangular with
exp-diagonal). d_theta=5; data X (n_iid bivariate observations) flattened to ℝ²⁰.
The closed-form joint pivot r* (Task 2) is built from the Wishart Bartlett
decomposition — see the design spec §1–§2. Autoregressive order: covariance
Cholesky (diagonals, then off-diagonal) then mean.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from scipy.stats import chi2, norm


class NormalBivariateUnknownCov:
    def __init__(self, n_iid: int = 10,
                 mu_range: Tuple[float, float] = (-3.0, 3.0),
                 log_chol_range: Tuple[float, float] = (math.log(0.4), math.log(2.5)),
                 l21_range: Tuple[float, float] = (-1.5, 1.5)):
        self.n_iid = n_iid
        self.mu_range = mu_range
        self.log_chol_range = log_chol_range
        self.l21_range = l21_range
        self.d_theta = 5
        self.p = 2

    @property
    def d_x(self) -> int:
        return self.n_iid * self.p                                   # 20

    @property
    def theta_signs(self) -> Tuple[float, ...]:
        return (1.0, 1.0, -1.0, 1.0, 1.0)

    @property
    def feat_signs(self) -> Tuple[float, ...]:
        return (-1.0, -1.0, 1.0, -1.0, -1.0)

    def _draw_theta(self, n: int, rng: np.random.Generator) -> np.ndarray:
        l11 = rng.uniform(*self.log_chol_range, size=(n, 1))
        l22 = rng.uniform(*self.log_chol_range, size=(n, 1))
        L21 = rng.uniform(*self.l21_range, size=(n, 1))
        mu = rng.uniform(*self.mu_range, size=(n, 2))
        return np.concatenate([l11, l22, L21, mu], axis=1)           # (n,5)

    def _chol(self, theta_np: np.ndarray) -> np.ndarray:
        """Lower-Cholesky L per row, shape (n, 2, 2)."""
        n = theta_np.shape[0]
        L = np.zeros((n, 2, 2))
        L[:, 0, 0] = np.exp(theta_np[:, 0])
        L[:, 1, 1] = np.exp(theta_np[:, 1])
        L[:, 1, 0] = theta_np[:, 2]
        return L

    def _sample_x(self, theta_np: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """(n, n_iid, 2) bivariate-normal draws given θ rows."""
        n = theta_np.shape[0]
        L = self._chol(theta_np)                                     # (n,2,2)
        mu = theta_np[:, 3:5]                                        # (n,2)
        z = rng.standard_normal(size=(n, self.n_iid, 2))             # (n,n_iid,2)
        x = mu[:, None, :] + np.einsum('nij,nkj->nki', L, z)         # μ + z Lᵀ
        return x

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_np = self._draw_theta(n, rng)
        x = self._sample_x(theta_np, rng).reshape(n, self.d_x)       # flatten (n,20)
        return torch.from_numpy(theta_np).float(), torch.from_numpy(x).float()

    def sample_x_given_theta(self, theta_0, n: int, rng: np.random.Generator) -> torch.Tensor:
        theta_vec = np.atleast_1d(np.asarray(theta_0, dtype=np.float64))
        assert theta_vec.shape == (self.d_theta,), (
            f"theta_0 has shape {theta_vec.shape}, expected ({self.d_theta},)"
        )
        theta_rep = np.tile(theta_vec[None, :], (n, 1))
        x = self._sample_x(theta_rep, rng).reshape(n, self.d_x)
        return torch.from_numpy(x).float()
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_normal_bivariate_unknown_cov.py -v`
Expected: 3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/simulators/normal_bivariate_unknown_cov.py tests/unit/test_normal_bivariate_unknown_cov.py
git commit -m "feat(sim): NormalBivariateUnknownCov sampling (log-Cholesky, d=5, flat ℝ²⁰)"
```

---

## Task 2: Bartlett `r_star` + `oracle_summary` + `data_entropy_lower_bound`

**Files:**
- Modify: `src/cdsbi/simulators/normal_bivariate_unknown_cov.py`
- Test: `tests/unit/test_normal_bivariate_unknown_cov.py` (append)

The load-bearing test: `r*(θ₀, X)` for `X ~ p(·|θ₀)` is `~ N(0, I₅)` (the N0 result, now pinned as a regression).

- [ ] **Step 1: Write the failing tests (append)**

```python
def test_r_star_is_standard_normal_at_truth():
    import numpy as np
    from scipy.stats import kstest
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    sim = NormalBivariateUnknownCov()
    rng = np.random.default_rng(3)
    theta0 = np.array([0.3, -0.2, 0.6, 0.5, -1.0])
    x = sim.sample_x_given_theta(theta0, 8000, rng)
    theta = torch.tensor(theta0, dtype=torch.float32).expand(8000, 5)
    r = sim.r_star(theta, x).numpy()
    assert r.shape == (8000, 5)
    # each coordinate ~ N(0,1) at the truth (the Bartlett pivot)
    for j in range(5):
        assert abs(r[:, j].mean()) < 0.06, f"coord {j} mean {r[:, j].mean():.3f}"
        assert abs(r[:, j].std() - 1.0) < 0.06, f"coord {j} std {r[:, j].std():.3f}"
        assert kstest(r[:, j], "norm").statistic < 0.04, f"coord {j} KS"
    # ~ independent (off-diagonal correlations small)
    corr = np.corrcoef(r.T)
    assert np.abs(corr - np.eye(5))[~np.eye(5, dtype=bool)].max() < 0.05


def test_oracle_summary_shape_and_data_entropy():
    import numpy as np, math
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    sim = NormalBivariateUnknownCov()
    x = sim.sample_x_given_theta((0.0, 0.0, 0.0, 0.0, 0.0), 32, np.random.default_rng(0))
    feats = sim.oracle_summary(x)
    assert feats.shape == (32, 5)
    H = sim.data_entropy_lower_bound()
    mid = 0.5 * (sim.log_chol_range[0] + sim.log_chol_range[1])
    expected = sim.n_iid * (0.5 * sim.p * (1 + math.log(2 * math.pi)) + 2 * mid)
    assert abs(H - expected) < 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_normal_bivariate_unknown_cov.py::test_r_star_is_standard_normal_at_truth -v` → FAIL (no `r_star`).

- [ ] **Step 3: Implement `r_star`, `oracle_summary`, `data_entropy_lower_bound`**

Add these methods (vectorized; numpy bridge for `chi2`/`norm`, like the 1-D simulator):

```python
    def _bartlett(self, x: torch.Tensor):
        """Returns (xbar (n,2), D (n,2,2) lower-Cholesky of the scatter)."""
        n = x.shape[0]
        obs = x.reshape(n, self.n_iid, self.p)                       # (n,n_iid,2)
        xbar = obs.mean(dim=1)                                       # (n,2)
        Xc = obs - xbar[:, None, :]
        A = Xc.transpose(1, 2) @ Xc                                  # (n,2,2) scatter
        D = torch.linalg.cholesky(A)                                 # (n,2,2) lower-tri
        return xbar, D

    def r_star(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Closed-form Bartlett joint pivot (n, 5) ~ N(0, I₅) at the truth."""
        n_obs = self.n_iid
        xbar, D = self._bartlett(x)
        D11 = D[:, 0, 0]; D21 = D[:, 1, 0]; D22 = D[:, 1, 1]
        l11 = theta[:, 0]; l22 = theta[:, 1]; L21 = theta[:, 2]
        mu1 = theta[:, 3]; mu2 = theta[:, 4]
        C11 = torch.exp(l11); C22 = torch.exp(l22)
        T11 = D11 / C11; T22 = D22 / C22
        T21 = (D21 - (L21 / C11) * D11) / C22
        # χ² covariance pivots (numpy bridge), increasing-θ convention (1 − F)
        w1 = (T11 ** 2).detach().cpu().numpy(); w2 = (T22 ** 2).detach().cpu().numpy()
        u1 = np.clip(1.0 - chi2.cdf(w1, df=n_obs - 1), 1e-12, 1 - 1e-12)
        u2 = np.clip(1.0 - chi2.cdf(w2, df=n_obs - 2), 1e-12, 1 - 1e-12)
        r1 = torch.from_numpy(norm.ppf(u1)).float().to(theta.device)
        r2 = torch.from_numpy(norm.ppf(u2)).float().to(theta.device)
        r3 = T21
        sq = math.sqrt(n_obs)
        r4 = sq * (mu1 - xbar[:, 0]) / C11
        r5 = sq * (-(L21 / (C11 * C22)) * (mu1 - xbar[:, 0]) + (mu2 - xbar[:, 1]) / C22)
        return torch.stack([r1, r2, r3, r4, r5], dim=-1)

    def oracle_summary(self, x: torch.Tensor) -> torch.Tensor:
        """(log D₁₁, log D₂₂, D₂₁, X̄₁, X̄₂), shape (n, 5) — the flow's feature order."""
        xbar, D = self._bartlett(x)
        D11 = D[:, 0, 0].clamp_min(1e-12); D22 = D[:, 1, 1].clamp_min(1e-12)
        return torch.stack([torch.log(D11), torch.log(D22), D[:, 1, 0],
                            xbar[:, 0], xbar[:, 1]], dim=-1)

    def data_entropy_lower_bound(self) -> float:
        """H(X|θ) = n·[½ p (1+log2π) + log|L|], averaged over the prior;
        log|L| = ℓ₁₁ + ℓ₂₂ → prior mean 2·midpoint(log_chol_range)."""
        mid = 0.5 * (self.log_chol_range[0] + self.log_chol_range[1])
        return self.n_iid * (0.5 * self.p * (1 + math.log(2 * math.pi)) + 2 * mid)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_normal_bivariate_unknown_cov.py -v`
Expected: 5 pass. The `test_r_star_is_standard_normal_at_truth` is load-bearing (the Bartlett pivot — KS < 0.04 each coord, correlations ≈ I).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/simulators/normal_bivariate_unknown_cov.py tests/unit/test_normal_bivariate_unknown_cov.py
git commit -m "feat(sim): Bartlett r* (N(0,I₅) at truth) + oracle_summary + data_entropy_lower_bound"
```

---

## Task 3: `BartlettSummaryConditioner` (oracle X → 5-D features)

**Files:**
- Create: `src/cdsbi/conditioners/bartlett_summary.py`
- Test: `tests/unit/test_bartlett_summary_conditioner.py`

READ `src/cdsbi/conditioners/sufficient_stat.py` (the encode contract + the `_SUFFICIENT_STAT_LOG_DET_CONST` pattern).

- [ ] **Step 1: Write the failing test**

```python
"""BartlettSummaryConditioner: oracle X → (log D11, log D22, D21, X̄1, X̄2)."""
import numpy as np
import torch


def test_encode_matches_simulator_oracle_summary():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    from cdsbi.conditioners.bartlett_summary import BartlettSummaryConditioner
    sim = NormalBivariateUnknownCov()
    cond = BartlettSummaryConditioner(n_iid=sim.n_iid)
    x = sim.sample_x_given_theta((0.0, 0.0, 0.3, 1.0, -1.0), 64, np.random.default_rng(0))
    feats, log_det = cond.encode(x)
    assert feats.shape == (64, 5) and log_det.shape == (64,)
    assert torch.allclose(feats, sim.oracle_summary(x), atol=1e-5)   # same transform
    assert cond.n_params() == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_bartlett_summary_conditioner.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `src/cdsbi/conditioners/bartlett_summary.py`**

```python
"""BartlettSummaryConditioner: oracle reduction X → (log D₁₁, log D₂₂, D₂₁, X̄₁, X̄₂)
for the bivariate-normal (μ, Σ) target. D = lower-Cholesky of the sample scatter.
Zero-parameter; θ-independent constant log-det (sufficiency). Pairs with the
log-Cholesky θ-order (ℓ₁₁, ℓ₂₂, L₂₁, μ₁, μ₂) in the autoregressive flow.
"""
from __future__ import annotations

from typing import Tuple

import torch

# θ-independent volume constant for the sufficient-statistic reduction (as in the
# 1-D SufficientStatConditioner: by sufficiency p(X | suff) is θ-free, so this only
# shifts the loss scale, not the argmin). Set to 0.0.
_BARTLETT_LOG_DET_CONST = 0.0


class BartlettSummaryConditioner:
    def __init__(self, n_iid: int, p: int = 2):
        self.n_iid = n_iid
        self.p = p

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[-1] == self.n_iid * self.p, (
            f"BartlettSummaryConditioner expected width {self.n_iid * self.p}, got {x.shape[-1]}"
        )
        n = x.shape[0]
        obs = x.reshape(n, self.n_iid, self.p)
        xbar = obs.mean(dim=1)
        Xc = obs - xbar[:, None, :]
        A = Xc.transpose(1, 2) @ Xc
        D = torch.linalg.cholesky(A)
        D11 = D[:, 0, 0].clamp_min(1e-12); D22 = D[:, 1, 1].clamp_min(1e-12)
        feats = torch.stack([torch.log(D11), torch.log(D22), D[:, 1, 0],
                             xbar[:, 0], xbar[:, 1]], dim=-1)         # (n,5)
        log_det = torch.full((n,), _BARTLETT_LOG_DET_CONST, dtype=x.dtype, device=x.device)
        return feats, log_det

    def n_params(self) -> int:
        return 0
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_bartlett_summary_conditioner.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/conditioners/bartlett_summary.py tests/unit/test_bartlett_summary_conditioner.py
git commit -m "feat(cond): BartlettSummaryConditioner oracle X→(log D11,log D22,D21,X̄1,X̄2)"
```

---

## Task 4: configs + wiring verification

**Files:**
- Create: `configs/target/normal_mu_cov.yaml`, `configs/conditioner/bartlett_summary.yaml`, `configs/experiment/mu_cov_replication.yaml`

READ `configs/target/normal_mu_sigma.yaml`, `configs/conditioner/sufficient_stat.yaml`, `configs/experiment/mu_sigma_replication.yaml` to mirror their shape.

- [ ] **Step 1: `configs/target/normal_mu_cov.yaml`**

```yaml
name: normal_mu_cov
_target_: cdsbi.simulators.normal_bivariate_unknown_cov.NormalBivariateUnknownCov
n_iid: 10
```

- [ ] **Step 2: `configs/conditioner/bartlett_summary.yaml`**

```yaml
name: bartlett_summary
_target_: cdsbi.conditioners.bartlett_summary.BartlettSummaryConditioner
n_iid: 10
```

- [ ] **Step 3: `configs/experiment/mu_cov_replication.yaml`** (5-D LHS coverage grid is N2; N1 uses a small fixed grid for the smoke's plumbing)

```yaml
# @package _global_
defaults:
  - override /target: normal_mu_cov
  - override /flow: single_index_monotone
  - override /conditioner: bartlett_summary
  - override /method: cd_sbi
  - override /budget: medium

method:
  flow: single_index_monotone        # self-contained flow dispatch (cb1e08e guard)

experiment:
  name: mu_cov_replication
  n_eval: 6000
  # placeholder 3-point θ₀ grid for plumbing; N2 replaces with a ~16-pt LHS grid
  eval_thetas_interior:
    - [0.0, 0.0, 0.0, 0.0, 0.0]
    - [0.4, -0.4, 0.8, 1.5, -1.5]
    - [-0.4, 0.4, -0.8, -1.5, 1.5]
  eval_thetas_edge: []
  alpha_grid: [0.5, 0.68, 0.9, 0.95]
  n_eval_per_theta: 2000
  joint_mahalanobis_n_per_theta: 2000
```

- [ ] **Step 4: Verify the flow wires at d=5 with the right signs**

```bash
python -c "
from hydra import initialize, compose
from cdsbi.experiments.run import _build_flow, _build_simulator
with initialize(version_base=None, config_path='configs'):
    cfg = compose(config_name='config', overrides=['experiment=mu_cov_replication'])
sim = _build_simulator(cfg); flow = _build_flow(cfg, sim)
print('sim d_theta/d_x:', sim.d_theta, sim.d_x)
print('flow:', type(flow).__name__, 'd=', flow.d)
print('signs θ:', flow._s_theta.tolist(), 'feat:', flow._s_feat.tolist())
"
```
Expected: `d_theta/d_x: 5 20`, flow `SingleIndexMonotoneFlow d= 5`, `θ signs [1,1,-1,1,1]`, `feat [-1,-1,1,-1,-1]`. If the flow isn't `SingleIndexMonotoneFlow` or d≠5, the `method.flow` override / sign injection is wrong — STOP and check `run.py`'s `_build_flow` single_index branch (it should `setdefault('d', simulator.d_theta)` + inject signs).

- [ ] **Step 5: Commit**

```bash
git add configs/target/normal_mu_cov.yaml configs/conditioner/bartlett_summary.yaml configs/experiment/mu_cov_replication.yaml
git commit -m "config: normal_mu_cov target + bartlett_summary conditioner + mu_cov_replication experiment"
```

---

## Task 5: Stage-A recovery smoke (intensive)

**Files:**
- Create: `tests/integration/test_mu_cov_smoke.py`

Confirms a single CDSBI run recovers the 5-D Bartlett pivot in a loose band (the d=5 analog of the 1-D smoke; N0 got 0.098 against the analytic pivot, training adds slack → band 0.20).

- [ ] **Step 1: Write the test**

```python
"""Stage-A smoke: CDSBI on (μ,Σ) with the Bartlett oracle recovers r* in a band."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.intensive
def test_stage_a_mu_cov_recovers_pivot():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    from cdsbi.conditioners.bartlett_summary import BartlettSummaryConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi import CDSBIRunner

    torch.manual_seed(0)
    sim = NormalBivariateUnknownCov()
    flow = SingleIndexMonotoneFlow(d=5, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=48, depth=2)
    cond = BartlettSummaryConditioner(n_iid=sim.n_iid)
    runner = CDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss())
    config = {"lr": 3e-3, "batch_size": 256, "n_steps": 6000, "n_train": 10000,
              "optimizer": "adamw", "fresh_batch": False}
    trained = runner.fit(simulator=sim, config=config, seed=0)

    rng = np.random.default_rng(123)
    theta, x = sim.sample(4000, rng)
    with torch.no_grad():
        r_hat = trained.procedure.pivot(theta, x).cpu()
        r_true = sim.r_star(theta, x).cpu()
    rmse = float(((r_hat - r_true) ** 2).mean().sqrt())
    print(f"mu_cov rmse={rmse:.4f}")
    assert rmse < 0.20, f"pivot RMSE {rmse:.3f} too high — Stage A did not recover r*"
    # joint Mahalanobis PIT ~ χ²₅ at a central θ₀
    from scipy.stats import kstest, chi2
    xv = sim.sample_x_given_theta((0.0, 0.0, 0.0, 0.0, 0.0), 3000, np.random.default_rng(7))
    th = torch.zeros(3000, 5)
    with torch.no_grad():
        r = trained.procedure.pivot(th, xv).cpu().numpy()
    ks = kstest(chi2.cdf((r ** 2).sum(1), df=5), "uniform").statistic
    print(f"mu_cov joint Mahalanobis KS={ks:.3f}")
    assert ks < 0.08
```

- [ ] **Step 2: Run it (intensive; ~minutes on GPU)**

Run: `pytest tests/integration/test_mu_cov_smoke.py -v -s -m intensive`
Expected: PASS — RMSE ≲ 0.15 (N0 was 0.098 against the analytic pivot; the trained pipeline adds slack), joint KS < 0.08. If RMSE is high, print per-coordinate RMSE (`((r_hat-r_true)**2).mean(0).sqrt()`) to see which coordinate lags — the off-diagonal `L₂₁` or `μ₂` are the likely ones (the cross-coupled coords); report rather than loosen the band. (A coordinate stuck high would indicate the single-index ctx-magnitude isn't capturing the cross-scaling at the trained recipe — bump `n_steps` or `hidden`, not the tolerance.)

- [ ] **Step 3: Confirm fast suite green**

Run: `pytest -q` (intensive deselected). Report the count.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_mu_cov_smoke.py
git commit -m "test(intensive): (μ,Σ) Stage-A recovery smoke (d=5 Bartlett pivot)"
```

---

## Self-review

- **Spec coverage (§2–§3):** simulator sampling + log-Cholesky + signs (Task 1) ✓; Bartlett `r*` + oracle_summary + data_entropy_lower_bound (Task 2) ✓; `BartlettSummaryConditioner` (Task 3) ✓; configs + d=5 flow wiring verification (Task 4) ✓; recovery smoke (Task 5) ✓. **Deferred (noted):** `entropy_lower_bound` (NF-MLE feature-scale floor) → N2; LHS coverage grid → N2; MarginalCDRecovery generalization → N2; Stage-B I-A → N3.
- **Placeholder scan:** every step has runnable code + commands + expected output; no TBD.
- **Type consistency:** θ order `(ℓ₁₁, ℓ₂₂, L₂₁, μ₁, μ₂)` and features `(log D₁₁, log D₂₂, D₂₁, X̄₁, X̄₂)` are consistent across the simulator `r_star`/`oracle_summary`, the conditioner `encode`, the signs, and the configs. `d_theta=5`, `d_x=20`. The conditioner's `encode` matches `simulator.oracle_summary` (Task 3 test asserts it). Flow signs come from the simulator (Task 4 verifies). `r_star`/`oracle_summary` both use the shared `_bartlett` (single source for the Cholesky).
- **Risk — Cholesky PD-ness.** `torch.linalg.cholesky(A)` needs `A` strictly PD; with `n_iid=10 > p=2` the scatter is PD almost surely, but a degenerate draw would raise. Acceptable for n_iid=10 (no clamp added; if it ever bites in N2's larger sweeps, add a tiny ridge — not premature here).
- **Risk — cross-coupled coordinate recovery.** N0 validated all 5 coords recover (RMSE 0.098) at a longer/cleaner recipe; the smoke band (0.20) has margin. Task 5 prints per-coord RMSE to catch a lagging cross-term early.
- **Known follow-on:** N2 (entropy_lower_bound + MarginalCDRecovery χ²×Hotelling + LHS coverage + intensive replication), N3 (Stage-B I-A invertible ℝ²⁰→ℝ⁵).
