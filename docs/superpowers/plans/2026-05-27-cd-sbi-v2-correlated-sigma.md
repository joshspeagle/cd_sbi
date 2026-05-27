# CD-SBI v2 — §8.3 multivariate location-Gaussian (correlated Σ) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the v1 §8.2 infrastructure from `Σ = I` to a correlated Σ = [[1, 0.5], [0.5, 1]], replicate manuscript §8.3 with CDSBI, validate the Knothe–Rosenblatt uniqueness empirical claim via a new Jacobian-recovery diagnostic, and produce a cross-method comparison sweep + manuscript §8.3 integration.

**Architecture:**
- `LocationGaussian2D_corr` simulator: `X | θ ~ N(θ, Σ)`, `θ ~ U[-7, 7]^2`, closed-form `r*(θ, X) = L⁻¹(θ - X)` where `L L^T = Σ`. New method `r_star_jacobian(theta, x)` returning `L⁻¹` (constant per the location-family structure); diagnostic compares trained `∂r/∂θ` to this.
- `JacobianRecovery` diagnostic: at `K = 200` random joint `(θ, X)` points, compute `J_emp = ∂r/∂θ` via autograd of the trained pivot, aggregate to `E[J_emp]`, and compare to the simulator's `r_star_jacobian`. Reports `max_residual = max |J_emp_mean - J_true|` and `norm_residual = ||J_emp_mean - J_true||_F / ||J_true||_F`. Only applies to PivotBased procedures (others return NaN + skip-reason in `meta`).
- No new flow: `TriangularAdditiveFlow` from v1 covers the correlated case; only the data distribution changes. The conditioning network for `r_2` now learns the nontrivial cross-coupling slope `-0.577` on `(θ_1, X_1)`.
- No new method runners, no new confidence-set code, no new training infrastructure: v1 already produced d > 1 versions of every layer.
- New target / experiment YAMLs (`target/loc_gauss_2d_corr`, `experiment/8_3_replication`, `experiment/8_3_baseline_sweep`).
- New `paper_table_8_3` thin shim that extends `paper_table_8_2` with a `jacobian_max_residual` column for CDSBI rows.
- Manuscript §8.3 integration as a final task with the same seven-block template that landed §8.1 / §8.2.

**Tech Stack:** PyTorch (autograd for `∂r/∂θ`), Hydra, nflows / sbi (MAF backbone for NPE/NLE, posteriors for credible-region inversion), pytest, pandas + parquet.

**Out of scope (carried from v1 addendum + v0 backlog):**
- True joint HPD region for NPE's `PosteriorBasedProcedure` at d > 1 (currently Bonferroni-conservative marginal product). The §8.3 cross-method table inherits the same caveat as §8.2; v2 does NOT fix this, but Task 9 must footnote it the same way Task 15 of v1 did.
- (R2) ablation tests (deferred to v3 §8.4).
- `r_star_jacobian` on `LocationNormal1D` / `LocationGaussian2D_iid` — only the correlated simulator needs it for v2; iid backfill is a v3+ generalization if it ever becomes useful.

---

## File structure

```
NEW
  src/cdsbi/simulators/location_gauss_2d_corr.py    # correlated-Σ 2D Gaussian simulator
  src/cdsbi/diagnostics/jacobian_recovery.py        # autograd ∂r/∂θ vs r_star_jacobian
  configs/target/loc_gauss_2d_corr.yaml
  configs/experiment/8_3_replication.yaml
  configs/experiment/8_3_baseline_sweep.yaml
  tests/unit/test_location_gauss_2d_corr.py
  tests/diagnostics/test_jacobian_recovery.py
  tests/intensive/test_replicate_8_3.py

MODIFY
  src/cdsbi/experiments/run.py                      # wire JacobianRecovery into the diagnostic battery; lift jacobian_max_residual into index_row
  src/cdsbi/analysis/paper_tables.py                # paper_table_8_3 shim
  docs/superpowers/plans/2026-05-27-cd-sbi-v2-correlated-sigma.md  # this file (self-review may patch it)
  cd_sbi_v7.tex                                     # §8.3 manuscript integration (Task 9)
```

No changes to: flows, confidence_set procedures, methods runners, training utils, loss, conditioners, reproducibility/, six existing diagnostics. v1 already lifted these to d > 1.

---

## Task 1: `LocationGaussian2D_corr` simulator

The §8.2 iid simulator covers `Σ = I`. v2 adds a correlated covariance simulator that exposes `r_star_jacobian` for the Jacobian-recovery diagnostic. Σ is class-defaulted to the manuscript's `[[1, 0.5], [0.5, 1]]` and the Cholesky factor `L` is precomputed once at construction.

**Files:**
- Create: `src/cdsbi/simulators/location_gauss_2d_corr.py`
- Test: `tests/unit/test_location_gauss_2d_corr.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_location_gauss_2d_corr.py`:

```python
"""LocationGaussian2D_corr — correlated-Σ 2D Gaussian simulator."""
from __future__ import annotations

import numpy as np
import torch


def test_simulator_sample_shape_and_correlation(seed):
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    rng = np.random.default_rng(seed)
    sim = LocationGaussian2D_corr()
    assert sim.d_theta == 2 and sim.d_x == 2
    theta, x = sim.sample(20000, rng)
    assert theta.shape == (20000, 2)
    assert x.shape == (20000, 2)
    # X - θ ~ N(0, Σ); empirical Σ ≈ [[1, 0.5], [0.5, 1]].
    eps = (x - theta).numpy()
    cov_emp = np.cov(eps.T)
    np.testing.assert_allclose(cov_emp, [[1.0, 0.5], [0.5, 1.0]], atol=0.06)


def test_sample_x_given_theta_shape_and_correlation(seed):
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    rng = np.random.default_rng(seed)
    sim = LocationGaussian2D_corr()
    x = sim.sample_x_given_theta(theta_0=[1.5, -0.5], n=10000, rng=rng)
    assert x.shape == (10000, 2)
    # X | θ=[1.5, -0.5] ~ N([1.5, -0.5], Σ)
    np.testing.assert_allclose(x.mean(0).numpy(), [1.5, -0.5], atol=0.05)
    cov_emp = np.cov((x - torch.tensor([1.5, -0.5])).T.numpy())
    np.testing.assert_allclose(cov_emp, [[1.0, 0.5], [0.5, 1.0]], atol=0.07)


def test_r_star_returns_decorrelated_normal(seed):
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    rng = np.random.default_rng(seed)
    sim = LocationGaussian2D_corr()
    theta, x = sim.sample(20000, rng)
    r = sim.r_star(theta, x).numpy()
    # r* = L⁻¹(θ - X) ~ N(0, I_2). Mean ≈ 0, cov ≈ I.
    np.testing.assert_allclose(r.mean(0), [0.0, 0.0], atol=0.05)
    np.testing.assert_allclose(np.cov(r.T), [[1.0, 0.0], [0.0, 1.0]], atol=0.05)


def test_r_star_jacobian_is_L_inverse():
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    sim = LocationGaussian2D_corr()
    J = sim.r_star_jacobian()
    # L⁻¹ for Σ = [[1, 0.5], [0.5, 1]]; L is lower-triangular with
    # L = [[1, 0], [0.5, sqrt(0.75)]]; L⁻¹ = [[1, 0], [-0.5/sqrt(0.75), 1/sqrt(0.75)]]
    # = [[1, 0], [-0.5774, 1.1547]] to 4 decimals.
    expected = np.array([[1.0, 0.0], [-0.5773502691896258, 1.1547005383792515]])
    np.testing.assert_allclose(J.numpy(), expected, atol=1e-6)


def test_log_prob_matches_independent_gaussian_formula(seed):
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    rng = np.random.default_rng(seed)
    sim = LocationGaussian2D_corr()
    theta, x = sim.sample(100, rng)
    # Reference: scipy.stats.multivariate_normal.logpdf
    from scipy.stats import multivariate_normal
    Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    expected = np.array([
        multivariate_normal(mean=t.numpy(), cov=Sigma).logpdf(xi.numpy())
        for t, xi in zip(theta, x)
    ])
    got = sim.log_prob(x, theta).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_location_gauss_2d_corr.py -v`

Expected: FAIL on import with `ModuleNotFoundError: No module named 'cdsbi.simulators.location_gauss_2d_corr'`.

- [ ] **Step 3: Implement the simulator**

Create `src/cdsbi/simulators/location_gauss_2d_corr.py`:

```python
"""LocationGaussian2D_corr: X | θ ~ N(θ, Σ) with Σ = [[1, 0.5], [0.5, 1]]
and θ ~ U[-7, 7]^2. r*(θ, X) = L⁻¹(θ - X) where L L^T = Σ.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch


_SIGMA_DEFAULT = ((1.0, 0.5), (0.5, 1.0))


@dataclass
class LocationGaussian2D_corr:
    theta_range: Tuple[float, float] = (-7.0, 7.0)
    d_theta: int = 2
    d_x: int = 2
    # Σ as a nested tuple so the @dataclass remains hashable + YAML-friendly.
    sigma: Tuple[Tuple[float, float], Tuple[float, float]] = field(
        default_factory=lambda: _SIGMA_DEFAULT,
    )

    def __post_init__(self) -> None:
        assert self.d_theta == self.d_x, (
            "correlated 2D Gaussian assumes d_theta == d_x"
        )
        sigma_np = np.asarray(self.sigma, dtype=np.float64)
        assert sigma_np.shape == (self.d_theta, self.d_theta), (
            f"sigma has shape {sigma_np.shape}, expected ({self.d_theta}, {self.d_theta})"
        )
        # Cholesky factor; lower-triangular by convention.
        self._L = np.linalg.cholesky(sigma_np)
        self._L_inv = np.linalg.inv(self._L)
        self._log_det_sigma = float(np.log(np.linalg.det(sigma_np)))

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = self.theta_range
        theta_np = rng.uniform(a, b, size=(n, self.d_theta))
        z_np = rng.standard_normal(size=(n, self.d_x))
        # X = θ + L @ z  ⇒  X - θ ~ N(0, Σ)
        eps_np = z_np @ self._L.T
        x_np = theta_np + eps_np
        return (
            torch.from_numpy(theta_np).float(),
            torch.from_numpy(x_np).float(),
        )

    def sample_x_given_theta(self, theta_0, n: int, rng: np.random.Generator) -> torch.Tensor:
        """Draw n samples of X conditional on θ = θ_0."""
        theta_vec = np.atleast_1d(np.asarray(theta_0, dtype=np.float64))
        assert theta_vec.shape == (self.d_theta,), (
            f"theta_0 has shape {theta_vec.shape}, expected ({self.d_theta},)"
        )
        z_np = rng.standard_normal(size=(n, self.d_x))
        eps_np = z_np @ self._L.T
        x_np = theta_vec[None, :] + eps_np
        return torch.from_numpy(x_np).float()

    def r_star(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Truth pivot: r* = L⁻¹(θ - X). Returns shape (n, d)."""
        L_inv = torch.from_numpy(self._L_inv).to(dtype=theta.dtype, device=theta.device)
        # (theta - x) is (n, d); (L_inv @ (theta-x)^T)^T == (theta-x) @ L_inv^T
        return (theta - x) @ L_inv.T

    def r_star_jacobian(self) -> torch.Tensor:
        """Jacobian ∂r*/∂θ = L⁻¹ (constant across (θ, X) by the location-family structure)."""
        return torch.from_numpy(self._L_inv).float()

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # log p(X | θ) = -d/2 log(2π) - ½ log|Σ| - ½ (X-θ)^T Σ⁻¹ (X-θ)
        L_inv = torch.from_numpy(self._L_inv).to(dtype=x.dtype, device=x.device)
        z = (x - theta) @ L_inv.T  # whitened residuals; ||z||² == (X-θ)^T Σ⁻¹ (X-θ)
        return (
            -0.5 * self.d_x * math.log(2 * math.pi)
            - 0.5 * self._log_det_sigma
            - 0.5 * z.pow(2).sum(dim=-1)
        )

    def entropy_lower_bound(self) -> float:
        # H(N(θ, Σ)) = ½ log((2πe)^d |Σ|)
        return 0.5 * (self.d_x * math.log(2 * math.pi * math.e) + self._log_det_sigma)
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/unit/test_location_gauss_2d_corr.py -v`

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/simulators/location_gauss_2d_corr.py tests/unit/test_location_gauss_2d_corr.py
git commit -m "$(cat <<'EOF'
feat(simulators): LocationGaussian2D_corr (§8.3 target)

Correlated-Σ 2D Gaussian: X | θ ~ N(θ, Σ) with Σ = [[1, 0.5], [0.5, 1]]
and θ ~ U[-7, 7]^2. Truth pivot r*(θ, X) = L⁻¹(θ - X) where L L^T = Σ.

Adds r_star_jacobian() returning L⁻¹ as the constant truth Jacobian
(location-family structure makes ∂r*/∂θ independent of (θ, X)). The
v2 JacobianRecovery diagnostic consumes this for the KR-uniqueness
empirical check; r_star is also used by PivotRMSE per the v1 protocol.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `JacobianRecovery` diagnostic

The §8.3 KR-uniqueness empirical claim is: at the autoregressive triangular flow's population optimum, `r(θ, X) = L⁻¹(θ - X)` and the recovered `∂r/∂θ` should approach `L⁻¹` (constant). At finite training, the average over random `(θ, X)` should land within a few percent of the truth Jacobian even when pointwise pivot RMSE is large (because the deviation is finite-network noise around the unique KR rearrangement, not a non-canonical fit). This diagnostic encodes that claim.

**Files:**
- Modify: `src/cdsbi/diagnostics/base.py` — extend the `Diagnostic` protocol to declare the `x_per_theta` keyword that v1's `JointMahalanobis` / `Coverage` / `SetSize` already accept de facto.
- Create: `src/cdsbi/diagnostics/jacobian_recovery.py`
- Test: `tests/diagnostics/test_jacobian_recovery.py`

- [ ] **Step 1a: Extend the `Diagnostic` protocol**

In `src/cdsbi/diagnostics/base.py`, replace the existing protocol declaration:

```python
@runtime_checkable
class Diagnostic(Protocol):
    name: str

    def __call__(self, trained, simulator, eval_data) -> DiagnosticResult: ...
```

with:

```python
@runtime_checkable
class Diagnostic(Protocol):
    name: str

    def __call__(
        self, trained, simulator, eval_data=None, x_per_theta=None,
    ) -> DiagnosticResult: ...
```

This is a no-op for `PivotRMSE`, `MarginalPIT`, and `ConditionalPIT` (their `__call__` already accepts `eval_data` and ignores the rest); `Coverage`, `SetSize`, and `JointMahalanobis` already accept `x_per_theta=None`. The change just makes the contract explicit.

Run `pytest -q` after this edit — expected: 163+ passed, no regressions.

- [ ] **Step 1b: Write the failing test**

Create `tests/diagnostics/test_jacobian_recovery.py`:

```python
"""Jacobian-recovery diagnostic regression tests."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import pandas as pd


class _LinearPivotProcedure:
    """Pivot procedure that implements r(θ, X) = J_const (θ - X) for a fixed J_const."""
    def __init__(self, J_const: torch.Tensor):
        self._J = J_const

    def pivot(self, theta, x):
        return (theta - x) @ self._J.T

    @property
    def d_theta(self):
        return int(self._J.shape[1])


def test_jacobian_recovery_recovers_truth_when_pivot_is_truth():
    """If procedure.pivot is literally r* = L⁻¹(θ - X), the diagnostic must
    report max_residual ≈ 0 (any non-zero is numerical noise from the K-point
    Monte-Carlo aggregation)."""
    from cdsbi.diagnostics.jacobian_recovery import JacobianRecovery
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    sim = LocationGaussian2D_corr()
    L_inv = sim.r_star_jacobian()
    trained = SimpleNamespace(procedure=_LinearPivotProcedure(L_inv))
    diag = JacobianRecovery(n_points=200)
    rng = np.random.default_rng(0)
    eval_data = sim.sample(500, rng)
    result = diag(trained, sim, eval_data=eval_data)
    # Constant-Jacobian pivot ⇒ empirical mean Jacobian should equal truth exactly
    # (modulo float32 autograd noise; tolerance well below the diagnostic floor).
    assert result.value["max_residual"].iloc[0] < 1e-4, result.value


def test_jacobian_recovery_flags_wrong_pivot():
    """If procedure.pivot uses the wrong constant Jacobian, max_residual must
    be the elementwise distance between the two matrices."""
    from cdsbi.diagnostics.jacobian_recovery import JacobianRecovery
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    sim = LocationGaussian2D_corr()
    # Use the iid identity as the (wrong) trained pivot.
    wrong = torch.eye(2)
    trained = SimpleNamespace(procedure=_LinearPivotProcedure(wrong))
    diag = JacobianRecovery(n_points=200)
    rng = np.random.default_rng(0)
    eval_data = sim.sample(500, rng)
    result = diag(trained, sim, eval_data=eval_data)
    L_inv = sim.r_star_jacobian()
    expected_max = float((L_inv - wrong).abs().max())
    assert abs(result.value["max_residual"].iloc[0] - expected_max) < 1e-4


def test_jacobian_recovery_skips_non_pivot_procedure():
    """Non-pivot procedures get NaN + skip reason in meta, mirroring JointMahalanobis."""
    from cdsbi.diagnostics.jacobian_recovery import JacobianRecovery
    from cdsbi.simulators.location_gauss_2d_corr import LocationGaussian2D_corr
    sim = LocationGaussian2D_corr()
    trained = SimpleNamespace(procedure=object())  # not PivotBasedProcedure
    diag = JacobianRecovery(n_points=200)
    rng = np.random.default_rng(0)
    eval_data = sim.sample(100, rng)
    result = diag(trained, sim, eval_data=eval_data)
    assert np.isnan(result.value["max_residual"].iloc[0])
    assert "not a pivot-based procedure" in result.meta.get("reason", "")


def test_jacobian_recovery_skips_simulator_without_truth_jacobian():
    """If the simulator does not expose r_star_jacobian(), the diagnostic
    no-ops the same way (the truth Jacobian is needed to score the empirical one).
    Forward-compatible: lets the diagnostic be wired by default in the run.py
    battery without breaking §8.1 / §8.2 runs."""
    from cdsbi.diagnostics.jacobian_recovery import JacobianRecovery
    from cdsbi.simulators.location_normal_1d import LocationNormal1D
    sim = LocationNormal1D()
    trained = SimpleNamespace(procedure=_LinearPivotProcedure(torch.tensor([[1.0]])))
    diag = JacobianRecovery(n_points=50)
    rng = np.random.default_rng(0)
    eval_data = sim.sample(100, rng)
    result = diag(trained, sim, eval_data=eval_data)
    assert np.isnan(result.value["max_residual"].iloc[0])
    assert "no r_star_jacobian" in result.meta.get("reason", "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/diagnostics/test_jacobian_recovery.py -v`

Expected: FAIL on import with `ModuleNotFoundError: No module named 'cdsbi.diagnostics.jacobian_recovery'`.

- [ ] **Step 3: Implement the diagnostic**

Create `src/cdsbi/diagnostics/jacobian_recovery.py`:

```python
"""JacobianRecovery: at K random (θ, X) joint samples, compare ∂r/∂θ
(via autograd of the trained pivot) to the simulator's r_star_jacobian.

Validates the §8.3 Knothe–Rosenblatt uniqueness claim: even when pointwise
pivot RMSE is large (because L⁻¹ scales the (θ - X) residual nontrivially),
the average Jacobian should match the unique KR rearrangement to within a
few percent. Only meaningful for PivotBasedProcedure + simulators that
expose r_star_jacobian (currently LocationGaussian2D_corr only; falls
through to a no-op skip otherwise).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.base import Diagnostic, DiagnosticResult


def _skip(name: str, reason: str) -> DiagnosticResult:
    df = pd.DataFrame([{
        "max_residual": float("nan"),
        "norm_residual": float("nan"),
        "passed": True,
        "n_points": 0,
    }])
    return DiagnosticResult(
        name=name, value=df, passed=True, noise_floor=0.0,
        n_samples=0, meta={"reason": reason},
    )


class JacobianRecovery(Diagnostic):
    name = "jacobian_recovery"

    def __init__(self, n_points: int = 200, max_residual_tol: float = 0.05) -> None:
        self.n_points = n_points
        self.max_residual_tol = max_residual_tol

    def __call__(self, trained, simulator, eval_data=None, x_per_theta=None) -> DiagnosticResult:
        if not isinstance(trained.procedure, PivotBasedProcedure):
            return _skip(self.name, "not a pivot-based procedure")
        if not hasattr(simulator, "r_star_jacobian"):
            return _skip(self.name, "simulator has no r_star_jacobian (no truth target)")

        J_true = simulator.r_star_jacobian()
        d = int(simulator.d_theta)
        assert J_true.shape == (d, d), (
            f"r_star_jacobian returned shape {J_true.shape}, expected ({d}, {d})"
        )

        # Sample n_points joint (θ, X) from the eval split if it covers enough
        # points; otherwise draw fresh ones from the simulator.
        if eval_data is not None:
            unpacked = eval_data if not isinstance(eval_data, tuple) or len(eval_data) == 2 else eval_data[:2]
            theta_eval, x_eval = unpacked
            if theta_eval.shape[0] >= self.n_points:
                idx = torch.randperm(theta_eval.shape[0])[: self.n_points]
                theta_pts = theta_eval[idx].clone()
                x_pts = x_eval[idx].clone()
            else:
                rng = np.random.default_rng(0)
                theta_pts, x_pts = simulator.sample(self.n_points, rng)
        else:
            rng = np.random.default_rng(0)
            theta_pts, x_pts = simulator.sample(self.n_points, rng)

        # Per-point Jacobian via autograd. We avoid jacfwd/jacrev to stay
        # compatible with the existing flow forward (which doesn't promise to
        # be vmap-traceable); a Python loop is fine at n_points=200.
        # x_k is detached: the Jacobian is ∂r/∂θ holding (X, flow weights)
        # fixed, so we explicitly cut grad propagation through X to avoid
        # tracing through the flow parameters during the inner grad call.
        J_emp = torch.empty(self.n_points, d, d)
        for k in range(self.n_points):
            theta_k = theta_pts[k : k + 1].clone().detach().requires_grad_(True)
            x_k = x_pts[k : k + 1].detach()
            r_k = trained.procedure.pivot(theta_k, x_k)  # shape (1, d)
            for i in range(d):
                grad_i = torch.autograd.grad(
                    r_k[0, i], theta_k, retain_graph=(i < d - 1),
                )[0]  # shape (1, d)
                J_emp[k, i, :] = grad_i[0].detach()

        J_emp_mean = J_emp.mean(dim=0)  # (d, d)
        diff = (J_emp_mean - J_true)
        max_residual = float(diff.abs().max())
        norm_residual = float(diff.norm() / J_true.norm())
        passed = max_residual <= self.max_residual_tol

        df = pd.DataFrame([{
            "max_residual": max_residual,
            "norm_residual": norm_residual,
            "passed": passed,
            "n_points": int(self.n_points),
        }])
        return DiagnosticResult(
            name=self.name, value=df, passed=passed, noise_floor=self.max_residual_tol,
            n_samples=int(self.n_points),
            meta={"d": d, "tol": self.max_residual_tol},
        )
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/diagnostics/test_jacobian_recovery.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/diagnostics/jacobian_recovery.py tests/diagnostics/test_jacobian_recovery.py
git commit -m "$(cat <<'EOF'
feat(diagnostics): JacobianRecovery (manuscript §8.3 KR-uniqueness check)

At K = 200 random joint (θ, X) samples, computes ∂r/∂θ via autograd of
the trained pivot and compares its mean to the simulator's r_star_jacobian.
Skips no-op for non-pivot procedures and simulators without a truth
Jacobian (forward-compatible: lets the diagnostic stay in the default
run.py battery without breaking §8.1 / §8.2 runs).

The §8.3 empirical claim: pointwise pivot RMSE is large (because
L⁻¹ scales the (θ - X) residual nontrivially), but the average
Jacobian matches the unique KR rearrangement to within a few percent.
This diagnostic encodes that claim with a default tolerance of 5%.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire `JacobianRecovery` into the diagnostic battery

Add the new diagnostic to `_run_diagnostics` in `src/cdsbi/experiments/run.py` so every run that uses a PivotBased procedure + a simulator with `r_star_jacobian` produces a `jacobian_recovery.parquet`. Lift `jacobian_max_residual` into `index_row.parquet` so `paper_table_8_3` can read it without parsing the per-run parquet.

**Files:**
- Modify: `src/cdsbi/experiments/run.py`
- Test: extend `tests/integration/test_run_diagnostics.py` (file exists from v0)

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_run_diagnostics.py`:

```python
def test_run_produces_jacobian_recovery_parquet_on_corr_target(tmp_path, seed):
    """End-to-end: a single CDSBI run on the correlated simulator writes a
    jacobian_recovery.parquet with sane numeric columns and lifts
    jacobian_max_residual into index_row.parquet."""
    import subprocess, pandas as pd
    from pathlib import Path

    out = tmp_path / "run"
    cmd = [
        "python", "-m", "cdsbi.experiments.run",
        f"hydra.run.dir={out}",
        "experiment=8_3_replication",
        "method=cd_sbi",
        "budget=small",
        f"seed={seed}",
        "training.fresh_batch=false",
        # Trim diagnostic costs for the integration smoke. set_size_n_per_theta
        # is NOT in 8_3_replication.yaml, so it needs `+`; joint_mahalanobis_n_per_theta
        # IS in the YAML (= 2000), so use plain assignment to override it.
        "+experiment.set_size_n_per_theta=25",
        "experiment.joint_mahalanobis_n_per_theta=300",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
    assert result.returncode == 0, result.stderr
    diag = pd.read_parquet(out / "diagnostics" / "jacobian_recovery.parquet")
    assert {"max_residual", "norm_residual", "passed", "n_points"}.issubset(diag.columns)
    assert diag["n_points"].iloc[0] > 0
    idx = pd.read_parquet(out / "index_row.parquet")
    assert "jacobian_max_residual" in idx.columns
    assert idx["jacobian_max_residual"].iloc[0] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_run_diagnostics.py::test_run_produces_jacobian_recovery_parquet_on_corr_target -v`

Expected: FAIL — either `experiment=8_3_replication` config doesn't exist (Task 4 lands it; expected here) OR the parquet isn't written (this task lands the wiring). For sequencing, write Task 4 (configs) immediately before re-running this test.

- [ ] **Step 3: Wire JacobianRecovery into `_run_diagnostics`**

In `src/cdsbi/experiments/run.py`, locate the `diagnostics = [...]` list inside `_run_diagnostics` (around lines 207-227) and add `JacobianRecovery` as the seventh entry. After the existing import block at the top of `_run_diagnostics`, add:

```python
    from cdsbi.diagnostics.jacobian_recovery import JacobianRecovery
```

Then add to the `diagnostics` list, after `joint_mahalanobis`:

```python
        ("jacobian_recovery", JacobianRecovery(
            n_points=int(OmegaConf.select(
                cfg, "experiment.jacobian_recovery_n_points", default=200,
            )),
            max_residual_tol=float(OmegaConf.select(
                cfg, "experiment.jacobian_recovery_tol", default=0.05,
            )),
        )),
```

The diagnostic's protocol matches Coverage / SetSize / JointMahalanobis (takes `(trained, simulator, eval_data, x_per_theta=None)`), but it doesn't need the shared `x_per_theta` dict — it samples its own (θ, X) joint pairs from `eval_data`. Do NOT add `"jacobian_recovery"` to `x_sharing_names`; let it fall through the else branch.

- [ ] **Step 4: Lift `jacobian_max_residual` into `index_row.parquet`**

In `_write_index_row` (around lines 283-320), after the `jm_path` block, add:

```python
    jac_path = rd.path / "diagnostics" / "jacobian_recovery.parquet"
    if jac_path.exists():
        jac_df = pd.read_parquet(jac_path)
        if "max_residual" in jac_df.columns and len(jac_df):
            row["jacobian_max_residual"] = float(jac_df["max_residual"].iloc[0])
```

- [ ] **Step 5: Run the integration test (Task 4's configs must land first)**

After Task 4 lands the `experiment=8_3_replication` config, run:

```bash
pytest tests/integration/test_run_diagnostics.py::test_run_produces_jacobian_recovery_parquet_on_corr_target -v
```

Expected: PASS.

- [ ] **Step 6: Run the full fast test suite**

Run: `pytest -q`

Expected: 163+ passed (the existing fast tests + the new diagnostic tests + the integration test). No regressions.

- [ ] **Step 7: Commit**

```bash
git add src/cdsbi/experiments/run.py tests/integration/test_run_diagnostics.py
git commit -m "$(cat <<'EOF'
feat(run): wire JacobianRecovery; lift jacobian_max_residual to index_row

Adds the v2 Jacobian-recovery diagnostic as the 7th entry in the
default diagnostic battery and lifts max_residual into the per-run
index_row.parquet so paper_table_8_3 can aggregate without re-parsing
per-run parquets. The diagnostic skips no-op on non-correlated targets
(missing r_star_jacobian) and non-pivot procedures, so §8.1 / §8.2
runs are unaffected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Hydra configs for §8.3

Three YAMLs: the target config, the single-run replication config, and the multi-method baseline-sweep config. All mirror v1's §8.2 YAMLs with `loc_gauss_2d_iid` replaced by `loc_gauss_2d_corr`. The replication config additionally pins `joint_mahalanobis_n_per_theta` and `jacobian_recovery_n_points` to the values the intensive test will assert against.

**Files:**
- Create: `configs/target/loc_gauss_2d_corr.yaml`
- Create: `configs/experiment/8_3_replication.yaml`
- Create: `configs/experiment/8_3_baseline_sweep.yaml`

- [ ] **Step 1: Write `configs/target/loc_gauss_2d_corr.yaml`**

Create:

```yaml
name: loc_gauss_2d_corr
_target_: cdsbi.simulators.location_gauss_2d_corr.LocationGaussian2D_corr
theta_range: [-7.0, 7.0]
# sigma defaults to [[1.0, 0.5], [0.5, 1.0]] per the manuscript; override here
# only if you want to sweep correlation strength (e.g. for a §11 follow-up).
```

- [ ] **Step 2: Write `configs/experiment/8_3_replication.yaml`**

Create:

```yaml
# @package _global_
defaults:
  - override /target: loc_gauss_2d_corr
  - override /flow: triangular_additive
  - override /method: cd_sbi
  - override /budget: medium

experiment:
  name: 8_3_replication
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
  joint_mahalanobis_n_per_theta: 2000
  jacobian_recovery_n_points: 200
  jacobian_recovery_tol: 0.05
```

- [ ] **Step 3: Write `configs/experiment/8_3_baseline_sweep.yaml`**

Create:

```yaml
# @package _global_
defaults:
  - override /target: loc_gauss_2d_corr
  - override /flow: triangular_additive

experiment:
  name: 8_3_baseline_sweep
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
  joint_mahalanobis_n_per_theta: 2000
  jacobian_recovery_n_points: 200
  jacobian_recovery_tol: 0.05

# Launch with explicit sweep dims (Hydra ignores hydra.sweeper.params inside
# experiment group configs):
#   python -m cdsbi.experiments.run -m experiment=8_3_baseline_sweep \
#     method=cd_sbi,npe,nle,nre,lf2i_bff \
#     budget=small,medium,large,xlarge \
#     seed=0,1,2,3,4 \
#     training.fresh_batch=false
```

- [ ] **Step 4: Verify configs load via a 1-run smoke**

Run:

```bash
python -m cdsbi.experiments.run \
  experiment=8_3_replication \
  method=cd_sbi \
  budget=small \
  seed=0 \
  training.fresh_batch=false \
  +experiment.set_size_n_per_theta=25 \
  +experiment.joint_mahalanobis_n_per_theta=300 \
  hydra.run.dir=/tmp/cdsbi_8_3_smoke
```

Note the `joint_mahalanobis_n_per_theta` override uses `experiment.…=300` (no `+` prefix) because the key is defined by `8_3_replication.yaml`; the `+` prefix is only for keys not yet in the config.

Expected: exit code 0; `/tmp/cdsbi_8_3_smoke/STATUS` reads `OK`; the run dir contains `diagnostics/jacobian_recovery.parquet`.

- [ ] **Step 5: Commit**

```bash
git add configs/target/loc_gauss_2d_corr.yaml configs/experiment/8_3_replication.yaml configs/experiment/8_3_baseline_sweep.yaml
git commit -m "$(cat <<'EOF'
config(experiment): §8.3 target + replication + baseline sweep YAMLs

Mirrors the §8.2 YAML structure with loc_gauss_2d_iid → loc_gauss_2d_corr.
Adds two new experiment knobs (jacobian_recovery_n_points,
jacobian_recovery_tol) consumed by the v2 JacobianRecovery diagnostic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Budget pre-flight check (no retune expected, but verify)

v1's TriangularAdditiveFlow at d=2 took budget config `cdsbi_flow_hidden_d2` (1k → 100k parameter targets). The §8.3 simulator changes the data distribution but NOT the flow architecture, so the same hidden sizes apply. This task verifies that the d=2 flow lands inside its budget tolerance band on the new simulator (no Σ-dependence in parameter count is expected, but confirm), and that LF2I-BFF's retuned `quantile_hidden` schedule (from the v1 close-out fix, commit 8bae8c9) carries over.

**Files:** (no code changes; this task is a one-shot validation)

- [ ] **Step 1: Run the per-method n_params check**

Run, from the repo root:

```bash
python <<'PY'
from omegaconf import OmegaConf
from cdsbi.flows.triangular_additive import TriangularAdditiveFlow
from cdsbi.methods.lf2i_bff import LF2IBFFRunner

print(f"{'budget':10s} {'cdsbi_hidden':>12s} {'cdsbi_params':>12s} {'lf2i_total':>11s}")
for budget in ["small", "medium", "large", "xlarge"]:
    cfg = OmegaConf.load(f"configs/budget/{budget}.yaml")
    cdsbi_h = int(cfg.cdsbi_flow_hidden_d2)
    cl_h = int(cfg.classifier_hidden)
    q_h = int(cfg.quantile_hidden)
    flow = TriangularAdditiveFlow(d=2, hidden=cdsbi_h)
    cdsbi_params = sum(p.numel() for p in flow.parameters())
    runner = LF2IBFFRunner(
        classifier_hidden=cl_h, classifier_depth=2,
        quantile_hidden=q_h, quantile_depth=2,
    )
    lf2i_n = runner.n_params(d_theta=2, d_x=2, alpha_grid_len=4)["total"]
    print(f"{budget:10s} {cdsbi_h:>12d} {cdsbi_params:>12d} {lf2i_n:>11d}")
PY
```

Reading the hidden sizes directly from the YAMLs avoids the v0.3 retune-drift trap: if `cdsbi_flow_hidden_d2` or `quantile_hidden` ever changes in `configs/budget/*.yaml`, this script automatically picks it up rather than silently checking against stale hardcoded numbers.

Expected (matches the v1 close-out numbers):

```
budget       cdsbi_hidden cdsbi_params  lf2i_total
small                  14         1080        1499
medium                 32         4752        5489
large                  76        24640       26201
xlarge                156       100480      102079
```

If any number deviates from these by more than 1% the budget configs need a retune — open a follow-up issue and pause Task 6 until resolved.

- [ ] **Step 2: Document this as a pre-flight check in CLAUDE.md**

Append to `CLAUDE.md` under "v0 codebase status" (or its v2 equivalent if the section has been reorganized):

```
**v2 §8.3 pre-flight:** TriangularAdditiveFlow(d=2) parameter counts are
Σ-independent; the v1 cdsbi_flow_hidden_d2 schedule carries over unchanged.
LF2I-BFF's retuned quantile_hidden schedule (commit 8bae8c9) also applies
without modification. Verify via `python tools/check_v2_budgets.py` (or
the inline snippet in v2 plan Task 5) before launching the §8.3 sweep.
```

- [ ] **Step 3: Commit** (only the CLAUDE.md note; no code changes)

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(v2): note §8.3 budget grid carries over from §8.2 unchanged

TriangularAdditiveFlow(d=2) parameter counts are Σ-independent, and
LF2I-BFF's retuned quantile_hidden schedule from the v1 close-out
already covers the §8.3 sweep. No new budget retune needed; pre-flight
check documented for v2 executors.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: §8.3 CDSBI replication intensive test

Mirror v1's `test_replicate_8_2.py` for §8.3. Single-seed wall is ~1 min on GPU; 5 seeds ~5-8 min. Tolerance bands chosen against the manuscript's single-seed §8.3 numbers (pivot RMSE 0.27, joint Mahalanobis 0.012, coverage error < 0.015, Jacobian recovery within 1-2%) with the same 1.5-2× headroom we used in v1.

**Files:**
- Create: `tests/intensive/test_replicate_8_3.py`

- [ ] **Step 1: Write the test**

Create `tests/intensive/test_replicate_8_3.py`:

```python
"""§8.3 replication: CDSBI on LocationGaussian2D_corr matches tolerance bands."""
from pathlib import Path
import subprocess

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.intensive
def test_replicate_8_3_cdsbi_matches_tolerance(tmp_path):
    """Run CDSBI at medium budget across 5 seeds; seed-averaged metrics within tolerance.

    Tolerance bands per manuscript §8.3 (loosened ~1.5–2× from the single-seed
    point estimates to absorb seed scatter, mirroring the v1 approach):
      - pivot_rmse mean ≤ 0.45 (manuscript: 0.27 — high pointwise residual is
        expected because L⁻¹ scales r_2 toward unit marginal variance, not 1/30
        of dynamic range as in §8.2; the Jacobian + joint Mahalanobis are the
        load-bearing diagnostics here, per the §8.3 commentary)
      - coverage_error_max mean ≤ 0.035 (manuscript: < 0.015)
      - joint_mahal_ks ≤ 2 × per-θ_0 floor for ≥ 4/5 seeds
      - jacobian_max_residual mean ≤ 0.05 (manuscript: 1–2%)
    """
    out_dir = tmp_path / "sweep"
    for seed in range(5):
        cmd = [
            "python", "-m", "cdsbi.experiments.run",
            f"hydra.run.dir={out_dir}/run_seed_{seed}",
            "experiment=8_3_replication",
            "method=cd_sbi",
            "budget=medium",
            f"seed={seed}",
            "training.fresh_batch=false",
            # SetSize at d>1 falls through to ray-bisection; cap it the same way
            # v1's §8.2 intensive test does.
            "+experiment.set_size_n_per_theta=25",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        assert result.returncode == 0, f"seed {seed} failed:\n{result.stderr}"

    from cdsbi.analysis.loaders import load_runs
    df = load_runs(str(out_dir / "*"))
    assert len(df) == 5
    assert df["pivot_rmse"].mean() <= 0.45, f"pivot_rmse mean = {df['pivot_rmse'].mean()}"
    assert df["coverage_error_max"].mean() <= 0.035, (
        f"coverage_error mean = {df['coverage_error_max'].mean()}"
    )
    # JacobianRecovery wiring is load-bearing for the §8.3 KR-uniqueness claim
    # — fail loudly if the column is missing rather than silently skipping.
    assert "jacobian_max_residual" in df.columns, (
        "JacobianRecovery not wired through to index_row.parquet"
    )
    assert df["jacobian_max_residual"].mean() <= 0.05, (
        f"jacobian_max_residual mean = {df['jacobian_max_residual'].mean()}"
    )

    # Joint Mahalanobis: load per-run parquet and check KS ≤ 2× floor for ≥ 4/5.
    # Fail loudly if a JM parquet is missing — a silent skip would let a corrupt
    # run mask a real failure by deflating n_pass.
    import glob, os
    n_pass = 0
    run_dirs = sorted(glob.glob(str(out_dir / "*")))
    assert len(run_dirs) == 5, f"expected 5 run dirs, got {len(run_dirs)}"
    for rd in run_dirs:
        jm_path = os.path.join(rd, "diagnostics/joint_mahalanobis.parquet")
        assert os.path.exists(jm_path), f"JM parquet missing for {rd}"
        jm = pd.read_parquet(jm_path)
        if (jm["ks"] <= 2.0 * jm["noise_floor"]).all():
            n_pass += 1
    assert n_pass >= 4, (
        f"only {n_pass}/5 seeds passed JointMahalanobis at 2× floor; expected ≥ 4"
    )
```

- [ ] **Step 2: Run the test**

Run: `pytest -m intensive tests/intensive/test_replicate_8_3.py -v`

Expected: PASS. Wall time ~5–8 min on GPU.

If `pivot_rmse mean` exceeds 0.45 but the Jacobian recovery and joint Mahalanobis pass, the band needs widening (not the implementation fixing) — match the manuscript framing that pivot RMSE is the wrong primary diagnostic in the correlated case. Document any band-widening in the commit message.

- [ ] **Step 3: Commit**

```bash
git add tests/intensive/test_replicate_8_3.py
git commit -m "$(cat <<'EOF'
test(intensive): §8.3 replication for CDSBI on correlated-Σ target

5 seeds × medium budget; asserts pivot_rmse, coverage_error_max,
joint_mahal_ks (≥4/5 seeds at 2× floor), and jacobian_max_residual
all land in tolerance bands derived from the manuscript's §8.3
single-seed numbers with 1.5–2× headroom for seed scatter.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `paper_table_8_3` shim

Extend `paper_table_8_2` with `jacobian_max_residual` as a 6th aggregated column. The function gracefully handles methods that didn't compute the diagnostic (non-CDSBI rows have NaN, which `groupby.mean()` silently drops).

**Files:**
- Modify: `src/cdsbi/analysis/paper_tables.py`
- Test: extend `tests/unit/test_paper_tables.py` (file exists from v1)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_paper_tables.py`:

```python
def test_paper_table_8_3_includes_jacobian_column():
    import pandas as pd
    from cdsbi.analysis.paper_tables import paper_table_8_3
    df = pd.DataFrame([
        {"method": "cd_sbi", "budget_name": "medium", "coverage_error_max": 0.02,
         "marginal_ks": 0.013, "pivot_rmse": 0.27, "joint_mahal_ks": 0.012,
         "jacobian_max_residual": 0.03, "actual_params_total": 4752},
        {"method": "cd_sbi", "budget_name": "medium", "coverage_error_max": 0.022,
         "marginal_ks": 0.014, "pivot_rmse": 0.28, "joint_mahal_ks": 0.011,
         "jacobian_max_residual": 0.04, "actual_params_total": 4752},
        {"method": "npe", "budget_name": "medium", "coverage_error_max": 0.07,
         "marginal_ks": None, "pivot_rmse": None, "joint_mahal_ks": None,
         "jacobian_max_residual": None, "actual_params_total": 5064},
    ])
    t = paper_table_8_3(df)
    assert "jacobian_max_residual_mean" in t.columns
    assert "jacobian_max_residual_std" in t.columns
    # CDSBI row should have the mean of [0.03, 0.04] = 0.035
    assert abs(t.loc[("cd_sbi", "medium"), "jacobian_max_residual_mean"] - 0.035) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_paper_tables.py::test_paper_table_8_3_includes_jacobian_column -v`

Expected: FAIL with `AttributeError: module 'cdsbi.analysis.paper_tables' has no attribute 'paper_table_8_3'`.

- [ ] **Step 3: Implement the shim**

Append to `src/cdsbi/analysis/paper_tables.py`:

```python
def paper_table_8_3(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot (method × budget) → seed-averaged §8.3 metrics.

    Extends paper_table_8_2 with jacobian_max_residual (the §8.3-specific
    KR-uniqueness empirical metric). Non-pivot methods will have NaN in
    this column; groupby.mean() drops them silently.
    """
    metrics = [
        "coverage_error_max", "marginal_ks", "pivot_rmse",
        "joint_mahal_ks", "jacobian_max_residual", "actual_params_total",
    ]
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby(["method", "budget_name"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_paper_tables.py -v`

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/analysis/paper_tables.py tests/unit/test_paper_tables.py
git commit -m "$(cat <<'EOF'
feat(analysis): paper_table_8_3 — adds jacobian_max_residual column

Thin shim over paper_table_8_2 that surfaces the v2 KR-uniqueness
diagnostic in the seed-aggregated table. Non-pivot methods report NaN
for the Jacobian column; pandas groupby drops them silently so the
CDSBI row carries the meaningful aggregate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Launch the §8.3 baseline sweep + generate paper table

End-to-end validation of the v2 deliverable: 5 methods × 4 budgets × 5 seeds = 100 runs. Same protocol as v1 Task 14. Mid-sweep GPU memory at xlarge stays bounded because the multivariate ray-bisection is already wrapped in `torch.no_grad()` (commit b4ba38c, landed late in v1 close-out).

**Files:** (no code changes; this task documents the v2 deliverable end-to-end)

- [ ] **Step 1: Launch the cross-method sweep**

Run:

```bash
python -m cdsbi.experiments.run -m \
  experiment=8_3_baseline_sweep \
  method=cd_sbi,npe,nle,nre,lf2i_bff \
  budget=small,medium,large,xlarge \
  seed=0,1,2,3,4 \
  training.fresh_batch=false
```

Expected: 100 runs complete (5 methods × 4 budgets × 5 seeds). Total wall ~30–90 min depending on GPU contention with other workloads; the LF2I-BFF xlarge runs are the slowest (~15–20 min each) per the v1 §8.2 experience.

Watch for two known-good behaviors:
1. GPU memory at LF2I-BFF xlarge stays around 15.5/16.3 GB — high but not OOMing, because the d > 1 ray-bisection is wrapped in `torch.no_grad()` (commit b4ba38c).
2. NPE's credible region is Bonferroni-conservative at d=2 (marginal product, per the v1 addendum). NPE will look worse than it should; flag this in the §8.3 paper table as it was flagged in §8.2.

If any run fails with STATUS != OK, diagnose before regenerating the table — do not silently re-run with different parameters.

- [ ] **Step 2: Aggregate the paper table**

Run:

```python
from cdsbi.analysis.loaders import load_runs
from cdsbi.analysis.paper_tables import paper_table_8_3

df = load_runs("outputs/8_3_baseline_sweep/<timestamp>/*/")
assert len(df) == 100, f"expected 100 OK runs, got {len(df)}"
print(paper_table_8_3(df))
```

Expected output: a DataFrame indexed by (method, budget) with mean and std for `coverage_error_max`, `marginal_ks`, `pivot_rmse`, `joint_mahal_ks`, `jacobian_max_residual`, `actual_params_total`. CDSBI's row should show:
- `coverage_error_max_mean` at the MC noise floor (~0.02 — match §8.2 pattern)
- `jacobian_max_residual_mean` ≤ 0.05 (consistent with the manuscript's 1–2%)
- `joint_mahal_ks_mean` at or below the per-θ_0 noise floor (~0.022, mirroring §8.2)

For non-CDSBI methods, expect:
- NLE: 3–4× the floor (same gap as §8.2; the Bayes/frequentist coincidence remains broken)
- NPE: 3–5× the floor (Bonferroni-conservative; same caveat as §8.2)
- LF2I-BFF: 5× the floor, roughly flat across budgets after the v1 head retune
- NRE: 6–8× the floor

- [ ] **Step 3: Record the sweep dir hash in a session note** (optional)

The v1 work-flow demonstrated that re-runs (e.g., NPE re-run after the d>1 assertion fix, LF2I re-run after the head retune) produce multiple sweep dirs that have to be union-globbed at table time. If any seed needs a re-run during v2, record both the original and re-run dir paths so Task 9 can reproduce the aggregation.

This is the v2 sweep deliverable. Task 9 integrates these numbers into manuscript §8.3.

---

## Task 9: Integrate v2 results into manuscript §8.3

**Files:**
- Modify: `cd_sbi_v7.tex` — replace existing §8.3 (single-seed table at lines ~2738–2802) with the sweep-averaged version + cross-method comparison, mirroring the §8.1 / §8.2 restructure already landed.
- Verify: `/usr/bin/pdflatex cd_sbi_v7 && /usr/bin/bibtex cd_sbi_v7 && /usr/bin/pdflatex cd_sbi_v7 && /usr/bin/pdflatex cd_sbi_v7` builds cleanly. (Use the absolute path; the conda `pdflatex` on this machine is broken — see CLAUDE.md.)

**Prerequisite:** Task 8 complete and the user has signed off on the v2 sweep results. Do not invent or paraphrase numbers; pull them from the sweep output via `paper_table_8_3(load_runs(...))` and from per-run diagnostic parquets for the CDSBI calibration table.

**Template:** The §8.1 + §8.2 updates in `cd_sbi_v7.tex` (commits `7a88c15`, `9992570`, `0c3820c`) already show the target structure. Reproduce the same seven-block shape for §8.3:

1. **Setup** — keep the current §8.3 setup (`Σ = [[1, 0.5], [0.5, 1]]`, truth `r* = L⁻¹(θ - X)`, explicit `r_2* = -0.577(θ_1 - X_1) + 1.155(θ_2 - X_2)`).
2. **Architecture** — keep the current note that the conditioning network for `r_2` must learn the cross-coupling slope `-0.577`. Drop the implicit "5,004 parameters" assumption from §8.2 (already absent in §8.3; verify).
3. **Sweep** — same paragraph as §8.1 / §8.2 adapted: 5 seeds × 4 budgets, the same evaluation sample sizes. Quote the Monte-Carlo noise floors with each diagnostic.
4. **CDSBI calibration diagnostics (medium budget)** — replace the existing single-seed §8.3 table with a sweep-averaged version. Rows to include:
   - Pivot RMSE (total)
   - Marginal PIT KS `r_1, r_2`
   - Joint Mahalanobis vs `χ²_2` KS (mean over θ_0 + max-over-θ_0 in parens)
   - Conditional PIT bulk + edge per coord
   - Jacobian recovery max residual + Frobenius-norm residual
   - Coverage error max
   All values as mean ± std across 5 seeds; noise-floor column instead of "Status".

   **Data source.** `paper_table_8_3` aggregates only the columns that get lifted into `index_row.parquet` (`coverage_error_max`, `marginal_ks`, `pivot_rmse`, `joint_mahal_ks`, `jacobian_max_residual`, `actual_params_total`). The ConditionalPIT bulk/edge rows are NOT in `index_row` and must be extracted directly from the per-run `conditional_pit.parquet` files at table-build time (same pattern as the §8.2 manuscript update — see the Step 1 extraction script that v1 Task 15 used: load each CDSBI medium-budget run's `conditional_pit.parquet`, split by `theta_0_bin ∈ {1, 2, 3}` (bulk) vs `{0, 4}` (edge) per coord, mean-of-mean across seeds).

   Keep the manuscript's commentary that pivot RMSE is the *wrong* primary diagnostic in the correlated case (the high value reflects `L⁻¹`-scaling of (θ − X), not a poor fit). The Jacobian-recovery row replaces the manuscript's current ad-hoc paragraph reporting `E[J_θ]` matches `L⁻¹` to 1–2%.
5. **Budget invariance** — short paragraph stating what the data shows. Do not assume it matches the §8.2 saturation pattern; describe what the v2 sweep actually produced.
6. **Comparison with baselines** — new 5-method × 4-budget table. Headline metric is `coverage_error_max` (matches §8.2 cross-method table semantics — joint Mahalanobis is CDSBI-only, so can't go in the cross-method headline). NPE's row needs the same "marginal-product credible region" footnote as §8.2's table; LF2I-BFF will be slightly worse than §8.2 (the correlated coupling makes the marginal-vs-joint classifier task harder), confirm against the sweep data.
7. **Synthesis** — close §8.3 with the KR-uniqueness empirical claim landed. The §8.3-specific point: (a) the recovered Jacobian matches the unique KR rearrangement (Theorem A-d, §6.3) to within a few percent on average, (b) the cross-method gap to CDSBI widens further than §8.2 — the cross-coupling slope is exactly what NLE's generic flow + MCMC gets wrong, and CDSBI's architectural (R1) + triangular structure carries the coverage cleanly. Foreshadow §8.4 (non-additive case, the (R2) ablation).

- [ ] **Step 1: Extract the §8.3 numbers**

Run, from the repo root, a script analogous to the §8.2 extraction we used in v1 Task 15. For the CDSBI calibration table, pull per-run diagnostic parquets for the 5 medium-budget CDSBI seeds and aggregate mean ± std per diagnostic. For the cross-method table, run `paper_table_8_3(load_runs(...))` over the §8.3 sweep dir(s).

Sanity-check the numbers against the manuscript's existing §8.3 values:
- pivot RMSE total: 0.27 (manuscript) vs sweep mean (expect 0.25-0.35 with similar narrative)
- marginal KS r_1, r_2: 0.010, 0.003 (manuscript) vs sweep mean (likely higher across seeds)
- joint Mahalanobis KS: 0.012 (manuscript) vs sweep mean (likely 0.02-0.03, near floor)
- coverage error: < 0.015 (manuscript) vs sweep mean (expect 0.020-0.027, at MC floor)
- `E[J_θ]` matches L⁻¹ to 1-2%: should reproduce as `jacobian_max_residual ≤ 0.03` mean across seeds

- [ ] **Step 2: Write the LaTeX edit**

In `cd_sbi_v7.tex`, replace lines ~2738–2802 (current §8.3 block, including the long explanatory paragraph about pivot RMSE being the wrong diagnostic) with the seven-block structure above, using the actual numbers from Step 1. Mirror the LaTeX patterns of §8.1 / §8.2: `longtable` for both tables (3-col for the CDSBI calibration table, 5-col for the cross-method comparison), mean ± std in math mode, noise-floor column instead of "Status", cross-references via `\S\ref{sec:10}` / `\S\ref{subsec:3.4}` / `\S\ref{sec:4}` / `\S\ref{subsec:6.3}` not unresolved labels.

The "What this validates" paragraph (lines ~2793-2802) becomes part of the Synthesis block; preserve its KR-uniqueness framing but ground it in the seed-averaged Jacobian-recovery number rather than the single-seed `0.004 / 0.049 / 0.007` matrix.

- [ ] **Step 3: Compile the manuscript**

```bash
/usr/bin/pdflatex -interaction=nonstopmode -halt-on-error cd_sbi_v7 \
  && /usr/bin/bibtex cd_sbi_v7 \
  && /usr/bin/pdflatex -interaction=nonstopmode -halt-on-error cd_sbi_v7 \
  && /usr/bin/pdflatex -interaction=nonstopmode -halt-on-error cd_sbi_v7
```

Expected: "Output written on cd_sbi_v7.pdf (48–49 pages, ...)". No "undefined reference", no "missing $ inserted", no "extra alignment tab". If the page count moves more than ±1, the table widths or column counts likely don't match the spec — re-run the structural-sanity script from the §8.1 commit history before chasing visual fixes.

- [ ] **Step 4: Structural sanity check on the new TeX**

Run the helper script we used in the §8.1 / §8.2 commits (counts `&` per row, checks column-spec match, brace balance, cross-reference resolvability) against the §8.3 region. Both new longtables should report all-cells-OK and brace-balanced.

- [ ] **Step 5: Commit**

```bash
git add cd_sbi_v7.tex
git commit -m "$(cat <<'EOF'
manuscript(8.3): v2 sweep-averaged results + cross-method comparison

Replace single-seed §8.3 numbers with the seed-averaged v2 sweep
(5 seeds × 4 budgets × 5 methods on correlated-Σ 2D Gaussian).
Adds a CDSBI calibration table including the new Jacobian-recovery
row (the §8.3-primary KR-uniqueness empirical metric) and a cross-
method comparison table on coverage_error_max, parallel to the
§8.1 / §8.2 structure already in the manuscript. NPE row carries
the same Bonferroni-conservative-credible-region footnote as §8.2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Done condition:** PDF builds in 4 passes with no warnings beyond the natbib citation pass; §8.3 in the rendered PDF reads as parallel structure to §8.1 / §8.2; the user has reviewed the substantive empirical claims (particularly the KR-uniqueness Jacobian-recovery framing).

---

## Self-Review

**1. Spec coverage** — checked against §11–§13 of the design spec (`docs/superpowers/specs/2026-05-25-cd-sbi-experiment-infrastructure-design.md`):

- v2 deliverable per §12 row "v2 §8.3": new `target/loc_gauss_2d_corr` (Task 4), `experiment/8_3_*` (Task 4), KR-uniqueness empirical evidence (Tasks 2 + 3 + 6 + Task 9 manuscript framing). ✓
- §13 forward-hook commitments preserved: `Flow.forward(θ, context)` unchanged (no new flow); `monotonicity_guarantees` unchanged (no new flow); `log_det_jac_input` unchanged; `Simulator.entropy_lower_bound` extends to correlated Σ (Task 1 implements as `½ log((2πe)^d |Σ|)`); `ConfidenceProcedure` subtypes unchanged. ✓
- §11 done-criterion analogs for v2:
  1. CLI completes deterministically → Tasks 4 + 8 demonstrate.
  2. CDSBI tolerance bands at §8.3 → Task 6 (intensive test).
  3. Matched-budget across (method, budget) → Task 5 (pre-flight check confirms v1 schedule applies).
  4. Fast tests pass <30 s → preserved (only ~10 new fast tests across Tasks 1, 2, 7).
  5. `pytest -m intensive` matches bands → Task 6 implements.

**2. Placeholder scan**: no "TBD", "TODO without code", or hand-wave steps in any task. The explicit forward-deferred items are stated by name (NPE HPD region, R2 ablation) and route to v1 addendum / v3 plan, not "future work" handwaves.

**3. Type consistency**:
- `LocationGaussian2D_corr.sigma`: `Tuple[Tuple[float, float], ...]` (nested tuple, dataclass-hashable) across Task 1 + Task 4 YAML default.
- `LocationGaussian2D_corr.r_star_jacobian() -> torch.Tensor` shape `(d, d)` consistent in Task 1 implementation + Task 2 diagnostic consumer.
- `JacobianRecovery.value`: `pd.DataFrame` with columns `{max_residual, norm_residual, passed, n_points}` — consistent across Task 2 implementation, Task 3 wiring, and Task 7 `paper_table_8_3` consumer.
- `index_row.jacobian_max_residual` (float) — consistent between Task 3 (`_write_index_row` extension) and Task 7 (`paper_table_8_3` aggregation column list).
- `experiment.jacobian_recovery_n_points` and `experiment.jacobian_recovery_tol` cfg keys — consistent across Task 3 (consumer) and Task 4 (provider).

---

## Addendum: review-driven changes already applied + caveats for execution

An independent reviewer agent surfaced 10 findings (3 BLOCKER, 4 IMPORTANT, 3 MINOR) on the v2 plan draft. The plan has been updated inline to address the following:

- **Task 2 `Diagnostic` protocol mismatch** (BLOCKER): the `Diagnostic` protocol in `base.py` did not declare the `x_per_theta` keyword that v1's `JointMahalanobis` / `Coverage` / `SetSize` already accept de facto. Added Step 1a to extend the protocol explicitly so a future static check can't flag `JacobianRecovery.__call__` as protocol-incompatible.
- **Task 2 autograd loop** (BLOCKER): the inner `torch.autograd.grad` call was tracing through the flow's parameters via the undetached `x_k` slice. At `n_points=200` and xlarge budget (~100k flow params), this would balloon graph memory unnecessarily. Fixed by `.detach()`-ing `x_k` and a comment noting the diagnostic is `∂r/∂θ` holding (X, flow weights) fixed.
- **Task 3 integration test Hydra override syntax** (IMPORTANT): `+experiment.joint_mahalanobis_n_per_theta=300` would fail with a Hydra `ConfigCompositionException` because the key is defined in `8_3_replication.yaml`. Dropped the `+` prefix for that key; kept it for `set_size_n_per_theta` (which is NOT in the YAML).
- **Task 1 shape assert semantics** (IMPORTANT): the `__post_init__` sigma-shape check used `(d_theta, d_x)` rather than `(d_theta, d_theta)`. Added an explicit `assert self.d_theta == self.d_x` and tightened the shape check to `(d_theta, d_theta)`.
- **Task 5 hidden-size drift** (MINOR): the budget pre-flight script hardcoded `cdsbi_flow_hidden_d2` values that could drift from `configs/budget/*.yaml`. Rewrote to load values from the YAMLs via OmegaConf at run time.
- **Task 6 silent skips** (IMPORTANT + MINOR): the intensive test silently skipped runs whose JM parquet was missing, and guarded the Jacobian assertion behind `if "jacobian_max_residual" in df.columns:`. Both could mask real wiring failures; tightened to fail loudly if the parquet or the column is missing.
- **Task 9 cross-method-table data source** (MINOR): clarified that `paper_table_8_3` covers only the index_row scalars and that ConditionalPIT bulk/edge rows must be extracted from per-run `conditional_pit.parquet` files (same manual extraction pattern v1 used for the §8.2 manuscript update).

The following finding was downgraded after closer inspection of the v1 repo state:

- **Task 4 sweep YAML `override /flow:`** (claimed BLOCKER, downgraded): the reviewer flagged that `override /flow: triangular_additive` in the sweep YAML would mis-route non-CDSBI methods to the TriangularAdditiveFlow. Re-reading `_build_flow` in `src/cdsbi/experiments/run.py` and confirming the existing `configs/experiment/8_2_baseline_sweep.yaml` carries the same override (and the v1 sweep ran cleanly across all 5 methods), the override is intentional: it sets the Hydra group default, and the `_build_flow` slow path resolves the per-method flow from `cfg.method.flow + cfg.budget` regardless. The v2 sweep YAML mirrors this. **Caveat for execution:** if the v2 sweep produces method-flow mismatches, the first thing to investigate is whether `method.flow` is set on the LF2I-BFF method config (it is not, by design — LF2I-BFF uses a classifier, not a flow, and `_build_flow` returns a default flow that the runner ignores). This is fragile; if it surfaces as a real issue, refactor `_build_flow` to short-circuit when the method doesn't need a flow.

Things the v1 arc taught us to watch for during v2 execution. None require pre-execution patches to the plan; flag them at review checkpoints.

- **GPU memory at LF2I-BFF xlarge** (BLOCKER-class if missed): the multivariate ray-bisection through the trained classifier was OOM-prone until v1 commit b4ba38c wrapped it in `torch.no_grad()`. The v2 sweep at xlarge will hit the same code path; the fix is already in place. If a v2 run OOMs at xlarge, the first thing to check is that `_ray_sample_set_boundary` and `_ray_sample_set_boundary_batched` in `src/cdsbi/confidence_set/procedures.py` still have `torch.no_grad()` wrapping the bisection loop.
- **NPE marginal-product credible region** (IMPORTANT, inherited from v1 addendum): the §8.3 cross-method table will show NPE worse than it should be, because the d > 1 credible region is still a Bonferroni-conservative cartesian product of per-coordinate intervals. Task 9 must footnote this the same way Task 15 of v1 did. The proper fix (true HPD region via posterior sample density thresholding) is still deferred.
- **Determinism contract on new RNG draws** (IMPORTANT): the v2 `JacobianRecovery` diagnostic uses `np.random.default_rng(0)` for its fallback sample draw (when `eval_data` doesn't have enough points). This is intentional — keeps the diagnostic deterministic and independent of the rngs.eval sub-stream that v1's BFF marginal grid grabbed. If the diagnostic ever needs to use the seeded rng stack, derive a sub-stream via `hashlib.sha256((seed, "jacobian_recovery")).digest()` to avoid the kind of draw-order contract drift the v1 reviewer flagged.
- **Stale single-seed manuscript numbers** (MINOR): the manuscript's §8.3 values (pivot RMSE 0.27, joint Mahal 0.012, Jacobian residuals 0.004/0.049/0.007) are single-seed point estimates. The v2 sweep mean ± std will probably be slightly higher on most cells with bigger error bars — that is expected and reflects honest seed-averaged reporting. Task 9 framing should be honest about this without re-litigating it.
- **LF2I-BFF correlation difficulty** (MINOR, expected): the marginal-vs-joint classifier task that LF2I-BFF's stage-1 trains on is harder under correlated Σ — the marginals look more like the joint than they do under Σ = I. Expect LF2I-BFF coverage_error_max to be slightly worse than its §8.2 value (~0.13–0.15 vs §8.2's 0.11–0.12), still flat across budgets after the v1 head retune. If LF2I-BFF shows a U-shape again, the head retune did NOT carry over and Task 5 missed it; re-investigate.
- **Manuscript "5,004 parameters total" remnant** (MINOR): the §8.3 section in the current manuscript does NOT carry this line (it's a §8.2 artifact we already removed). Verify in Task 9.
- **`pdflatex` shadowing** (MINOR): the conda `pdflatex` on this machine is broken; CLAUDE.md pins the build command to `/usr/bin/pdflatex`. Task 9 uses the absolute path explicitly. Do not regress to bare `pdflatex` in commit-cycle commands.

---

## Execution handoff

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task with two-stage review (spec compliance → code quality) after each. Same pattern that landed v1.

**2. Inline Execution** — execute tasks in this session via `superpowers:executing-plans`, batch with checkpoints.

Before either path: send the draft plan to an independent reviewer agent (mirror the v1 reviewer pass that produced the 16-finding write-up the v1 addendum integrates). The reviewer should explicitly look for (a) BLOCKER-class issues in Tasks 1–3 (the load-bearing new code), (b) test coverage gaps, (c) cross-task type consistency, (d) anything the v1 lessons-learned addendum may have missed.
