# CD-SBI (μ, σ²) — M1: Stage-A Diagnostics + Replication

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the Stage-A `(μ, σ²)` pivot end-to-end: add the `MarginalCDRecovery` diagnostic (σ²→χ² direct, μ→Student-t via nuisance marginalization), switch the coverage θ₀-grid to a 2-D product grid, add a `paper_table_mu_sigma` aggregator, and add an intensive replication test asserting the pivot reaches the entropy floor and calibrates.

**Architecture:** M0 left a trained single-index pivot that recovers `r*` (RMSE 0.10). M1 adds the inferentially-primary marginal-CD checks. The σ² marginal CD reads off `r_σ` directly (`Φ(r_σ) = 1 − F_{χ²_{n−1}}((n−1)s²/σ²)`). The μ marginal CD is **not** `Φ(r_μ)` — it requires integrating the nuisance σ out of the joint confidence density, yielding the Student-`t_{n−1}` CD. The marginalization **identity** is numerically validated against the closed-form truth (see "Validated marginalization" below); the diagnostic's grid implementation recovers it to ≈ 1e-4. All other diagnostics (`Coverage`, `JointMahalanobis`, `MarginalPIT`, `PivotRMSE`) already consume `eval_thetas_interior` and need no code change — only a config-grid change.

**Tech Stack:** PyTorch (autograd for `∂r_σ/∂log σ`), numpy/scipy (`chi2`, `t`, `norm`, `kstest`), pandas (diagnostic parquet → index-row reduction), Hydra, pytest.

**Spec:** `docs/superpowers/specs/2026-05-29-cd-sbi-unknown-mean-variance-design.md` §A.3 (Stage-A validation). M2–M4 (learned summary / Stage-B) are out of scope.

---

## Validated marginalization (the centerpiece — pin this)

For X with `n_iid` replicates, θ = (log σ, μ), the trained flow produces a joint pivot `r(θ;X) = (r_σ, r_μ)`. The induced joint confidence density over θ is `c(θ|X) = φ₂(r(θ;X))·|det ∂r/∂θ|` (pushforward of `N(0,I₂)`; integrates to 1 since `r(·;X)` is a monotone bijection). Two marginals:

- **σ² (direct).** By the autoregressive ordering σ→μ, `r_σ` depends only on `(log σ, log s²)` — **not** on μ. So the σ²-marginal CD is `H_σ(log σ;X) = Φ(r_σ)`, which at truth is uniform. Closed form: `1 − F_{χ²_{n−1}}((n−1)s²/σ²)`.

- **μ (marginalize the nuisance σ).** The joint μ-component `r_μ = √n(μ−X̄)/σ` uses the unknown σ, so it is not itself a usable marginal CD. Integrate σ out:
  ```
  H_μ(μ₀|X) = ∫_{log σ} φ(r_σ(log σ;X)) · |∂r_σ/∂log σ| · Φ(r_μ((log σ, μ₀);X)) d log σ
  ```
  (the inner μ-integral `∫_{μ'≤μ₀} φ(r_μ)·∂r_μ/∂μ dμ' = Φ(r_μ(μ₀))` because `r_μ` is monotone in μ and → ±∞). With the closed-form `r*` this equals `F_{t_{n−1}}(√n(μ₀−X̄)/s)` **exactly** as an identity; both the analytic Gauss–Hermite check (per-X residual max ≈ 1e-4) and the diagnostic's finite-difference/trapezoid grid implementation (per-X residual max ≈ 1e-4, KS-vs-uniform = 0.0144, matching the analytic-t KS) confirm it. The naive `Φ(r*_μ at truth)` is *also* uniform but is **not** the usable marginal — it secretly conditions on the true σ.

**Diagnostic implementation note.** For the *trained* pivot, evaluate on a fixed-X, swept-`log σ` grid (μ held at μ₀): get `r_σ(log σ)` and `r_μ(log σ, μ₀)` from `trained.procedure.pivot`, get `∂r_σ/∂log σ` by autograd, and Riemann-sum the formula over a wide `log σ` grid. Reference numpy prototype (closed-form, the test oracle):
```python
import numpy as np
from scipy.stats import norm, chi2, t as tdist
gh_nodes, gh_w = np.polynomial.hermite_e.hermegauss(64); gh_w = gh_w / np.sqrt(2*np.pi)
def H_mu_marginalized_closed_form(mu0, xbar, s2, n):
    p = np.clip(1.0 - norm.cdf(gh_nodes), 1e-12, 1-1e-12)
    chi_q = chi2.ppf(p, df=n-1)
    sig_u = np.sqrt((n-1)*s2[:, None] / chi_q[None, :])      # (N,K)
    arg = np.sqrt(n)*(mu0 - xbar)[:, None] / sig_u           # (N,K)
    return (norm.cdf(arg) * gh_w[None, :]).sum(1)            # (N,) == F_t(sqrt(n)(mu0-xbar)/s)
```

---

## Conventions locked for M1

- θ = (log σ, μ); features = (log s², X̄); signs `s_θ=(+1,+1)`, `s_f=(−1,−1)` (from M0).
- **Scale coord = index 0 (marginalize over it); location coord = index 1 (marginal target).** The diagnostic reads these from the simulator's `marginal_cd_spec` (added in Task 1) so it stays target-agnostic.
- Diagnostic protocol (`src/cdsbi/diagnostics/base.py`): `name: str`; `__call__(self, trained, simulator, eval_data=None, x_per_theta=None) -> DiagnosticResult`; `DiagnosticResult(name, value, passed, noise_floor, n_samples, meta)`.
- Pivot access: `trained.procedure.pivot(theta, x) -> (B, d_theta)` (a `PivotBasedProcedure`). Gate the whole diagnostic to no-op (passed=True, meta reason) when the procedure is not pivot-based or the simulator lacks `marginal_cd_spec` (mirrors `JacobianRecovery`/`PivotRMSE`).
- KS noise floor via `cdsbi.diagnostics.ks_floor.ks_noise_floor(N)`.

---

## File structure

```
NEW
  src/cdsbi/diagnostics/marginal_cd_recovery.py   # MarginalCDRecovery
  tests/unit/test_marginal_cd_recovery.py
  tests/intensive/test_replicate_mu_sigma.py

MODIFY
  src/cdsbi/simulators/normal_unknown_mean_var.py # marginal_cd_spec + analytic_marginal_cd_pit()
  src/cdsbi/experiments/run.py                    # diagnostics list (gated) + index_row reduction cols
  configs/experiment/mu_sigma_replication.yaml    # 3×3 product θ₀ grid
  src/cdsbi/analysis/paper_tables.py              # paper_table_mu_sigma
  tests/unit/test_paper_tables.py                 # paper_table_mu_sigma coverage
```

---

## Task 1: simulator marginal-CD spec + analytic reference

**Files:**
- Modify: `src/cdsbi/simulators/normal_unknown_mean_var.py`
- Test: `tests/unit/test_normal_unknown_mean_var.py`

The simulator owns the target-specific marginal-CD knowledge (the diagnostic stays generic). Add (a) a `marginal_cd_spec` property naming the scale/location coords, and (b) `analytic_marginal_cd_pit(theta_0, x)` returning the **closed-form** σ²-CD and μ-CD PIT values at truth — the oracle the trained marginalization is compared against.

- [ ] **Step 1: Write the failing test**

```python
def test_analytic_marginal_cd_pit_uniform_at_truth():
    import numpy as np
    from scipy.stats import kstest
    sim = NormalUnknownMeanVar()
    theta_0 = (np.log(1.3), 0.7)                 # (log sigma, mu)
    rng = np.random.default_rng(0)
    x = sim.sample_x_given_theta(theta_0, 4000, rng)      # (4000, n_iid)
    out = sim.analytic_marginal_cd_pit(theta_0, x)
    # sigma^2 CD = 1 - F_chi2((n-1)s2/sigma0^2); mu CD = F_t(sqrt(n)(mu0-xbar)/s)
    assert kstest(out["sigma_pit"].numpy(), "uniform").statistic < 0.04
    assert kstest(out["mu_pit"].numpy(), "uniform").statistic < 0.04

def test_marginal_cd_spec_coords():
    sim = NormalUnknownMeanVar()
    spec = sim.marginal_cd_spec
    assert spec["scale_coord"] == 0 and spec["location_coord"] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_normal_unknown_mean_var.py::test_analytic_marginal_cd_pit_uniform_at_truth -v`
Expected: FAIL — `AttributeError: ... has no attribute 'analytic_marginal_cd_pit'`.

- [ ] **Step 3: Implement on `NormalUnknownMeanVar`**

```python
    @property
    def marginal_cd_spec(self) -> dict:
        """Which coord is the scale nuisance (marginalize over) vs the location
        target, for MarginalCDRecovery. θ = (log σ, μ)."""
        return {"scale_coord": 0, "location_coord": 1}

    def analytic_marginal_cd_pit(self, theta_0, x: torch.Tensor) -> dict:
        """Closed-form marginal-CD PIT values at the true θ₀, the oracle the
        trained marginalization is checked against. Returns {sigma_pit, mu_pit},
        each (n,). σ²-CD = 1 − F_{χ²_{n−1}}((n−1)s²/σ₀²); μ-CD = F_{t_{n−1}}(√n(μ₀−X̄)/s)."""
        log_sigma0, mu0 = float(theta_0[0]), float(theta_0[1])
        sigma0 = math.exp(log_sigma0)
        xbar, s2 = self._suff_stats(x)
        xbar = xbar.squeeze(-1).detach().cpu().numpy()
        s2 = s2.squeeze(-1).detach().cpu().numpy()
        n = self.n_iid
        w0 = (n - 1) * s2 / sigma0 ** 2
        sigma_pit = 1.0 - chi2.cdf(w0, df=n - 1)
        mu_pit = t.cdf(math.sqrt(n) * (mu0 - xbar) / np.sqrt(s2), df=n - 1)
        return {
            "sigma_pit": torch.from_numpy(sigma_pit).float(),
            "mu_pit": torch.from_numpy(mu_pit).float(),
        }
```
Add `from scipy.stats import t` (alongside the existing `chi2`, `norm`) and `import numpy as np` if not present at module scope.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_normal_unknown_mean_var.py -q`
Expected: PASS (existing 5 + 2 new = 7).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/simulators/normal_unknown_mean_var.py tests/unit/test_normal_unknown_mean_var.py
git commit -m "feat(sim): marginal_cd_spec + analytic_marginal_cd_pit (χ² σ², t μ) for M1"
```

---

## Task 2: `MarginalCDRecovery` diagnostic — σ² direct branch + scaffolding

**Files:**
- Create: `src/cdsbi/diagnostics/marginal_cd_recovery.py`
- Test: `tests/unit/test_marginal_cd_recovery.py`

Build the class, gating, and the σ² branch first (μ branch in Task 3). The σ² branch: at each θ₀, evaluate the trained `r_σ` on X|θ₀, PIT = Φ(r_σ); KS vs U; also a per-X residual vs the analytic χ²-CD PIT.

- [ ] **Step 1: Write the failing test (σ² branch, closed-form pivot recovers χ²)**

```python
import numpy as np
import torch
from scipy.stats import norm


class _OraclePivotProcedure:
    """Wraps simulator.r_star as a PivotBasedProcedure-like object."""
    d_theta = 2
    def __init__(self, sim): self._sim = sim
    def pivot(self, theta, x): return self._sim.r_star(theta, x)


class _Trained:
    def __init__(self, sim): self.procedure = _OraclePivotProcedure(sim)


def _x_per_theta(sim, grid, n, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for th in grid:
        key = repr([float(v) for v in th])
        out[key] = sim.sample_x_given_theta(th, n, rng)
    return out


def test_sigma_branch_recovers_chi2_cd_with_oracle():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.diagnostics.marginal_cd_recovery import MarginalCDRecovery
    sim = NormalUnknownMeanVar()
    grid = [(np.log(0.5), -1.0), (np.log(2.0), 1.0)]
    diag = MarginalCDRecovery(theta_0_grid=grid, n_per_theta=3000)
    trained = _Trained(sim)
    res = diag(trained, sim, x_per_theta=_x_per_theta(sim, grid, 3000))
    df = res.value
    assert {"sigma_ks", "sigma_chi2_resid"}.issubset(df.columns)
    # oracle pivot → σ² PIT uniform and matches analytic χ²-CD to ~0
    assert df["sigma_ks"].max() < 0.06
    assert df["sigma_chi2_resid"].max() < 1e-3


def test_noop_when_not_pivot_based():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.diagnostics.marginal_cd_recovery import MarginalCDRecovery
    sim = NormalUnknownMeanVar()
    class _NoPivot: procedure = object()
    res = MarginalCDRecovery(theta_0_grid=[(0.0, 0.0)], n_per_theta=10)(_NoPivot(), sim)
    assert res.passed and "reason" in res.meta
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_marginal_cd_recovery.py::test_sigma_branch_recovers_chi2_cd_with_oracle -v`
Expected: FAIL — module/class does not exist.

- [ ] **Step 3: Implement (σ² branch + gating; μ columns added in Task 3)**

```python
"""MarginalCDRecovery: validate the σ² (χ²) and μ (Student-t) marginal CDs.

σ² is direct (r_σ ⟂ μ → Φ(r_σ) is the χ²-based CD). μ requires marginalizing the
σ nuisance out of the joint confidence density → Student-t_{n−1} CD (Task 3).
No-ops (passed=True) when the procedure is not pivot-based or the simulator lacks
a marginal_cd_spec.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, kstest

from cdsbi.diagnostics.base import DiagnosticResult
from cdsbi.diagnostics.ks_floor import ks_noise_floor


class MarginalCDRecovery:
    name = "marginal_cd_recovery"

    def __init__(self, theta_0_grid: Sequence, n_per_theta: int = 2000):
        self.theta_0_grid = list(theta_0_grid)
        self.n_per_theta = n_per_theta

    def _noop(self, reason: str) -> DiagnosticResult:
        return DiagnosticResult(self.name, value=pd.DataFrame(), passed=True,
                                noise_floor=0.0, n_samples=0, meta={"reason": reason})

    def __call__(self, trained, simulator, eval_data=None, x_per_theta=None) -> DiagnosticResult:
        proc = getattr(trained, "procedure", None)
        if proc is None or not hasattr(proc, "pivot"):
            return self._noop("procedure is not pivot-based")
        spec = getattr(simulator, "marginal_cd_spec", None)
        if spec is None or not hasattr(simulator, "analytic_marginal_cd_pit"):
            return self._noop("simulator has no marginal_cd_spec")
        sc = spec["scale_coord"]
        floor = ks_noise_floor(self.n_per_theta)
        rows = []
        for theta_0 in self.theta_0_grid:
            key = repr([float(v) for v in theta_0])
            x = x_per_theta[key] if x_per_theta and key in x_per_theta else \
                simulator.sample_x_given_theta(theta_0, self.n_per_theta, np.random.default_rng(0))
            n = x.shape[0]
            theta_t = torch.tensor([[float(v) for v in theta_0]], dtype=x.dtype).expand(n, -1)
            with torch.no_grad():
                r = proc.pivot(theta_t, x)
            sigma_pit = norm.cdf(r[:, sc].detach().cpu().numpy())
            analytic = simulator.analytic_marginal_cd_pit(theta_0, x)
            sigma_ks = float(kstest(sigma_pit, "uniform").statistic)
            sigma_chi2_resid = float(np.abs(sigma_pit - analytic["sigma_pit"].numpy()).max())
            rows.append({
                "theta_0": key, "sigma_ks": sigma_ks,
                "sigma_chi2_resid": sigma_chi2_resid, "noise_floor": floor,
            })
        df = pd.DataFrame(rows)
        passed = bool((df["sigma_ks"] <= 2.0 * floor).all())
        return DiagnosticResult(self.name, value=df, passed=passed,
                                noise_floor=floor, n_samples=self.n_per_theta,
                                meta={"scale_coord": sc})
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_marginal_cd_recovery.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/diagnostics/marginal_cd_recovery.py tests/unit/test_marginal_cd_recovery.py
git commit -m "feat(diag): MarginalCDRecovery σ² branch + gating (χ² direct CD)"
```

---

## Task 3: `MarginalCDRecovery` — μ Student-t marginalization branch

**Files:**
- Modify: `src/cdsbi/diagnostics/marginal_cd_recovery.py`
- Test: `tests/unit/test_marginal_cd_recovery.py`

Add the nuisance-marginalization. For each θ₀ and each X|θ₀, marginalize the trained joint CD over the scale coord on a `log σ` grid (autograd `∂r_σ/∂log σ`), evaluate `H_μ(μ₀|X)`, KS vs U, and a per-X residual vs the analytic t-CD.

- [ ] **Step 1: Write the failing test (μ branch, closed-form pivot recovers t)**

```python
def test_mu_branch_recovers_student_t_with_oracle():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.diagnostics.marginal_cd_recovery import MarginalCDRecovery
    sim = NormalUnknownMeanVar()
    grid = [(np.log(0.5), -1.0), (np.log(2.0), 1.0)]
    diag = MarginalCDRecovery(theta_0_grid=grid, n_per_theta=2000)
    trained = _Trained(sim)
    res = diag(trained, sim, x_per_theta=_x_per_theta(sim, grid, 2000))
    df = res.value
    assert {"mu_ks", "mu_t_resid"}.issubset(df.columns)
    # oracle pivot, marginalized over σ, must recover the t-CD closely
    assert df["mu_ks"].max() < 0.06
    assert df["mu_t_resid"].max() < 0.02     # grid-marginalization vs exact t
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_marginal_cd_recovery.py::test_mu_branch_recovers_student_t_with_oracle -v`
Expected: FAIL — `KeyError: 'mu_ks'`.

- [ ] **Step 3: Implement the μ marginalization (finite-difference, device-safe)**

Add a helper and extend `__call__`. The marginalization grids `log σ` over a wide band (cover the prior + tails), evaluates the trained pivot at `(log σ_grid, μ₀)` for each X, computes `∂r_σ/∂log σ` by **finite differences along the grid** (`torch.gradient`), and Riemann-sums `Σ_k φ(r_σ,k)·|∂r_σ/∂log σ_k|·Φ(r_μ,k)·Δ`.

**Why finite differences, not autograd:** `cd_sbi.py`'s `pivot_fn` does `theta = theta.to(device)` internally, so a `requires_grad_` leaf built outside is detached from the graph on GPU (`theta.to('cuda')` returns a new non-leaf) — `torch.autograd.grad` then raises "does not require grad". Finite differences over the 257-pt grid is the literal discretization of the validated integral, runs entirely under `torch.no_grad()` (device- and memory-safe), and recovers the t-CD to residual ≈ 1e-4 (verified vs the closed-form). It also lets us chunk X rows trivially.

```python
    def _marginalize_mu(self, proc, simulator, theta_0, x, sc, lc, chunk=256):
        """H_μ(μ₀|X) for each row of x, by integrating the joint CD over log σ.
        Finite-difference ∂r_σ/∂logσ on a fixed grid; no autograd (pivot_fn moves
        devices, which would detach an external grad leaf on GPU)."""
        mu0 = float(theta_0[lc])
        lo, hi = simulator.log_sigma_range
        grid = torch.linspace(lo - 2.0, hi + 2.0, 257, dtype=x.dtype, device=x.device)  # (K,)
        K = grid.shape[0]
        out = []
        with torch.no_grad():
            for start in range(0, x.shape[0], chunk):
                xb = x[start:start + chunk]                       # (b, n_iid)
                b = xb.shape[0]
                theta = torch.empty(b, K, 2, dtype=x.dtype, device=x.device)
                theta[:, :, sc] = grid.view(1, K)
                theta[:, :, lc] = mu0
                theta = theta.reshape(b * K, 2)
                x_rep = xb.repeat_interleave(K, dim=0)            # (b*K, n_iid)
                r = proc.pivot(theta, x_rep)                      # (b*K, 2)
                r_sigma = r[:, sc].reshape(b, K)
                r_mu = r[:, lc].reshape(b, K)
                # finite-diff dr_σ/dlogσ along the grid axis
                dr = torch.gradient(r_sigma, spacing=(grid,), dim=1)[0]   # (b, K)
                phi = torch.exp(-0.5 * r_sigma ** 2) / np.sqrt(2 * np.pi)
                Phi_mu = 0.5 * (1.0 + torch.erf(r_mu / np.sqrt(2.0)))
                # trapezoid in logσ via torch.trapezoid (handles the Δ exactly)
                integ = phi * dr.abs() * Phi_mu                   # (b, K)
                H = torch.trapezoid(integ, grid, dim=1)           # (b,)
                out.append(H.detach().cpu().numpy())
        return np.concatenate(out)
```

In `__call__`, after computing the σ² metrics for a θ₀, add:
```python
            lc = spec["location_coord"]
            H_mu = np.clip(self._marginalize_mu(proc, simulator, theta_0, x, sc, lc), 0.0, 1.0)
            mu_ks = float(kstest(H_mu, "uniform").statistic)
            mu_t_resid = float(np.abs(H_mu - analytic["mu_pit"].numpy()).max())
```
and extend the appended row dict with `"mu_ks": mu_ks, "mu_t_resid": mu_t_resid`. Update `passed` to also require `(df["mu_ks"] <= 2.0 * floor).all()`. The σ² `proc.pivot` call in `__call__` keeps its `torch.no_grad()` — no conflict, since `_marginalize_mu` owns its own `no_grad` block and uses no autograd.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_marginal_cd_recovery.py -v`
Expected: PASS (3 tests). The `mu_t_resid < 0.02` confirms the grid marginalization matches the exact t.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/diagnostics/marginal_cd_recovery.py tests/unit/test_marginal_cd_recovery.py
git commit -m "feat(diag): MarginalCDRecovery μ branch — Student-t via σ-nuisance marginalization"
```

---

## Task 4: 2-D product coverage θ₀-grid

**Files:**
- Modify: `configs/experiment/mu_sigma_replication.yaml`

Replace the 5-point grid with a 3×3 product grid over (log σ, μ), addressing the "fixed 5-point grid" caveat (CLAUDE.md). The grid covers the heterogeneous (scale, location) box. Diagnostics consume it unchanged (`run.py` passes `list(cfg.experiment.eval_thetas_interior)` to `Coverage`, `JointMahalanobis`, `ConditionalPIT`, and the new `MarginalCDRecovery`).

- [ ] **Step 1: Edit the config**

```yaml
  # 3×3 product grid over (log σ, μ): log σ ∈ {log 0.5, 0, log 2}, μ ∈ {−2, 0, 2}.
  # Replaces the prior 5-point grid (addresses the fixed-5-point θ₀ caveat).
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
```

- [ ] **Step 2: Verify config composes + grid length**

Run:
```bash
python -c "
from hydra import initialize, compose
with initialize(version_base=None, config_path='configs'):
    cfg = compose(config_name='config', overrides=['experiment=mu_sigma_replication'])
print(len(cfg.experiment.eval_thetas_interior))
"
```
Expected: `9`.

- [ ] **Step 3: Commit**

```bash
git add configs/experiment/mu_sigma_replication.yaml
git commit -m "config: 2-D 3×3 product θ₀ grid for (μ,σ²) coverage (M1)"
```

---

## Task 5: wire `MarginalCDRecovery` into the runner + index-row reduction

**Files:**
- Modify: `src/cdsbi/experiments/run.py`

Add the diagnostic to the runner's diagnostic list (it self-gates to no-op when inapplicable, so it is safe for all experiments), and reduce its parquet to scalar `index_row` columns for the table + intensive test.

- [ ] **Step 1: Add to the diagnostics list**

In the diagnostics-construction block (where `Coverage`, `JointMahalanobis`, etc. are appended), add:
```python
        ("marginal_cd_recovery", MarginalCDRecovery(
            theta_0_grid=list(cfg.experiment.eval_thetas_interior),
            n_per_theta=int(cfg.experiment.n_eval_per_theta),
        )),
```
and add the import at the top with the other diagnostics:
```python
from cdsbi.diagnostics.marginal_cd_recovery import MarginalCDRecovery
```

Also add `"marginal_cd_recovery"` to the `x_sharing_names` set (`run.py:397`) so the diagnostic reuses the pre-drawn X|θ₀ (shared with `coverage`/`joint_mahalanobis`) instead of re-sampling — this makes the X comparable across diagnostics and the `repr([float(v) for v in theta_0])` key lookup load-bearing (it matches run.py's `str(list(map(float, list(theta_0))))`, verified identical):
```python
    x_sharing_names = {"coverage", "set_size", "joint_mahalanobis", "marginal_cd_recovery"}
```

- [ ] **Step 2: Add index-row reduction**

In `_write_index_row`, after the `jacobian_recovery` reduction block, add:
```python
    mcd_path = rd.path / "diagnostics" / "marginal_cd_recovery.parquet"
    if mcd_path.exists():
        mcd_df = pd.read_parquet(mcd_path)
        for col, out in [("sigma_ks", "marginal_cd_sigma_ks"),
                         ("mu_ks", "marginal_cd_mu_ks"),
                         ("mu_t_resid", "marginal_cd_mu_t_resid")]:
            if col in mcd_df.columns and len(mcd_df):
                row[out] = float(mcd_df[col].mean())
```
(The diagnostic writes an empty DataFrame when it no-ops, so the `len(mcd_df)` guard keeps these columns absent for non-(μ,σ²) experiments — matching how `joint_mahal_ks` is conditionally present.)

- [ ] **Step 3: Verify the diagnostic writes its parquet (fast smoke, tiny budget)**

Run:
```bash
python -m cdsbi.experiments.run experiment=mu_sigma_replication seed=0 \
  budget=small training.n_steps=50 hydra.run.dir=/tmp/m1_smoke 2>&1 | tail -3
python -c "
import pandas as pd
df = pd.read_parquet('/tmp/m1_smoke/diagnostics/marginal_cd_recovery.parquet')
print(sorted(df.columns)); print(len(df), 'rows')
row = pd.read_parquet('/tmp/m1_smoke/index_row.parquet')
print([c for c in row.columns if c.startswith('marginal_cd')])
"
```
Expected: parquet has `mu_ks, mu_t_resid, sigma_chi2_resid, sigma_ks, ...`, 9 rows; index_row has the three `marginal_cd_*` columns. (Loss is meaningless at 50 steps — this only checks plumbing.)

- [ ] **Step 4: Commit**

```bash
git add src/cdsbi/experiments/run.py
git commit -m "feat(run): wire MarginalCDRecovery + reduce to index_row columns"
```

---

## Task 6: `paper_table_mu_sigma` aggregator

**Files:**
- Modify: `src/cdsbi/analysis/paper_tables.py`
- Test: `tests/unit/test_paper_tables.py`

- [ ] **Step 1: Write the failing test**

```python
def test_paper_table_mu_sigma_aggregates():
    import pandas as pd
    from cdsbi.analysis.paper_tables import paper_table_mu_sigma
    df = pd.DataFrame([
        {"method": "cd_sbi", "budget_name": "medium", "coverage_error_max": 0.02,
         "pivot_rmse": 0.10, "joint_mahal_ks": 0.03, "marginal_cd_sigma_ks": 0.02,
         "marginal_cd_mu_ks": 0.03, "marginal_cd_mu_t_resid": 0.01,
         "final_loss": 0.99, "actual_params_total": 12000},
        {"method": "cd_sbi", "budget_name": "medium", "coverage_error_max": 0.03,
         "pivot_rmse": 0.11, "joint_mahal_ks": 0.04, "marginal_cd_sigma_ks": 0.03,
         "marginal_cd_mu_ks": 0.04, "marginal_cd_mu_t_resid": 0.012,
         "final_loss": 1.00, "actual_params_total": 12000},
    ])
    tbl = paper_table_mu_sigma(df)
    assert ("cd_sbi", "medium") in tbl.index
    assert "marginal_cd_mu_ks_mean" in tbl.columns
    assert abs(tbl.loc[("cd_sbi", "medium"), "coverage_error_max_mean"] - 0.025) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_paper_tables.py::test_paper_table_mu_sigma_aggregates -v`
Expected: FAIL — `ImportError: cannot import name 'paper_table_mu_sigma'`.

- [ ] **Step 3: Implement (mirror `paper_table_8_4`)**

```python
def paper_table_mu_sigma(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged (μ,σ²) Stage-A table: coverage, pivot RMSE, joint Mahalanobis,
    marginal-CD recovery (σ² χ² + μ t), entropy-floor loss."""
    metrics = [
        "coverage_error_max", "pivot_rmse", "joint_mahal_ks",
        "marginal_cd_sigma_ks", "marginal_cd_mu_ks", "marginal_cd_mu_t_resid",
        "final_loss", "actual_params_total",
    ]
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby(["method", "budget_name"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_paper_tables.py -q`
Expected: PASS (existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/analysis/paper_tables.py tests/unit/test_paper_tables.py
git commit -m "feat(analysis): paper_table_mu_sigma aggregator (M1)"
```

---

## Task 7: intensive replication test

**Files:**
- Create: `tests/intensive/test_replicate_mu_sigma.py`

Mirror `test_replicate_8_4.py` (entropy floor) + `test_replicate_8_3.py` (coverage/JM aggregates), adapted to the (μ,σ²) target and the new marginal-CD metrics. 5 seeds, subprocess per seed, load via `load_runs`.

- [ ] **Step 1: Write the test**

```python
"""Intensive: (μ,σ²) Stage-A replication — pivot recovers, calibrates, reaches floor."""
import glob
import os
import subprocess

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.intensive
def test_replicate_mu_sigma_stage_a(tmp_path):
    out_dir = tmp_path / "mu_sigma"
    for seed in range(5):
        cmd = [
            "python", "-m", "cdsbi.experiments.run",
            f"hydra.run.dir={out_dir}/run_seed_{seed}",
            "experiment=mu_sigma_replication", "budget=medium",
            f"seed={seed}", "training.fresh_batch=false",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        assert r.returncode == 0, f"seed {seed} failed:\n{r.stderr[-2000:]}"

    from cdsbi.analysis.loaders import load_runs
    df = load_runs(str(out_dir / "*"))
    assert len(df) == 5

    # (a) pivot recovery + coverage (single-index flow, 2-D)
    assert df["pivot_rmse"].mean() <= 0.30
    assert df["coverage_error_max"].mean() <= 0.05

    # (b) marginal-CD recovery: σ² (χ²) and μ (Student-t) both calibrate
    assert df["marginal_cd_sigma_ks"].mean() <= 0.06
    assert df["marginal_cd_mu_ks"].mean() <= 0.06
    assert df["marginal_cd_mu_t_resid"].mean() <= 0.05    # trained marginalization ≈ exact t

    # (c) entropy floor — final NF-MLE loss must not sink below the floor (R2 sanity)
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    H = NormalUnknownMeanVar().entropy_lower_bound()
    assert df["final_loss"].mean() > H - 0.10, (
        f"final_loss {df['final_loss'].mean():.3f} below entropy floor {H:.3f}"
    )

    # (d) per-run JointMahalanobis at 2× floor on ≥4/5 seeds
    n_pass = 0
    for rd in glob.glob(str(out_dir / "*")):
        jm = os.path.join(rd, "diagnostics", "joint_mahalanobis.parquet")
        if not os.path.exists(jm):
            continue
        jmdf = pd.read_parquet(jm)
        if (jmdf["ks"] <= 2.0 * jmdf["noise_floor"]).all():
            n_pass += 1
    assert n_pass >= 4, f"only {n_pass}/5 seeds pass JointMahalanobis"
```

- [ ] **Step 2: Run it (intensive; minutes on GPU)**

Run: `pytest tests/intensive/test_replicate_mu_sigma.py -v -s -m intensive`
Expected: PASS. If `pivot_rmse`/`coverage`/`marginal_cd_*` bands are tight, inspect the printed values and the run dirs before loosening — a real miss may indicate the single-index flow needs the longer recipe (raise `training.n_steps` in the experiment config, not the tolerance) or a budget-calibrated `hidden` (the d=2 budget-calibration follow-on noted in M0).

- [ ] **Step 3: Confirm fast suite still green**

Run: `pytest -q`
Expected: all fast tests pass; the new intensive test is deselected.

- [ ] **Step 4: Commit**

```bash
git add tests/intensive/test_replicate_mu_sigma.py
git commit -m "test(intensive): (μ,σ²) Stage-A replication — recovery + marginal-CD + floor"
```

---

## Self-review

- **Spec coverage (§A.3):** PivotRMSE/MarginalPIT/JointMahalanobis already exist and consume the grid (no change) ✓; `MarginalCDRecovery` σ²-direct + μ-Student-t (Tasks 1–3) ✓; 2-D product coverage grid (Task 4) ✓; runner wiring + table (Tasks 5–6) ✓; intensive replication with entropy-floor + calibration (Task 7) ✓. `JacobianRecovery` correctly no-ops (no `r_star_jacobian`) — spec confirms this; nothing to do.
- **Placeholder scan:** all steps carry runnable code + exact commands + expected output. No TBD.
- **Type consistency:** the diagnostic uses the `Diagnostic` protocol `(trained, simulator, eval_data, x_per_theta) -> DiagnosticResult`; `marginal_cd_spec` keys (`scale_coord`/`location_coord`) and `analytic_marginal_cd_pit` keys (`sigma_pit`/`mu_pit`) are referenced identically in Tasks 1/2/3; index-row columns (`marginal_cd_sigma_ks`/`marginal_cd_mu_ks`/`marginal_cd_mu_t_resid`) match between Task 5 (write) and Tasks 6/7 (read).
- **Risk — μ marginalization grid width/resolution.** `_marginalize_mu` uses a 257-pt `log σ` grid over `[lo−2, hi+2]`; the oracle test (`mu_t_resid < 0.02`) guards this. If a trained pivot's `r_σ` saturates outside the band, the integral underflows — the test would catch it. The finite-difference implementation runs under `torch.no_grad()` and chunks X rows (`chunk=256`), so memory is bounded (no autograd backward graph) — this resolves the GPU-autograd break and the OOM concern the plan review raised. Device-safe because the grid/theta tensors are built on `x.device`.
- **Known follow-on (M2):** unchanged from M0 — `fit()` optimizer extension for the learned `DeepSetsConditioner` is M2, not needed here.
