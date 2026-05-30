# CD-SBI bivariate-normal (μ, Σ) — N2: Stage-A diagnostics + replication

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Stage-A validation for the bivariate-normal (μ,Σ) target: the deferred `entropy_lower_bound` (the NF-MLE floor), the marginal-CD recovery generalized to d=5 (three covariance χ² marginals + the **joint Hotelling-T²** μ-marginal), an LHS coverage grid, a `paper_table_mu_cov`, and an intensive replication — plus tightening the μ-coordinate recovery the N1 smoke flagged.

**Architecture:** All three new math pieces were de-risked by an N2 prototype: (a) the NF-MLE floor uses the **lower-triangular** 5×5 feature-Jacobian (`r₁←feat₁; r₂←feat₂; r₃←feat₁,feat₃; r₄←feat₄; r₅←feat₄,feat₅`), so `log|det| = Σₖ log|∂rₖ/∂featₖ|` (5 diagonal terms; MC floor ≈ 2.03); (b) the analytic Hotelling-T² μ-PIT is uniform at truth (KS 0.008); (c) the joint CD marginalizes to Hotelling — validated by sampling `r~N(0,I₅)`, inverting the (monotone, autoregressive) pivot to θ-samples, and checking the μ-block's Hotelling stat ~ F (KS 0.004). The trained-pivot μ-recovery reuses that: a **vectorized autoregressive inverter** (each flow coord is monotone in θₖ given ctx → per-coord bisection).

**Tech Stack:** PyTorch (vectorized bisection), numpy/scipy (`chi2`, `f`, `norm`, `kstest`, LHS via `scipy.stats.qmc`), Hydra, pytest.

**Spec:** `docs/superpowers/specs/2026-05-30-cd-sbi-bivariate-normal-mu-cov-design.md` §4 + §5. **Removes the N1 guard-rail** (adding `entropy_lower_bound` makes `FloorIntegrity` + the full `run.py` work for `mu_cov_replication`).

---

## File structure

```
NEW
  src/cdsbi/flows/invert.py                              # autoregressive_invert(flow, r, context)
  src/cdsbi/diagnostics/multivariate_marginal_cd.py      # MultivariateMarginalCDRecovery
  tests/unit/test_autoregressive_invert.py
  tests/unit/test_multivariate_marginal_cd.py
  tests/intensive/test_replicate_mu_cov.py

MODIFY
  src/cdsbi/simulators/normal_bivariate_unknown_cov.py   # entropy_lower_bound + analytic_marginal_cd_pit
  src/cdsbi/analysis/paper_tables.py                     # paper_table_mu_cov
  src/cdsbi/experiments/run.py                           # wire MultivariateMarginalCDRecovery + index_row
  configs/experiment/mu_cov_replication.yaml             # LHS θ₀ grid + a training recipe (μ-tightening)
  tests/unit/test_normal_bivariate_unknown_cov.py        # entropy + analytic-PIT tests
```

No change to `SingleIndexMonotoneFlow`, `BartlettSummaryConditioner`, or the existing diagnostics.

---

## Task 1: `entropy_lower_bound` (triangular 5×5 floor)

**Files:**
- Modify: `src/cdsbi/simulators/normal_bivariate_unknown_cov.py`
- Test: `tests/unit/test_normal_bivariate_unknown_cov.py` (append)

The NF-MLE floor `E[½‖r*‖² + (d/2)log2π − log|∂r*/∂feat|]`; the feature-Jacobian is lower-triangular so `log|det| = Σₖ log|∂rₖ/∂featₖ|` (the 5 diagonal terms, derived in the prototype). MC estimate ≈ 2.03.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_entropy_lower_bound_is_finite_and_stable():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    sim = NormalBivariateUnknownCov()
    H = sim.entropy_lower_bound(n_mc=40000, seed=0)
    assert 1.5 < H < 2.5, f"entropy floor {H:.3f} outside expected ~2.0"   # prototype MC ≈ 2.03
    # stable across seeds (MC noise small)
    H2 = sim.entropy_lower_bound(n_mc=40000, seed=1)
    assert abs(H - H2) < 0.05
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_normal_bivariate_unknown_cov.py::test_entropy_lower_bound_is_finite_and_stable -v` → FAIL (no `entropy_lower_bound`).

- [ ] **Step 3: Implement (vectorized MC; uses the existing `_chol`/`_bartlett` + numpy bridge)**

```python
    def entropy_lower_bound(self, n_mc: int = 50000, seed: int = 42) -> float:
        """MC estimate of E[NF-MLE loss at r*] on the (θ, oracle-feature) scale.
        The feature-Jacobian ∂r*/∂feat is lower-triangular ⇒ log|det| = Σₖ log|∂rₖ/∂featₖ|:
          r₁: 2 w₁ f_{χ²_{n−1}}(w₁)/φ(r₁),  w₁=T₁₁²    (feat₁ = log D₁₁)
          r₂: 2 w₂ f_{χ²_{n−2}}(w₂)/φ(r₂),  w₂=T₂₂²    (feat₂ = log D₂₂)
          r₃: 1/C₂₂ (feat₃ = D₂₁);  r₄: √n/C₁₁ (feat₄ = X̄₁);  r₅: √n/C₂₂ (feat₅ = X̄₂).
        """
        rng = np.random.default_rng(seed)
        theta_np = self._draw_theta(n_mc, rng)
        x = self._sample_x(theta_np, rng).reshape(n_mc, self.d_x)
        theta = torch.from_numpy(theta_np).float()
        r = self.r_star(theta, torch.from_numpy(x).float()).numpy()       # (n_mc,5)
        xbar, D = self._bartlett(torch.from_numpy(x).float())
        D11 = D[:, 0, 0].numpy(); D22 = D[:, 1, 1].numpy()
        C11 = np.exp(theta_np[:, 0]); C22 = np.exp(theta_np[:, 1])
        n = self.n_iid
        w1 = (D11 / C11) ** 2; w2 = (D22 / C22) ** 2
        ld1 = np.log(2 * w1 * chi2.pdf(w1, n - 1) / np.clip(norm.pdf(r[:, 0]), 1e-30, None))
        ld2 = np.log(2 * w2 * chi2.pdf(w2, n - 2) / np.clip(norm.pdf(r[:, 1]), 1e-30, None))
        ld3 = np.log(1.0 / C22); ld4 = np.log(math.sqrt(n) / C11); ld5 = np.log(math.sqrt(n) / C22)
        logdet = ld1 + ld2 + ld3 + ld4 + ld5
        loss = 0.5 * (r ** 2).sum(axis=1) + 0.5 * self.d_theta * math.log(2 * math.pi) - logdet
        return float(loss.mean())
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_normal_bivariate_unknown_cov.py -q` → all pass (existing 5 + 1 new); `entropy_lower_bound` ≈ 2.03.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/simulators/normal_bivariate_unknown_cov.py tests/unit/test_normal_bivariate_unknown_cov.py
git commit -m "feat(sim): entropy_lower_bound (triangular 5×5 feature-Jacobian floor, ≈2.03)"
```

---

## Task 2: `analytic_marginal_cd_pit` (3 covariance χ² + joint Hotelling-T² μ)

**Files:**
- Modify: `src/cdsbi/simulators/normal_bivariate_unknown_cov.py`
- Test: `tests/unit/test_normal_bivariate_unknown_cov.py` (append)

The reference PITs at truth: three SCALAR covariance PITs (= `Φ(r*₁), Φ(r*₂), Φ(r*₃)` — the χ²/χ²/N Bartlett pivots) and one JOINT Hotelling-T² μ-PIT. All uniform at truth.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_analytic_marginal_cd_pit_uniform_at_truth():
    import numpy as np
    from scipy.stats import kstest
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    sim = NormalBivariateUnknownCov()
    theta0 = (0.2, -0.1, 0.4, 1.0, -1.5)
    x = sim.sample_x_given_theta(theta0, 8000, np.random.default_rng(2))
    out = sim.analytic_marginal_cd_pit(theta0, x)
    for key in ("cov1_pit", "cov2_pit", "cov3_pit", "mu_hotelling_pit"):
        assert out[key].shape == (8000,)
        assert kstest(out[key].numpy(), "uniform").statistic < 0.04, f"{key} not uniform"
```

- [ ] **Step 2: Run to verify failure** → FAIL (no `analytic_marginal_cd_pit`).

- [ ] **Step 3: Implement**

```python
    def analytic_marginal_cd_pit(self, theta_0, x: torch.Tensor) -> dict:
        """Reference marginal-CD PITs at the true θ₀ (each ~ U at truth):
        three scalar covariance PITs (the Bartlett χ²/χ²/N pivots = Φ(r*₁..₃)) and one
        JOINT Hotelling-T² μ-PIT: F_{p, n−p}( T² (n−p)/(p(n−1)) ), T²=n(μ₀−X̄)ᵀS⁻¹(μ₀−X̄)."""
        from scipy.stats import f as _f, norm as _norm
        tv = np.atleast_1d(np.asarray(theta_0, dtype=np.float64))
        n = self.n_iid; p = self.p
        theta = torch.tensor(tv, dtype=x.dtype).expand(x.shape[0], self.d_theta)
        r = self.r_star(theta, x).numpy()                            # (m,5)
        cov_pits = _norm.cdf(r[:, :3])                               # Φ(r*₁..₃) ~ U
        # joint Hotelling on the SAMPLE covariance S (unbiased, ddof=1)
        obs = x.reshape(x.shape[0], n, p)
        xbar = obs.mean(dim=1).numpy()
        Xc = (obs - obs.mean(dim=1, keepdim=True)).numpy()
        S = np.einsum('mki,mkj->mij', Xc, Xc) / (n - 1)              # (m,2,2)
        d = tv[3:5][None, :] - xbar                                  # (m,2)
        T2 = n * np.einsum('mi,mij,mj->m', d, np.linalg.inv(S), d)
        F = T2 * (n - p) / (p * (n - 1))
        mu_pit = _f.cdf(F, p, n - p)
        return {
            "cov1_pit": torch.from_numpy(cov_pits[:, 0]).float(),
            "cov2_pit": torch.from_numpy(cov_pits[:, 1]).float(),
            "cov3_pit": torch.from_numpy(cov_pits[:, 2]).float(),
            "mu_hotelling_pit": torch.from_numpy(mu_pit).float(),
        }
```

- [ ] **Step 4: Run to verify pass** → all 4 PITs uniform (KS < 0.04).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/simulators/normal_bivariate_unknown_cov.py tests/unit/test_normal_bivariate_unknown_cov.py
git commit -m "feat(sim): analytic_marginal_cd_pit (3 covariance χ² + joint Hotelling-T² μ)"
```

---

## Task 3: autoregressive pivot inverter

**Files:**
- Create: `src/cdsbi/flows/invert.py`
- Test: `tests/unit/test_autoregressive_invert.py`

Each `SingleIndexMonotoneFlow` coordinate `rₖ = Gₖ(s_θk·…·θₖ + …; ctx)` is strictly monotone in `θₖ` (sign `s_θk`) and depends only on `θ_{≤k}` (ctx = θ_{<k}, feat). So invert autoregressively: solve `θₖ` by 1-D bisection given `θ_{<k}` already solved. Vectorized over the batch.

- [ ] **Step 1: Write the failing test**

```python
"""autoregressive_invert: forward∘invert ≈ identity for SingleIndexMonotoneFlow."""
import torch


def test_invert_recovers_theta():
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.flows.invert import autoregressive_invert
    torch.manual_seed(0)
    flow = SingleIndexMonotoneFlow(d=5, theta_signs=[1, 1, -1, 1, 1],
                                   feat_signs=[-1, -1, 1, -1, -1], hidden=16).eval()
    theta = torch.empty(64, 5).uniform_(-2, 2)
    feat = torch.randn(64, 5)
    with torch.no_grad():
        r, _ = flow.forward(theta, context=feat)
        theta_rec = autoregressive_invert(flow, r, feat, lo=-6.0, hi=6.0, iters=40)
    assert torch.allclose(theta_rec, theta, atol=1e-2), \
        f"max err {(theta_rec - theta).abs().max():.4f}"
```

- [ ] **Step 2: Run to verify failure** → FAIL (module missing).

- [ ] **Step 3: Implement `src/cdsbi/flows/invert.py`**

```python
"""autoregressive_invert: invert a SingleIndexMonotoneFlow coordinate-by-coordinate.

Each coordinate r_k is strictly monotone in θ_k (sign = flow._s_theta[k]) and depends
only on θ_{≤k}, so we solve θ_k by bisection given θ_{<k} (already solved). Vectorized
over the batch; used to SAMPLE the CD (draw r~N(0,I), invert → θ-samples).
"""
from __future__ import annotations

import torch


def autoregressive_invert(flow, r_target: torch.Tensor, context: torch.Tensor,
                          lo: float = -8.0, hi: float = 8.0, iters: int = 40) -> torch.Tensor:
    """r_target (n,d), context (n,d) → θ (n,d) with flow.forward(θ, context)[:, k] ≈ r_target[:, k]."""
    n, d = r_target.shape
    theta = torch.zeros(n, d, dtype=r_target.dtype, device=r_target.device)
    for k in range(d):
        sign = float(flow._s_theta[k])                 # +1: r increases in θ_k; −1: decreases
        lo_k = torch.full((n,), lo, dtype=r_target.dtype, device=r_target.device)
        hi_k = torch.full((n,), hi, dtype=r_target.dtype, device=r_target.device)
        for _ in range(iters):
            mid = 0.5 * (lo_k + hi_k)
            theta[:, k] = mid
            rk = flow.forward(theta, context=context)[0][:, k]
            # increasing: if rk < target, raise lo; decreasing: if rk < target, lower hi
            below = rk < r_target[:, k]
            if sign > 0:
                lo_k = torch.where(below, mid, lo_k); hi_k = torch.where(below, hi_k, mid)
            else:
                hi_k = torch.where(below, mid, hi_k); lo_k = torch.where(below, lo_k, mid)
        theta[:, k] = 0.5 * (lo_k + hi_k)
    return theta
```

- [ ] **Step 4: Run to verify pass** → PASS (max err < 1e-2).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/flows/invert.py tests/unit/test_autoregressive_invert.py
git commit -m "feat(flows): autoregressive_invert — per-coord bisection inverse of SingleIndexMonotoneFlow"
```

---

## Task 4: `MultivariateMarginalCDRecovery` diagnostic

**Files:**
- Create: `src/cdsbi/diagnostics/multivariate_marginal_cd.py`
- Test: `tests/unit/test_multivariate_marginal_cd.py`

Two parts: (i) the **three covariance direct PITs** — `Φ(r_k(θ₀;X))` for k=1,2,3 at truth, KS vs U + residual vs the analytic χ²/N reference (cheap, the direct d=5 analog of the 1-D σ² check); (ii) the **μ Hotelling-recovery** — sample `r~N(0,I₅)` per dataset, invert (Task 3) → θ-samples, take the μ-block, compute its Hotelling F-stat vs the dataset's `S`, KS-compare the pooled stats to `F_{p,n−p}`. Gated to no-op without `marginal_cd_spec`/`encode_fn`.

- [ ] **Step 1: Write the failing tests**

```python
"""MultivariateMarginalCDRecovery: covariance direct PITs + μ Hotelling-recovery."""
import numpy as np
import torch


class _OracleProc:
    d_theta = 5
    def __init__(self, sim):
        self._sim = sim
        self.encode_fn = lambda x: sim.oracle_summary(x)
    def pivot(self, theta, x): return self._sim.r_star(theta, x)


class _Trained:
    def __init__(self, sim): self.procedure = _OracleProc(sim)


def test_covariance_direct_pits_uniform_with_oracle():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    from cdsbi.diagnostics.multivariate_marginal_cd import MultivariateMarginalCDRecovery
    sim = NormalBivariateUnknownCov()
    grid = [(0.2, -0.1, 0.4, 1.0, -1.5), (-0.3, 0.2, -0.5, -1.0, 1.0)]
    res = MultivariateMarginalCDRecovery(theta_0_grid=grid, n_per_theta=3000)(_Trained(sim), sim)
    df = res.value
    assert {"cov1_ks", "cov2_ks", "cov3_ks", "mu_hotelling_ks"}.issubset(df.columns)
    assert df[["cov1_ks", "cov2_ks", "cov3_ks"]].to_numpy().max() < 0.06


def test_mu_hotelling_recovery_with_oracle():
    # With the closed-form r* pivot, the sampled μ-marginal must recover Hotelling-T²
    # (the joint CD ↦ Hotelling result, validated in the N2 prototype, KS ≈ 0.004).
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    from cdsbi.diagnostics.multivariate_marginal_cd import MultivariateMarginalCDRecovery
    sim = NormalBivariateUnknownCov()
    grid = [(0.2, -0.1, 0.4, 1.0, -1.5)]
    res = MultivariateMarginalCDRecovery(theta_0_grid=grid, n_per_theta=4000)(_Trained(sim), sim)
    assert res.value["mu_hotelling_ks"].iloc[0] < 0.06


def test_noop_without_encode_fn():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    from cdsbi.diagnostics.multivariate_marginal_cd import MultivariateMarginalCDRecovery
    sim = NormalBivariateUnknownCov()
    class _P: d_theta = 5; encode_fn = None
    class _T: procedure = _P()
    res = MultivariateMarginalCDRecovery(theta_0_grid=[(0,)*5], n_per_theta=10)(_T(), sim)
    assert res.passed and "reason" in res.meta
```

- [ ] **Step 2: Run to verify failure** → FAIL (module missing).

- [ ] **Step 3: Implement `src/cdsbi/diagnostics/multivariate_marginal_cd.py`**

```python
"""MultivariateMarginalCDRecovery: d=5 (μ,Σ) marginal-CD recovery.

(i) three covariance DIRECT PITs Φ(r_k(θ₀;X)), k=1,2,3 (the Bartlett χ²/χ²/N
    pivots) — KS vs U + max residual vs the analytic reference;
(ii) the μ JOINT Hotelling-T² recovery — sample r~N(0,I₅) per dataset, invert the
    trained pivot (autoregressive_invert) → θ-samples, take the μ-block, form its
    Hotelling F-stat vs the dataset's sample cov S, KS-compare pooled F-stats to
    F_{p,n−p}. (One r-sample per dataset; pooled over the n_per_theta datasets.)
No-ops unless the procedure exposes encode_fn and the simulator analytic_marginal_cd_pit.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, kstest, f as fdist

from cdsbi.diagnostics.base import DiagnosticResult
from cdsbi.diagnostics.ks_floor import ks_noise_floor
from cdsbi.flows.invert import autoregressive_invert


class MultivariateMarginalCDRecovery:
    name = "multivariate_marginal_cd"

    def __init__(self, theta_0_grid: Sequence, n_per_theta: int = 2000, seed: int = 0):
        self.theta_0_grid = list(theta_0_grid)
        self.n_per_theta = n_per_theta
        self.seed = seed

    def _noop(self, reason):
        return DiagnosticResult(self.name, value=pd.DataFrame(), passed=True,
                                noise_floor=0.0, n_samples=0, meta={"reason": reason})

    def __call__(self, trained, simulator, eval_data=None, x_per_theta=None) -> DiagnosticResult:
        proc = getattr(trained, "procedure", None)
        encode_fn = getattr(proc, "encode_fn", None)
        # Gate on `p` (the bivariate (μ,Σ) sim sets self.p=2). CRITICAL: the 1-D
        # NormalUnknownMeanVar ALSO has analytic_marginal_cd_pit (from M1) and
        # encode_fn is always set, so encode_fn+analytic alone is NOT a sufficient
        # gate — it would fire on a mu_sigma run and IndexError on the 2-wide r.
        # `hasattr(simulator, "p")` excludes the 1-D sim (which has no p attribute).
        if (encode_fn is None or not hasattr(simulator, "analytic_marginal_cd_pit")
                or not hasattr(simulator, "p")):
            return self._noop("not a multivariate (μ,Σ) target with a learned summary")
        n = simulator.n_iid; p = simulator.p
        floor = ks_noise_floor(self.n_per_theta)
        rng = np.random.default_rng(self.seed)
        rows = []
        for theta_0 in self.theta_0_grid:
            x = simulator.sample_x_given_theta(theta_0, self.n_per_theta, rng)
            m = x.shape[0]
            theta_t = torch.tensor([[float(v) for v in theta_0]], dtype=x.dtype).expand(m, -1)
            with torch.no_grad():
                r = proc.pivot(theta_t, x)
            analytic = simulator.analytic_marginal_cd_pit(theta_0, x)
            row = {"theta_0": repr([float(v) for v in theta_0]), "noise_floor": floor}
            for j, key in enumerate(["cov1", "cov2", "cov3"]):
                pit = norm.cdf(r[:, j].detach().cpu().numpy())
                row[f"{key}_ks"] = float(kstest(pit, "uniform").statistic)
                row[f"{key}_resid"] = float(np.abs(pit - analytic[f"{key}_pit"].numpy()).max())
            # μ Hotelling-recovery: invert one r-sample/dataset → μ-block → F-stat
            mu_ks = self._mu_hotelling(proc, simulator, x, rng, n, p)
            row["mu_hotelling_ks"] = mu_ks
            rows.append(row)
        df = pd.DataFrame(rows)
        passed = bool(df[["cov1_ks", "cov2_ks", "cov3_ks", "mu_hotelling_ks"]].to_numpy().max()
                      <= 2.0 * floor + 0.02)
        return DiagnosticResult(self.name, value=df, passed=passed, noise_floor=floor,
                                n_samples=self.n_per_theta, meta={})

    def _sample_mu_marginal(self, proc, simulator, x, n, p):
        """Sample the marginal μ-CD per dataset (one draw each). Returns μ-samples (m,p).
        Trained path: r~N(0,I_d) → autoregressive_invert(flow) → θ-samples → μ-block.
        Oracle path (flow is None): the validated closed-form Bartlett inversion —
        invert the covariance pivots from r_cov, then μ = X̄ + (1/√n)·C·z."""
        feat = proc.encode_fn(x)                                     # (m, d_theta)
        m = x.shape[0]
        flow = getattr(proc, "flow", None)
        if flow is not None:
            r_samp = torch.randn(m, simulator.d_theta, dtype=x.dtype)
            with torch.no_grad():
                theta_s = autoregressive_invert(flow, r_samp, feat)
            return theta_s[:, 3:5].cpu().numpy()
        # --- closed-form Bartlett fallback (oracle r*): the prototype-validated path ---
        from scipy.stats import chi2 as _chi2
        f = feat.cpu().numpy()
        D11 = np.exp(f[:, 0]); D22 = np.exp(f[:, 1]); D21 = f[:, 2]
        xbar = f[:, 3:5]                                             # (m,2) = (X̄₁, X̄₂)
        rc = np.random.default_rng(self.seed + 1).standard_normal((m, 3))
        z = np.random.default_rng(self.seed + 2).standard_normal((m, 2))
        T11sq = _chi2.ppf(np.clip(1 - norm.cdf(rc[:, 0]), 1e-12, 1 - 1e-12), n - 1)
        T22sq = _chi2.ppf(np.clip(1 - norm.cdf(rc[:, 1]), 1e-12, 1 - 1e-12), n - 2)
        C11 = D11 / np.sqrt(T11sq); C22 = D22 / np.sqrt(T22sq)
        L21 = (D21 - rc[:, 2] * C22) * C11 / D11
        mu = np.empty((m, 2))
        mu[:, 0] = xbar[:, 0] + (C11 * z[:, 0]) / math.sqrt(n)
        mu[:, 1] = xbar[:, 1] + (L21 * z[:, 0] + C22 * z[:, 1]) / math.sqrt(n)
        return mu

    def _mu_hotelling(self, proc, simulator, x, rng, n, p) -> float:
        mu_s = self._sample_mu_marginal(proc, simulator, x, n, p)    # (m,2) marginal-μ draws
        obs = x.reshape(x.shape[0], n, p)
        xbar = obs.mean(dim=1).cpu().numpy()
        Xc = (obs - obs.mean(dim=1, keepdim=True)).cpu().numpy()
        S = np.einsum('mki,mkj->mij', Xc, Xc) / (n - 1)
        d = mu_s - xbar
        T2 = n * np.einsum('mi,mij,mj->m', d, np.linalg.inv(S), d)
        F = T2 * (n - p) / (p * (n - 1))
        return float(kstest(F, "f", args=(p, n - p)).statistic)
```
(Add `import math` at the top of the module. The oracle path is the N2-prototype-validated closed-form sampling — so `test_mu_hotelling_recovery_with_oracle` exercises the real joint-CD→Hotelling recovery, NOT a `0.0` stub. The trained path uses `autoregressive_invert`.)

**Also expose the trained flow on the procedure (so the TRAINED μ-path works).** Two minimal, non-breaking edits:
1. `src/cdsbi/confidence_set/procedures.py` — add a keyword arg to `PivotBasedProcedure.__init__` (keep it last, default `None`, so the energy/exact-density runners that build procedures positionally are unaffected):
```python
    def __init__(self, pivot_fn, d_theta, theta_range=(-20.0, 20.0), encode_fn=None, flow=None):
        ...
        self.encode_fn = encode_fn
        self.flow = flow            # the trained flow, for autoregressive inversion (None if N/A)
```
2. `src/cdsbi/methods/cd_sbi.py` `fit()` — pass `flow=self.flow` where it constructs the procedure:
```python
        procedure = PivotBasedProcedure(pivot_fn=pivot_fn, d_theta=simulator.d_theta,
                                        encode_fn=encode_fn, flow=self.flow)
```
The oracle test's `_OracleProc` has no `flow` attribute, so `getattr(proc, "flow", None)` is `None` → the closed-form fallback runs (intended). A `CDSBIRunner`-trained model gets `procedure.flow` set → the `autoregressive_invert` path runs.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_multivariate_marginal_cd.py -v` → 3 pass. The covariance PITs and the μ-Hotelling (KS < 0.06) both recover under the oracle.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/diagnostics/multivariate_marginal_cd.py tests/unit/test_multivariate_marginal_cd.py \
        src/cdsbi/confidence_set/procedures.py src/cdsbi/methods/cd_sbi.py
git commit -m "feat(diag): MultivariateMarginalCDRecovery (3 covariance PITs + μ Hotelling-recovery)"
```

---

## Task 5: LHS coverage grid + runner wiring + paper table

**Files:**
- Modify: `configs/experiment/mu_cov_replication.yaml`, `src/cdsbi/experiments/run.py`, `src/cdsbi/analysis/paper_tables.py`
- Test: `tests/unit/test_paper_tables.py` (append)

- [ ] **Step 1: LHS θ₀ grid + training recipe (μ-tightening) in the experiment config**

Replace the 3-point placeholder grid with a 16-point Latin-hypercube sample over the 5-D prior box, and pin a longer recipe (N1 flagged μ₂ lagging at the default; bump steps). Generate the LHS points once and paste them:
```bash
python -c "
from scipy.stats import qmc
import numpy as np, math
lo = np.array([math.log(0.4), math.log(0.4), -1.5, -3, -3])
hi = np.array([math.log(2.5), math.log(2.5),  1.5,  3,  3])
pts = qmc.scale(qmc.LatinHypercube(d=5, seed=0).random(16), lo, hi)
for row in pts: print('    - [' + ', '.join(f'{v:.3f}' for v in row) + ']')
"
```
Paste the 16 rows into `eval_thetas_interior`, and add:
```yaml
training:
  lr: 3e-3
  n_steps: 12000          # μ-tightening: N1's 6000 left μ₂ RMSE ~0.32; 2× steps
  batch_size: 256
  n_train: 20000
  optimizer: adamw
  fresh_batch: false
```

- [ ] **Step 2: Wire `MultivariateMarginalCDRecovery` into `run.py`**

Import + add to the diagnostics list (it self-gates):
```python
from cdsbi.diagnostics.multivariate_marginal_cd import MultivariateMarginalCDRecovery
...
        ("multivariate_marginal_cd", MultivariateMarginalCDRecovery(
            theta_0_grid=list(cfg.experiment.eval_thetas_interior),
            n_per_theta=int(cfg.experiment.n_eval_per_theta))),
```
Index-row reduction in `_write_index_row` (after the others):
```python
    mmcd = rd.path / "diagnostics" / "multivariate_marginal_cd.parquet"
    if mmcd.exists():
        mdf = pd.read_parquet(mmcd)
        for col in ("cov1_ks", "cov2_ks", "cov3_ks", "mu_hotelling_ks"):
            if col in mdf.columns and len(mdf):
                row[f"mmcd_{col}"] = float(mdf[col].mean())
```
(With `entropy_lower_bound` now defined, `FloorIntegrity` works → the N1 guard-rail is lifted.)

- [ ] **Step 3: `paper_table_mu_cov`** in `paper_tables.py` (mirror `paper_table_mu_sigma`)

```python
def paper_table_mu_cov(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged (μ,Σ) Stage-A table."""
    metrics = ["coverage_error_max", "pivot_rmse", "joint_mahal_ks",
               "mmcd_cov1_ks", "mmcd_cov2_ks", "mmcd_cov3_ks", "mmcd_mu_hotelling_ks",
               "floor_margin", "final_loss", "actual_params_total"]
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby(["method", "budget_name"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg
```
Test (append to `tests/unit/test_paper_tables.py`):
```python
def test_paper_table_mu_cov_aggregates():
    import pandas as pd
    from cdsbi.analysis.paper_tables import paper_table_mu_cov
    df = pd.DataFrame([
        {"method": "cd_sbi", "budget_name": "medium", "coverage_error_max": 0.03,
         "pivot_rmse": 0.12, "mmcd_mu_hotelling_ks": 0.03, "final_loss": 2.0},
        {"method": "cd_sbi", "budget_name": "medium", "coverage_error_max": 0.04,
         "pivot_rmse": 0.14, "mmcd_mu_hotelling_ks": 0.04, "final_loss": 2.1},
    ])
    tbl = paper_table_mu_cov(df)
    assert ("cd_sbi", "medium") in tbl.index
    assert "mmcd_mu_hotelling_ks_mean" in tbl.columns
```

- [ ] **Step 4: Plumbing smoke + tests**

```bash
python -m cdsbi.experiments.run experiment=mu_cov_replication seed=0 \
  budget=small training.n_steps=50 hydra.run.dir=/tmp/mucov_smoke 2>&1 | tail -3
python -c "
import pandas as pd
m = pd.read_parquet('/tmp/mucov_smoke/diagnostics/multivariate_marginal_cd.parquet')
print('mmcd cols:', sorted(m.columns))
row = pd.read_parquet('/tmp/mucov_smoke/index_row.parquet')
print('floor cols:', [c for c in row.columns if 'floor' in c], '| mmcd:', [c for c in row.columns if c.startswith('mmcd')])
"
pytest tests/unit/test_paper_tables.py -q
```
Expected: the full `run.py` now completes (FloorIntegrity works); mmcd parquet has the 4 KS cols; index_row has `floor_margin`/`floor_cheats` + `mmcd_*`. (50 steps → values meaningless; plumbing only.)

- [ ] **Step 5: Commit**

```bash
git add configs/experiment/mu_cov_replication.yaml src/cdsbi/experiments/run.py \
        src/cdsbi/analysis/paper_tables.py tests/unit/test_paper_tables.py
git commit -m "feat(run): LHS θ₀ grid + wire MultivariateMarginalCDRecovery + paper_table_mu_cov (lifts N1 guard-rail)"
```

---

## Task 6: intensive replication (the N2 verdict)

**Files:**
- Create: `tests/intensive/test_replicate_mu_cov.py`

Single full run (the tightened recipe): recovers `r*`, the 3 covariance marginals + μ-Hotelling calibrate, coverage holds across the LHS grid, and the loss sits at the entropy floor (no cheat).

- [ ] **Step 1: Write the test**

```python
"""Intensive: (μ,Σ) Stage-A — recovery + covariance/μ-Hotelling marginal-CD + floor."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.intensive
def test_replicate_mu_cov_stage_a():
    from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
    from cdsbi.conditioners.bartlett_summary import BartlettSummaryConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi import CDSBIRunner
    from cdsbi.diagnostics.multivariate_marginal_cd import MultivariateMarginalCDRecovery

    torch.manual_seed(0)
    sim = NormalBivariateUnknownCov()
    flow = SingleIndexMonotoneFlow(d=5, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=64, depth=2)
    cond = BartlettSummaryConditioner(n_iid=sim.n_iid)
    runner = CDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss())
    config = {"lr": 3e-3, "batch_size": 256, "n_steps": 12000, "n_train": 20000,
              "optimizer": "adamw", "fresh_batch": False}
    trained = runner.fit(simulator=sim, config=config, seed=0)
    assert trained.procedure.flow is not None   # set by CDSBIRunner.fit (Task 4 edit)

    # (a) recovery (tightened: μ coords should improve vs N1's 6000-step run)
    rng = np.random.default_rng(123)
    theta, x = sim.sample(4000, rng)
    with torch.no_grad():
        per = ((trained.procedure.pivot(theta, x).cpu() - sim.r_star(theta, x).cpu()) ** 2).mean(0).sqrt()
    rmse = float((per ** 2).mean().sqrt())
    print(f"rmse={rmse:.3f} per-coord={per.numpy().round(3)}")
    assert rmse < 0.15, f"overall RMSE {rmse:.3f}"

    # (b) marginal-CD recovery (covariance + μ Hotelling)
    grid = [tuple(map(float, t)) for t in [
        (0.0, 0.0, 0.0, 0.0, 0.0), (0.4, -0.4, 0.8, 1.5, -1.5), (-0.4, 0.4, -0.8, -1.5, 1.5)]]
    mres = MultivariateMarginalCDRecovery(theta_0_grid=grid, n_per_theta=3000)(trained, sim).value
    print(mres[["cov1_ks", "cov2_ks", "cov3_ks", "mu_hotelling_ks"]].to_string())
    assert mres[["cov1_ks", "cov2_ks", "cov3_ks"]].to_numpy().max() < 0.07
    assert mres["mu_hotelling_ks"].max() < 0.08

    # (c) no cheat: loss respects the entropy floor
    H = sim.entropy_lower_bound()
    print(f"final_loss={trained.final_loss:.3f} floor={H:.3f}")
    assert trained.final_loss > H - 0.15

    # (d) joint Mahalanobis ‖r‖²~χ²₅
    from scipy.stats import kstest, chi2
    xv = sim.sample_x_given_theta((0.0, 0.0, 0.0, 0.0, 0.0), 3000, np.random.default_rng(7))
    with torch.no_grad():
        rr = trained.procedure.pivot(torch.zeros(3000, 5), xv).cpu().numpy()
    assert kstest(chi2.cdf((rr ** 2).sum(1), df=5), "uniform").statistic < 0.06
```

- [ ] **Step 2: Run it (intensive; ~minutes on GPU)**

Run: `pytest tests/intensive/test_replicate_mu_cov.py -v -s -m intensive`
Report ALL printed values (rmse + per-coord, the 4 marginal KS, final_loss vs floor, joint KS). Interpretation (do NOT loosen bands; report findings):
- **All pass** → N2 done: the (μ,Σ) Stage-A calibrates fully (covariance + Hotelling-μ marginals, coverage, floor). The tightened recipe should bring μ-coords down from N1's 0.32.
- **μ-coord RMSE still high / mu_hotelling_ks high** → report per-coord; the cross-coupled μ₂ may need more steps/width or the ctx-magnitude is the limit — report, do NOT loosen.
- **`final_loss` below floor** → a cheat/bookkeeping bug; STOP and report.

- [ ] **Step 3: Confirm fast suite green** — `pytest -q` (intensive deselected). Report count.

- [ ] **Step 4: Commit**

```bash
git add tests/intensive/test_replicate_mu_cov.py
git commit -m "test(intensive): (μ,Σ) Stage-A replication — recovery + covariance/Hotelling marginals + floor"
```

---

## Self-review

- **Spec coverage (§4):** `entropy_lower_bound` (Task 1) ✓; `analytic_marginal_cd_pit` 3 χ² + Hotelling (Task 2) ✓; `MultivariateMarginalCDRecovery` — 3 covariance direct PITs + the joint Hotelling-T² μ-recovery via the validated sampling-inversion (Tasks 3–4) ✓; LHS coverage grid (Task 5) ✓; `paper_table_mu_cov` (Task 5) ✓; intensive replication + μ-tightening (Task 6) ✓. All three pieces de-risked by the N2 prototype (floor 2.03, Hotelling uniform, joint→Hotelling KS 0.004).
- **Placeholder scan:** every step has runnable code + commands + expected output; the LHS rows are generated by a given one-liner (no hand-waving). No TBD.
- **Type consistency:** `analytic_marginal_cd_pit` keys (`cov1_pit`/`cov2_pit`/`cov3_pit`/`mu_hotelling_pit`) match `MultivariateMarginalCDRecovery`'s reads; `autoregressive_invert(flow, r, context)` signature matches its call in the diagnostic; the diagnostic's df columns (`cov{1,2,3}_ks`, `mu_hotelling_ks`) match the index-row reduction (`mmcd_*`) and `paper_table_mu_cov`. `entropy_lower_bound`/`data_entropy_lower_bound` are the FloorIntegrity loss→floor targets (NFMLELoss→entropy_lower_bound already mapped).
- **Risk — the diagnostic needs the trained flow.** Task 4 adds `flow` to `PivotBasedProcedure` (one optional arg) + sets it in `CDSBIRunner.fit`; the oracle test uses the closed-form sampling fallback. This is the one cross-cutting edit — flagged in Task 4's implementation note; the dual-review should scrutinize the procedure/runner wiring.
- **Risk — inversion cost.** `autoregressive_invert` is 5 coords × ~40 vectorized bisections over `n_per_theta` rows — cheap (one r-sample/dataset, not K/dataset). The intensive test uses n_per_theta=3000.
- **Known follow-on:** N3 (Stage-B I-A invertible ℝ²⁰→ℝ⁵ learned summary).
