# CD-SBI Moment-Summary — Phase 3 (d=5 non-oracle Bartlett) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the non-oracle moment-regression summary (regress `θ`) routes
the **quadratic covariance** parameters that I-A's invertibility failed to route (N3)
on the d=5 bivariate-normal-(μ,Σ) target — head-to-head vs the oracle Bartlett summary.

**Architecture:** Reuse of P1 (`MomentRegressionConditioner` with `d_θ=5` **and `p=2`**,
`TwoStageCDSBIRunner`, `SingleIndexMonotoneFlow` R1-on, `FisherRecovery`). New code is
small: an analytic `log_prob` on the mu_cov simulator (for the true-score
FisherRecovery), a per-direction eigen-ratio extension of FisherRecovery, and the d=5
experiment config + bake-off. **This is the spec's headline empirical test** and its
result (positive or negative) is a finding either way.

> **Execution order (hard dependency):** P1 → P2 not required, but **P1 MUST be
> landed before P3** — P3 imports `MomentRegressionConditioner`, `TwoStageCDSBIRunner`,
> `fisher_recovery._true_score/_cov/_poly_features`, the `cd_sbi_moment` method config,
> and relies on P1 Task 5's `_recipe_dict` pass-through of `stage*_steps`. None exist on
> `main` yet. The suite is meant to execute P1→P2→P3→P4 in order. P3 also requires P1's
> **p-aware conditioner fix** (the `p` parameter); without it the d=5 `ℝ²⁰` data fails
> the `n_iid·p` shape assertion.

**Tech stack:** Python, PyTorch, Hydra, pytest. **Depends on P1 (must be merged first).**

**Spec:** `docs/superpowers/specs/2026-05-31-cd-sbi-moment-summary-design.md` (Phase 3).
**Theory:** note §11–12 (efficiency = Fisher preservation), §13.5 (R1 realizable by MLR).

---

## File structure (Phase 3)

- Modify `src/cdsbi/simulators/normal_bivariate_unknown_cov.py` — add `log_prob`.
- Modify `src/cdsbi/diagnostics/fisher_recovery.py` — add `fisher_eig_ratios` (per-direction).
- Create `configs/conditioner/moment_regression_d5.yaml`, `configs/experiment/moment_mu_cov.yaml`.
- Tests: `tests/unit/test_mu_cov_log_prob.py`, `tests/unit/test_fisher_eig_ratios.py`,
  `tests/integration/test_mu_cov_cross_moment_routing.py`,
  `tests/intensive/test_replicate_moment_mu_cov.py`.

---

### Task 1: Analytic `log_prob` on the mu_cov simulator

**Files:**
- Modify: `src/cdsbi/simulators/normal_bivariate_unknown_cov.py`
- Test: `tests/unit/test_mu_cov_log_prob.py`

`FisherRecovery`'s true score needs `∇_θ log p(X|θ)`; the simulator currently lacks
`log_prob`. Add the analytic bivariate-normal log-density (θ in log-Cholesky coords).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_mu_cov_log_prob.py
import numpy as np
import torch
from scipy.stats import multivariate_normal
from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov


def test_log_prob_matches_scipy():
    sim = NormalBivariateUnknownCov(n_iid=4)
    rng = np.random.default_rng(0)
    theta, x = sim.sample(3, rng)                      # theta (3,5), x (3,8) = 4×2 flat
    lp = sim.log_prob(x, theta)                        # (3,)
    # reference: sum over the 4 iid bivariate obs of a scipy MVN logpdf
    for i in range(3):
        l11, l22, L21, m1, m2 = theta[i].tolist()
        L = np.array([[np.exp(l11), 0.0], [L21, np.exp(l22)]])
        Sigma = L @ L.T; mu = np.array([m1, m2])
        obs = x[i].numpy().reshape(4, 2)
        ref = multivariate_normal(mean=mu, cov=Sigma).logpdf(obs).sum()
        assert abs(lp[i].item() - ref) < 1e-4


def test_log_prob_is_differentiable_in_theta():
    sim = NormalBivariateUnknownCov(n_iid=4)
    rng = np.random.default_rng(1)
    _, x = sim.sample(5, rng)
    th = torch.tensor([[0.1, -0.1, 0.2, 0.5, -0.5]]).repeat(5, 1).detach().requires_grad_(True)
    g = torch.autograd.grad(sim.log_prob(x, th).sum(), th)[0]
    assert g.shape == (5, 5) and torch.isfinite(g).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_mu_cov_log_prob.py -v`
Expected: FAIL with `AttributeError: 'NormalBivariateUnknownCov' object has no attribute 'log_prob'`.

- [ ] **Step 3: Add `log_prob` (torch, differentiable, matches `_chol` convention)**

```python
# add to NormalBivariateUnknownCov (uses the same log-Cholesky θ order as _chol)
import math

def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Σ_i log N₂(X_i; μ, Σ=LLᵀ), θ=(ℓ11,ℓ22,L21,μ1,μ2). x: (n, n_iid*2)."""
    n = x.shape[0]
    obs = x.reshape(n, self.n_iid, self.p)                       # (n, n_iid, 2)
    c11 = torch.exp(theta[:, 0]); c22 = torch.exp(theta[:, 1])   # (n,)
    l21 = theta[:, 2]
    mu = theta[:, 3:5]                                           # (n, 2)
    # Σ⁻¹ via L: solve. For 2×2 lower-tri L, Σ⁻¹ = L⁻ᵀ L⁻¹; logdetΣ = 2(ℓ11+ℓ22)
    # L⁻¹ = [[1/c11, 0], [−l21/(c11 c22), 1/c22]]
    a = 1.0 / c11; b = -l21 / (c11 * c22); d = 1.0 / c22         # entries of L⁻¹
    cen = obs - mu[:, None, :]                                   # (n, n_iid, 2)
    # z = L⁻¹ (x−μ): z0 = a·cen0 ; z1 = b·cen0 + d·cen1
    z0 = a[:, None] * cen[:, :, 0]
    z1 = b[:, None] * cen[:, :, 0] + d[:, None] * cen[:, :, 1]
    quad = (z0 ** 2 + z1 ** 2).sum(dim=1)                        # Σ_i (x−μ)ᵀΣ⁻¹(x−μ)
    logdet = 2.0 * (theta[:, 0] + theta[:, 1])                   # log|Σ|
    return -0.5 * quad - self.n_iid * (0.5 * logdet + math.log(2 * math.pi))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_mu_cov_log_prob.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/simulators/normal_bivariate_unknown_cov.py tests/unit/test_mu_cov_log_prob.py
git commit -m "feat(sim): analytic differentiable log_prob for NormalBivariateUnknownCov"
```

---

### Task 2: Per-direction FisherRecovery (`fisher_eig_ratios`)

**Files:**
- Modify: `src/cdsbi/diagnostics/fisher_recovery.py`
- Test: `tests/unit/test_fisher_eig_ratios.py`

At d=5 a single `det` ratio hides *which* direction fails (the worry: the covariance
coords). Add `fisher_eig_ratios` returning the eigenvalues of `I_h I_X⁻¹` (each ∈
(0,1], →1 ⟺ that direction is recovered). The min eigenvalue is the worst direction.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_fisher_eig_ratios.py
import math
import numpy as np
from cdsbi.diagnostics.fisher_recovery import fisher_eig_ratios
from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar


def test_eig_ratios_near_one_for_sufficient_summary():
    sim = NormalUnknownMeanVar()
    eigs = fisher_eig_ratios(sim, (math.log(1.0), 0.0),
                             lambda s, x: s.oracle_summary(x), n_samples=40000, seed=0)
    assert len(eigs) == 2 and min(eigs) > 0.85, f"all directions should recover: {eigs}"


def test_eig_ratios_flag_the_dropped_direction():
    sim = NormalUnknownMeanVar()
    # keep only X̄ (drop σ): the scale eigen-direction should collapse toward 0
    eigs = fisher_eig_ratios(sim, (math.log(1.0), 0.0),
                             lambda s, x: x.mean(-1, keepdim=True).repeat(1, 2),
                             n_samples=40000, seed=0)
    assert min(eigs) < 0.3, f"the σ direction should be lost: {eigs}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_fisher_eig_ratios.py -v`
Expected: FAIL with `ImportError: cannot import name 'fisher_eig_ratios'`.

- [ ] **Step 3: Add `fisher_eig_ratios` (reuses the Task-2-of-P1 internals)**

```python
# add to src/cdsbi/diagnostics/fisher_recovery.py
def fisher_eig_ratios(simulator, theta0, encode_fn, n_samples: int = 40000,
                      seed: int = 0, degree: int = 3):
    """Eigenvalues of I_h I_X⁻¹ ∈ (0,1] — per-direction Fisher recovery. min → worst
    direction. Uses the same true-score + degree-`degree` polynomial Ê[U_X|h] as
    fisher_det_ratio."""
    rng = np.random.default_rng(seed)
    x = simulator.sample_x_given_theta(np.asarray(theta0), n_samples, rng)
    u = _true_score(simulator, theta0, x)
    h = encode_fn(simulator, x).detach()
    I_X = _cov(u)
    H = _poly_features(h.cpu().numpy(), degree=degree)
    beta, *_ = np.linalg.lstsq(H, u.cpu().numpy(), rcond=None)
    I_h = _cov(torch.from_numpy(H @ beta).float())
    # generalized eigenvalues of (I_h, I_X): eig(I_X⁻¹ I_h), clipped to [0,1]
    eigs = np.linalg.eigvals(np.linalg.solve(I_X, I_h)).real
    return sorted(float(np.clip(e, 0.0, 1.0)) for e in eigs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_fisher_eig_ratios.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/diagnostics/fisher_recovery.py tests/unit/test_fisher_eig_ratios.py
git commit -m "feat(diagnostics): per-direction Fisher eigen-ratios (worst-direction recovery)"
```

---

### Task 3: d=5 experiment config (learned summary vs oracle Bartlett)

**Files:**
- Create: `configs/conditioner/moment_regression_d5.yaml`, `configs/experiment/moment_mu_cov.yaml`
- Test: smoke via Step 3.

- [ ] **Step 1: Inspect the oracle replication to copy the eval grid**

Run: `cat configs/experiment/mu_cov_replication.yaml`
Expected: shows the 16-pt LHS coverage grid + diagnostics for the d=5 target. Copy
its `experiment:` block; swap `conditioner` to the learned moment summary.

- [ ] **Step 2: Write the configs**

```yaml
# configs/conditioner/moment_regression_d5.yaml
name: moment_regression
_target_: cdsbi.conditioners.moment_regression.MomentRegressionConditioner
n_iid: 10
p: 2                                 # BIVARIATE observations (d_x = n_iid·p = 20) —
                                     # requires P1's p-aware conditioner (pools over the
                                     # 10 obs, NOT the 2 coords, preserving cross-moments)
d_theta: 5
target: theta
hidden: 128                          # wider backbone: must compute scatter-matrix feats
```
```yaml
# configs/experiment/moment_mu_cov.yaml
# @package _global_
defaults:
  - override /target: normal_mu_cov
  - override /flow: single_index_monotone
  - override /conditioner: moment_regression_d5
  - override /method: cd_sbi_moment      # P1's method config (name=cd_sbi + two-stage runner)
  - override /budget: medium
method:
  flow: single_index_monotone
training:
  stage1_steps: 8000                  # d=5 regression is harder (cross-moments)
  stage2_steps: 10000
experiment:
  name: moment_mu_cov
  # copy eval_thetas_* / alpha_grid / n_eval* from mu_cov_replication.yaml (Step 1)
```

(Copy the exact `experiment:` eval grid/alpha/n_eval keys from `mu_cov_replication.yaml`
so the d=5 diagnostic battery — coverage, multivariate marginal-CD — produces the same
columns.)

- [ ] **Step 3: Smoke-run**

Run:
```bash
python -m cdsbi.experiments.run experiment=moment_mu_cov seed=0 \
  training.stage1_steps=200 training.stage2_steps=200 hydra.run.dir=/tmp/mucov_smoke
```
Expected: exits 0; `arch_metadata.conditioner_class == "MomentRegressionConditioner"`,
`flow_class == "SingleIndexMonotoneFlow"`.

- [ ] **Step 4: Commit**

```bash
git add configs/conditioner/moment_regression_d5.yaml configs/experiment/moment_mu_cov.yaml
git commit -m "feat(experiments): d=5 moment-summary experiment (vs oracle Bartlett)"
```

---

### Task 4: Integration — the cross-moment routing test (the headline)

**Files:**
- Create: `tests/integration/test_mu_cov_cross_moment_routing.py`

The decisive Phase-3 question: does the learned summary recover **all 5** directions
(esp. the covariance coords), measured by `min eig(I_h I_X⁻¹)`, approaching the oracle?

- [ ] **Step 1: Write the test (it is a measurement, with a clear pass/fail bar)**

```python
# tests/integration/test_mu_cov_cross_moment_routing.py
import numpy as np
import torch
import pytest
from cdsbi.conditioners.moment_regression import MomentRegressionConditioner
from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
from cdsbi.methods.cd_sbi_two_stage import TwoStageCDSBIRunner
from cdsbi.simulators.normal_bivariate_unknown_cov import NormalBivariateUnknownCov
from cdsbi.diagnostics.fisher_recovery import fisher_eig_ratios


@pytest.mark.intensive   # d=5 training is minutes-scale; opt-in
def test_learned_summary_routes_all_five_directions():
    sim = NormalBivariateUnknownCov(n_iid=10)
    cond = MomentRegressionConditioner(n_iid=10, p=2, d_theta=5, target="theta", hidden=128)
    flow = SingleIndexMonotoneFlow(d=5, theta_signs=sim.theta_signs,
                                   feat_signs=sim.feat_signs, hidden=64)
    runner = TwoStageCDSBIRunner(flow=flow, conditioner=cond, device="cpu")
    runner.fit(sim, {"lr": 2e-3, "stage1_steps": 8000, "stage2_steps": 0,
                     "batch_size": 512, "fresh_batch": True}, seed=0)

    def learned_h(simulator, x):
        with torch.no_grad():
            return runner.conditioner.encode(x)[0]

    theta0 = np.array([0.0, 0.0, 0.2, 0.5, -0.5])     # an interior θ₀
    learned = fisher_eig_ratios(sim, theta0, learned_h, n_samples=40000, seed=0)
    oracle = fisher_eig_ratios(sim, theta0, lambda s, x: s.oracle_summary(x),
                               n_samples=40000, seed=0)
    print("learned eig-ratios:", [round(e, 3) for e in learned])
    print("oracle  eig-ratios:", [round(e, 3) for e in oracle])
    # HYPOTHESIS (not a foregone conclusion, per spec §5 P3 / theory N3): regression
    # directly targets the Cholesky coords, so it SHOULD route them. Bar: the worst
    # learned direction reaches ≥ 0.6 of the worst oracle direction. A failure here
    # localizes to DeepSets-SGD not finding the cross-moment features — a sharp finding.
    assert min(learned) > 0.6 * min(oracle), (
        f"a covariance direction failed to route: learned {min(learned):.3f} "
        f"vs oracle {min(oracle):.3f}")
```

- [ ] **Step 2: Run it (opt-in)**

Run: `pytest tests/integration/test_mu_cov_cross_moment_routing.py -m intensive -v -s`
Expected: prints both eigen-ratio vectors. PASS confirms the headline (regression
routes the covariance). FAIL is a documented finding — try `hidden=256`,
`stage1_steps=15000`; if it persists, the negative result stands (record it).

- [ ] **Step 3: Record the outcome**

Whatever the result, note it in the spec's Phase-3 section (a one-line result + the
eigen-ratio vector). A negative result is not a code bug — do not "fix" it past the
recipe knobs above.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_mu_cov_cross_moment_routing.py
git commit -m "test(integration): d=5 cross-moment routing — learned summary vs oracle"
```

---

### Task 5: Intensive replication — d=5 coverage + per-direction recovery vs oracle

**Files:**
- Create: `tests/intensive/test_replicate_moment_mu_cov.py`

- [ ] **Step 1: Write the replication test**

```python
# tests/intensive/test_replicate_moment_mu_cov.py
from pathlib import Path
import subprocess
import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.intensive
def test_replicate_moment_mu_cov(tmp_path):
    out = tmp_path / "mucov"
    for seed in range(3):
        cmd = ["python", "-m", "cdsbi.experiments.run",
               f"hydra.run.dir={out}/seed_{seed}",
               "experiment=moment_mu_cov", f"seed={seed}", "training.fresh_batch=true"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, f"seed {seed} failed:\n{r.stderr[-2000:]}"

    from cdsbi.analysis.loaders import load_runs
    df = load_runs(str(out / "*"))
    assert len(df) == 3
    print(df[["coverage_error_max"]].to_string())
    # validity at d=5 with a LEARNED summary (the non-oracle replacement). Bar mirrors
    # the oracle Bartlett verdict's central-coverage band; μ₂-style tail is addressed
    # in P4 (S2b reweighting), so allow some headroom here.
    assert df["coverage_error_max"].mean() <= 0.20
```

- [ ] **Step 2: Run it (opt-in)** — `pytest tests/intensive/test_replicate_moment_mu_cov.py -m intensive -v -s`
Expected: PASS (central coverage holds; the residual extreme-θ₀ tail is P4's job).

- [ ] **Step 3: Confirm fast suite green** — `pytest -q -m "not intensive"`

- [ ] **Step 4: Commit**

```bash
git add tests/intensive/test_replicate_moment_mu_cov.py
git commit -m "test(intensive): d=5 moment-summary replication (non-oracle vs oracle)"
```

---

## Self-review (checklist)

- **Spec coverage (Phase 3):** log_prob for the true score (Task 1), per-direction
  recovery metric (Task 2), d=5 experiment vs oracle (Task 3), the cross-moment
  routing headline test (Task 4), replication (Task 5). ✓
- **P3 is P2-independent** (spec): uses only P1 machinery + R1-on flow; no
  non-monotone pivot, no disconnected extractor. ✓
- **Outcome-honest:** Task 4 is a measurement with a named failure mode; a negative
  result is recorded, not "fixed" (spec §5 P3). ✓
- **μ₂ tail** explicitly deferred to P4 (the coverage bar has headroom).
