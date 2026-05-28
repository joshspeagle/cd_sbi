# CD-SBI Visualizations — F2 Empirical Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 10 empirical figures (E1–E10) of the visualization suite by composing F1 panels over real run-dir data, including the data-sourcing work needed to feed the PIT/Jacobian figures, and insert all 10 into the manuscript.

**Architecture:** Three phases. **Phase A** extends three diagnostics (`MarginalPIT`, `JointMahalanobis`, `JacobianRecovery`) to persist their raw arrays via `DiagnosticResult.meta`, wires `run.py` to write companion `*_raw.parquet` files, adds `data_io` loaders for figure inputs, and re-runs a small set of CDSBI configs into **stable** `outputs/figure_data/<id>/` paths so figures can pin them. **Phase B** builds one figure builder per catalogue entry (E1–E10), each a pure `render(spec) -> Figure` composing F1 panels, with a manifest entry, a smoke test, and a controller visual-acceptance pass. **Phase C** inserts the rendered PDFs into `cd_sbi_v7.tex` and rebuilds.

**Tech Stack:** matplotlib 3.10.5 (F0 `style` + F1 `panels`), pandas/pyarrow, torch (checkpoints), Hydra (re-runs), pytest, pdflatex/bibtex (manuscript). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-28-cd-sbi-visualizations-design.md` (F2 = catalogue E1–E10 + manuscript integration). **Prereqs:** F0 (infra) + F1 (panels) merged.

---

## Conventions (read first)

- **Figure builder contract** (same as F0's `_hello`): `def render(spec: FigureSpec) -> matplotlib.figure.Figure`. The builder loads its own data via `data_io` from the paths on `spec` (`source_runs`, `checkpoint_runs`), composes F1 panels onto a Figure it creates, and returns it. It NEVER calls `savefig` — the render CLI (F0) owns that. Builders call `style.apply_style()` first.
- **Stable figure-data paths.** Phase A re-runs write to fixed, non-timestamped dirs via the `run_dir=` override (a top-level config key), e.g. `run_dir=outputs/figure_data/8_1_cdsbi`. This lets manifest entries pin exact paths that survive re-runs. The `outputs/` tree is gitignored; the rendered `figures/*.pdf|png` ARE tracked (F0 set this up), so the manuscript builds without the run-dirs.
- **Visual acceptance (mandatory per figure).** After a builder + manifest entry land and the figure renders, the controller `Read`s the PNG and judges it against the spec checklist — renders cleanly, style applied, mathtext real, **and the figure's one-sentence message lands** (this is the F2-level "message" check that F1 deferred). Each builder commit records a one-line `visual:` verdict.
- **Existing data vs re-run.** Figures that consume already-persisted data need no re-run: E2 (coverage curves per method), E6 (loss tails), E8/E9/E10 (`coverage_error_max`/params from `index_row.parquet`). Only E1/E3/E4/E5 need the Phase-A raw-data re-runs; E7 needs a folding-trajectory regen.
- **`d` per section:** §8.1 `d=1`; §8.2/§8.3 `d=2`; §8.4 `d=1` (on the sufficient statistic `T`).

---

## File structure

```
NEW (Phase A)
  src/cdsbi/analysis/figures/data_io/figure_data.py   # loaders: pit values, joint r², jacobian matrices, coverage curve/tile, loss tail
  tests/unit/test_figure_data_loaders.py
  tools/regen_figure_data.py                           # orchestrates the targeted CDSBI re-runs + folding regen

NEW (Phase B) — one builder + manifest entry each
  src/cdsbi/analysis/figures/figures/e1_loc_normal_calibration.py
  src/cdsbi/analysis/figures/figures/e2_loc_normal_cross_method.py
  src/cdsbi/analysis/figures/figures/e3_joint_diagnostics.py
  src/cdsbi/analysis/figures/figures/e4_jacobian_recovery.py
  src/cdsbi/analysis/figures/figures/e5_exp_rate_calibration.py
  src/cdsbi/analysis/figures/figures/e6_r2_ablation_bars.py
  src/cdsbi/analysis/figures/figures/e7_catastrophic_folding.py
  src/cdsbi/analysis/figures/figures/e8_headline_summary.py
  src/cdsbi/analysis/figures/figures/e9_budget_saturation.py
  src/cdsbi/analysis/figures/figures/e10_per_experiment_boxplots.py
  tests/integration/test_e_figures_smoke.py            # one smoke test per builder, on fixtures

MODIFY
  src/cdsbi/diagnostics/marginal_pit.py                # carry raw PIT u-values in meta
  src/cdsbi/diagnostics/joint_mahalanobis.py           # carry raw r² per θ_0 in meta
  src/cdsbi/diagnostics/jacobian_recovery.py           # carry J_emp_mean + J_true in meta
  src/cdsbi/experiments/run.py                         # write *_raw.parquet companions when meta carries raw arrays
  configs/figures/manifest.yaml                        # add e1..e10 entries
  cd_sbi_v7.tex                                         # insert 10 \includegraphics figures (Phase C)
  tests/figures_fixtures/__init__.py                   # add raw-companion fixture builders for smoke tests

No changes to F1 panels, F0 style/manifest/render/gallery.
```

---

# PHASE A — Data sourcing

## Task A1: `MarginalPIT` carries raw PIT values; `run.py` writes the companion

**Files:**
- Modify: `src/cdsbi/diagnostics/marginal_pit.py`
- Modify: `src/cdsbi/experiments/run.py`
- Test: `tests/unit/test_marginal_pit_raw.py` (create)

The diagnostic currently computes `u = Φ(r)` and discards it. Carry it in `meta["pit_u"]` (a numpy array; 1-D for d=1, or a dict of per-coord arrays for d>1). Then `_run_diagnostics` writes `diagnostics/marginal_pit_raw.parquet` when `meta` carries it.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_marginal_pit_raw.py`:

```python
"""MarginalPIT carries the raw PIT array in meta['pit_u'] for figure sourcing."""
from __future__ import annotations

import numpy as np
import torch


class _FakePivotProcedure:
    """Minimal PivotBasedProcedure stand-in: r(θ, x) = θ - x (1-D)."""
    def pivot(self, theta, x):
        return theta - x


class _FakeTrained:
    def __init__(self):
        from cdsbi.confidence_set.procedures import PivotBasedProcedure
        # Duck-type: MarginalPIT checks isinstance(procedure, PivotBasedProcedure),
        # so subclass it.
        class P(PivotBasedProcedure):
            def __init__(self): pass
            def pivot(self, theta, x): return theta - x
            def confidence_set(self, *a, **k): raise NotImplementedError
        self.procedure = P()


def test_marginal_pit_meta_carries_raw_u_1d():
    from cdsbi.diagnostics.marginal_pit import MarginalPIT
    rng = np.random.default_rng(0)
    theta = torch.zeros(500, 1)
    x = torch.tensor(rng.normal(0, 1, size=(500, 1)), dtype=torch.float32)
    res = MarginalPIT()(_FakeTrained(), simulator=None, eval_data=(theta, x))
    assert "pit_u" in res.meta
    u = np.asarray(res.meta["pit_u"])
    assert u.shape == (500,)
    assert (u >= 0).all() and (u <= 1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_marginal_pit_raw.py -v`
Expected: FAIL — `assert "pit_u" in res.meta` (meta is empty today).

- [ ] **Step 3: Modify `marginal_pit.py` to carry raw u in meta**

In the `d == 1` branch, change the return to include meta; in the `d > 1` branch, carry per-coord arrays. Replace the two `return DiagnosticResult(...)` blocks:

The `d == 1` block becomes:
```python
        if d == 1:
            u = norm.cdf(r_np.flatten())
            ks_stat, _ = kstest(u, "uniform")
            return DiagnosticResult(
                name=self.name, value=float(ks_stat),
                passed=ks_stat <= floor, noise_floor=floor, n_samples=u.size,
                meta={"pit_u": u, "d": 1},
            )
```

The `d > 1` final return becomes (build a per-coord array `u_all` of shape (N, d)):
```python
        import numpy as _np
        u_all = _np.column_stack([norm.cdf(r_np[:, k]) for k in range(d)])
        df = pd.DataFrame(rows)
        return DiagnosticResult(
            name=self.name, value=df,
            passed=bool(df["passed"].all()), noise_floor=floor,
            n_samples=r_np.shape[0], meta={"d": int(d), "pit_u": u_all},
        )
```
(Keep the existing `rows` construction loop above unchanged.)

- [ ] **Step 4: Modify `run.py` `_run_diagnostics` to write the companion**

In `_run_diagnostics`, inside the `for name, diag in diagnostics:` loop, AFTER `df.to_parquet(diag_dir / f"{name}.parquet")` and BEFORE `diag_results[name] = result`, add:

```python
        _write_raw_companion(name, result, diag_dir)
```

And add this module-level helper to `run.py` (just above `_run_diagnostics`):

```python
def _write_raw_companion(name: str, result, diag_dir) -> None:
    """Persist raw figure-sourcing arrays a diagnostic carries in result.meta.

    Keeps the main <name>.parquet a compact summary while making raw PIT
    values / Mahalanobis r² / Jacobian matrices available to the figure suite.
    """
    meta = result.meta or {}
    if name == "marginal_pit" and "pit_u" in meta:
        u = np.asarray(meta["pit_u"])
        if u.ndim == 1:
            df = pd.DataFrame({"u": u})
        else:
            df = pd.DataFrame({f"u{k}": u[:, k] for k in range(u.shape[1])})
        df.to_parquet(diag_dir / "marginal_pit_raw.parquet")
    elif name == "joint_mahalanobis" and "r_sq" in meta:
        # meta["r_sq"]: dict theta_repr -> 1-D np.array
        frames = [
            pd.DataFrame({"theta_0_repr": k, "r_sq": v})
            for k, v in meta["r_sq"].items()
        ]
        pd.concat(frames, ignore_index=True).to_parquet(
            diag_dir / "joint_mahalanobis_raw.parquet"
        )
    elif name == "jacobian_recovery" and "J_emp_mean" in meta:
        J_emp = np.asarray(meta["J_emp_mean"])
        J_true = np.asarray(meta["J_true"])
        d = J_emp.shape[0]
        rows = [
            {"i": i, "j": j, "j_emp": float(J_emp[i, j]), "j_true": float(J_true[i, j])}
            for i in range(d) for j in range(d)
        ]
        pd.DataFrame(rows).to_parquet(diag_dir / "jacobian_recovery_raw.parquet")
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/unit/test_marginal_pit_raw.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Run the full diagnostics + run integration tests to confirm no regression**

Run: `pytest tests/diagnostics/test_marginal_pit.py tests/integration/test_run_diagnostics.py -q`
Expected: PASS (existing tests still green — meta is additive).

- [ ] **Step 7: Commit**

```bash
git add src/cdsbi/diagnostics/marginal_pit.py src/cdsbi/experiments/run.py tests/unit/test_marginal_pit_raw.py
git commit -m "feat(diag): MarginalPIT carries raw PIT u-values; run.py writes companion parquet"
```

---

## Task A2: `JointMahalanobis` carries raw r² per θ_0

**Files:**
- Modify: `src/cdsbi/diagnostics/joint_mahalanobis.py`
- Test: `tests/unit/test_joint_mahalanobis_raw.py` (create)

Carry the per-θ_0 squared-norm arrays in `meta["r_sq"]` (dict `theta_repr -> np.array`). The `run.py` companion writer (Task A1) already handles the `joint_mahalanobis` case, so only the diagnostic changes here.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_joint_mahalanobis_raw.py`:

```python
"""JointMahalanobis carries raw r² arrays per θ_0 in meta['r_sq']."""
from __future__ import annotations

import numpy as np
import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure


class _Sim:
    d_theta = 2
    def sample_x_given_theta(self, theta_0, n, rng):
        # X ~ N(theta_0, I); pivot r = theta_0 - x so ||r||² ~ χ²_2 under truth.
        mean = np.asarray(theta_0, dtype=np.float32)
        return torch.tensor(rng.normal(mean, 1.0, size=(n, 2)), dtype=torch.float32)


class _Trained:
    def __init__(self):
        class P(PivotBasedProcedure):
            def __init__(self): pass
            def pivot(self, theta, x): return theta - x
            def confidence_set(self, *a, **k): raise NotImplementedError
        self.procedure = P()


def test_joint_mahalanobis_meta_carries_r_sq():
    from cdsbi.diagnostics.joint_mahalanobis import JointMahalanobis
    grid = [[0.0, 0.0], [2.0, -1.0]]
    res = JointMahalanobis(theta_0_grid=grid, n_per_theta=300)(_Trained(), _Sim())
    assert "r_sq" in res.meta
    assert set(res.meta["r_sq"].keys()) == {"[0.0, 0.0]", "[2.0, -1.0]"}
    arr = np.asarray(res.meta["r_sq"]["[0.0, 0.0]"])
    assert arr.shape == (300,)
    assert (arr >= 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_joint_mahalanobis_raw.py -v`
Expected: FAIL — `assert "r_sq" in res.meta`.

- [ ] **Step 3: Modify `joint_mahalanobis.py`**

In `__call__`, accumulate the raw arrays. Before the `for theta_0 in self.theta_0_grid:` loop add:
```python
        r_sq_by_theta = {}
```
Inside the loop, right after `r_sq = r.pow(2).sum(dim=-1).cpu().numpy()`, add:
```python
            r_sq_by_theta[str(list(map(float, list(theta_0))))] = r_sq
```
Then change the final return's `meta` from `meta={"d": int(d)}` to:
```python
            meta={"d": int(d), "r_sq": r_sq_by_theta},
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_joint_mahalanobis_raw.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Regression check**

Run: `pytest tests/diagnostics/test_joint_mahalanobis.py -q`
Expected: PASS (meta is additive).

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/diagnostics/joint_mahalanobis.py tests/unit/test_joint_mahalanobis_raw.py
git commit -m "feat(diag): JointMahalanobis carries raw r² per θ_0 in meta"
```

---

## Task A3: `JacobianRecovery` carries the trained + truth matrices

**Files:**
- Modify: `src/cdsbi/diagnostics/jacobian_recovery.py`
- Test: `tests/unit/test_jacobian_recovery_raw.py` (create)

Carry `J_emp_mean` and `J_true` (both `(d, d)` numpy arrays) in `meta`. The `run.py` companion writer (Task A1) already handles the `jacobian_recovery` case.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_jacobian_recovery_raw.py`:

```python
"""JacobianRecovery carries J_emp_mean and J_true matrices in meta."""
from __future__ import annotations

import numpy as np
import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure


class _Sim:
    d_theta = 2
    def __init__(self):
        self.L_inv = np.array([[1.0, 0.0], [-0.4, 0.9]], dtype=np.float32)
    def r_star_jacobian(self):
        return torch.tensor(self.L_inv)
    def sample(self, n, rng):
        theta = torch.tensor(rng.normal(0, 1, size=(n, 2)), dtype=torch.float32)
        x = torch.tensor(rng.normal(0, 1, size=(n, 2)), dtype=torch.float32)
        return theta, x


class _Trained:
    """pivot r = L_inv @ (θ - x): linear, so ∂r/∂θ = L_inv exactly."""
    def __init__(self, L_inv):
        L = torch.tensor(L_inv)
        class P(PivotBasedProcedure):
            def __init__(self): pass
            def pivot(self, theta, x):
                return (theta - x) @ L.T
            def confidence_set(self, *a, **k): raise NotImplementedError
        self.procedure = P()


def test_jacobian_recovery_meta_carries_matrices():
    from cdsbi.diagnostics.jacobian_recovery import JacobianRecovery
    sim = _Sim()
    res = JacobianRecovery(n_points=50)(_Trained(sim.L_inv), sim)
    assert "J_emp_mean" in res.meta and "J_true" in res.meta
    J_emp = np.asarray(res.meta["J_emp_mean"])
    J_true = np.asarray(res.meta["J_true"])
    assert J_emp.shape == (2, 2) and J_true.shape == (2, 2)
    np.testing.assert_allclose(J_true, sim.L_inv, atol=1e-5)
    np.testing.assert_allclose(J_emp, sim.L_inv, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_jacobian_recovery_raw.py -v`
Expected: FAIL — `assert "J_emp_mean" in res.meta`.

- [ ] **Step 3: Modify `jacobian_recovery.py`**

Change the final (non-skip) return's `meta`. The current line is:
```python
            meta={"d": d, "tol": self.max_residual_tol},
```
Replace with:
```python
            meta={
                "d": d, "tol": self.max_residual_tol,
                "J_emp_mean": J_emp_mean.cpu().numpy(),
                "J_true": (J_true.cpu().numpy() if hasattr(J_true, "cpu") else np.asarray(J_true)),
            },
```
(`J_emp_mean` and `J_true` are already computed above in the function.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_jacobian_recovery_raw.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Regression check**

Run: `pytest tests/diagnostics/test_jacobian_recovery.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/diagnostics/jacobian_recovery.py tests/unit/test_jacobian_recovery_raw.py
git commit -m "feat(diag): JacobianRecovery carries J_emp_mean + J_true matrices in meta"
```

---

## Task A4: `data_io/figure_data.py` loaders + fixtures

**Files:**
- Create: `src/cdsbi/analysis/figures/data_io/figure_data.py`
- Modify: `tests/figures_fixtures/__init__.py`
- Test: `tests/unit/test_figure_data_loaders.py`

Loaders that turn a run-dir into the arrays the E-figures need. Coverage curve/tile come from the existing `coverage.parquet`; the raw loaders come from the Phase-A companions.

- [ ] **Step 1: Add fixture builders for the raw companions**

Append to `tests/figures_fixtures/__init__.py`:

```python
def write_marginal_pit_raw(run_dir: Path, u) -> None:
    """Write a diagnostics/marginal_pit_raw.parquet with a 1-D 'u' column."""
    import numpy as _np
    diag = run_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"u": _np.asarray(u)}).to_parquet(diag / "marginal_pit_raw.parquet")


def write_jacobian_raw(run_dir: Path, j_emp, j_true) -> None:
    """Write a diagnostics/jacobian_recovery_raw.parquet (i, j, j_emp, j_true)."""
    import numpy as _np
    j_emp = _np.asarray(j_emp); j_true = _np.asarray(j_true)
    d = j_emp.shape[0]
    diag = run_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    rows = [{"i": i, "j": j, "j_emp": float(j_emp[i, j]), "j_true": float(j_true[i, j])}
            for i in range(d) for j in range(d)]
    pd.DataFrame(rows).to_parquet(diag / "jacobian_recovery_raw.parquet")


def write_coverage(run_dir: Path, theta0_list, alpha_list) -> None:
    """Write a diagnostics/coverage.parquet matching the real schema
    (theta_0_0, alpha, nominal, empirical, ...), empirical = nominal + small noise."""
    import numpy as _np
    diag = run_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    rng = _np.random.default_rng(0)
    rows = []
    for t in theta0_list:
        for a in alpha_list:
            rows.append({"theta_0_0": float(t), "alpha": float(a), "nominal": float(a),
                         "empirical": float(a) + rng.normal(0, 0.01), "n_eval": 2000})
    pd.DataFrame(rows).to_parquet(diag / "coverage.parquet")
```

- [ ] **Step 2: Write the failing loader test**

`tests/unit/test_figure_data_loaders.py`:

```python
"""figure_data loaders: PIT values, jacobian matrices, coverage curve/tile, loss tail."""
from __future__ import annotations

import numpy as np

from tests.figures_fixtures import (
    make_run_dir, write_marginal_pit_raw, write_jacobian_raw, write_coverage,
)


def test_load_pit_values(tmp_path):
    from cdsbi.analysis.figures.data_io.figure_data import load_pit_values
    rd = tmp_path / "run"; rd.mkdir()
    write_marginal_pit_raw(rd, np.linspace(0, 1, 200))
    u = load_pit_values(str(rd))
    assert u.shape == (200,)


def test_load_jacobian_matrices(tmp_path):
    from cdsbi.analysis.figures.data_io.figure_data import load_jacobian_matrices
    rd = tmp_path / "run"; rd.mkdir()
    j_true = np.array([[1.0, 0.0], [-0.4, 0.9]])
    write_jacobian_raw(rd, j_true + 0.01, j_true)
    j_emp, jt = load_jacobian_matrices(str(rd))
    assert j_emp.shape == (2, 2) and jt.shape == (2, 2)
    np.testing.assert_allclose(jt, j_true)


def test_load_coverage_curve(tmp_path):
    from cdsbi.analysis.figures.data_io.figure_data import load_coverage_curve
    rd = tmp_path / "run"; rd.mkdir()
    write_coverage(rd, theta0_list=[-2.0, 0.0, 2.0], alpha_list=[0.5, 0.68, 0.9, 0.95])
    nominal, empirical = load_coverage_curve(str(rd))
    # Averaged over θ_0 -> one (nominal, empirical) point per α.
    assert nominal.shape == (4,) and empirical.shape == (4,)
    assert list(nominal) == [0.5, 0.68, 0.9, 0.95]


def test_load_coverage_tile(tmp_path):
    from cdsbi.analysis.figures.data_io.figure_data import load_coverage_tile
    rd = tmp_path / "run"; rd.mkdir()
    write_coverage(rd, theta0_list=[-2.0, 0.0, 2.0], alpha_list=[0.5, 0.68, 0.9, 0.95])
    theta0, alpha, error = load_coverage_tile(str(rd))
    assert theta0.shape == (3,) and alpha.shape == (4,)
    assert error.shape == (3, 4)             # rows=θ_0, cols=α
    assert (error >= 0).all()


def test_load_loss_tail_mean(tmp_path):
    from cdsbi.analysis.figures.data_io.figure_data import load_loss_tail_mean
    rd = make_run_dir(tmp_path, method="cd_sbi", budget_name="medium", seed=0,
                      coverage_error_max=0.03, final_loss=0.99,
                      loss_history_tail=[1.1, 1.0, 0.99, 0.99])
    assert abs(load_loss_tail_mean(str(rd)) - np.mean([1.1, 1.0, 0.99, 0.99])) < 1e-9
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/unit/test_figure_data_loaders.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cdsbi.analysis.figures.data_io.figure_data'`

- [ ] **Step 4: Write the implementation**

`src/cdsbi/analysis/figures/data_io/figure_data.py`:

```python
"""Loaders that turn a run-dir into the arrays the E-figures consume.

Coverage curve/tile come from the standard diagnostics/coverage.parquet.
Raw PIT / Jacobian come from the *_raw.parquet companions written by
run.py when the diagnostics carry raw arrays in their result.meta (F2 Phase A).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cdsbi.analysis.figures.data_io.checkpoints import load_checkpoint


def load_pit_values(run_dir: str, coord: int = 0) -> np.ndarray:
    """Raw PIT values from diagnostics/marginal_pit_raw.parquet.

    For d=1 the column is 'u'; for d>1 columns are 'u0','u1',...; `coord` picks one.
    """
    df = pd.read_parquet(Path(run_dir) / "diagnostics" / "marginal_pit_raw.parquet")
    col = "u" if "u" in df.columns else f"u{coord}"
    return df[col].to_numpy()


def load_joint_mahalanobis_sq(run_dir: str) -> dict[str, np.ndarray]:
    """Per-θ_0 squared-norm arrays from joint_mahalanobis_raw.parquet."""
    df = pd.read_parquet(Path(run_dir) / "diagnostics" / "joint_mahalanobis_raw.parquet")
    return {k: g["r_sq"].to_numpy() for k, g in df.groupby("theta_0_repr")}


def load_jacobian_matrices(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """(J_emp_mean, J_true) matrices from jacobian_recovery_raw.parquet."""
    df = pd.read_parquet(Path(run_dir) / "diagnostics" / "jacobian_recovery_raw.parquet")
    d = int(df["i"].max()) + 1
    j_emp = np.zeros((d, d)); j_true = np.zeros((d, d))
    for _, row in df.iterrows():
        j_emp[int(row["i"]), int(row["j"])] = row["j_emp"]
        j_true[int(row["i"]), int(row["j"])] = row["j_true"]
    return j_emp, j_true


def load_coverage_curve(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """(nominal, empirical) averaged over θ_0, one point per α — sorted by α."""
    df = pd.read_parquet(Path(run_dir) / "diagnostics" / "coverage.parquet")
    g = df.groupby("alpha", as_index=False).agg(nominal=("nominal", "first"),
                                                 empirical=("empirical", "mean"))
    g = g.sort_values("alpha")
    return g["nominal"].to_numpy(), g["empirical"].to_numpy()


def load_coverage_tile(run_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(theta0, alpha, error) where error[i,j] = |empirical - nominal| at (θ_0_i, α_j).

    Uses the first θ_0 coordinate (theta_0_0) as the row axis; adequate for the
    d≤2 sections (each θ_0 grid point gets a row).
    """
    df = pd.read_parquet(Path(run_dir) / "diagnostics" / "coverage.parquet")
    theta0 = np.sort(df["theta_0_0"].unique())
    alpha = np.sort(df["alpha"].unique())
    error = np.zeros((len(theta0), len(alpha)))
    for i, t in enumerate(theta0):
        for j, a in enumerate(alpha):
            sub = df[(df["theta_0_0"] == t) & (df["alpha"] == a)]
            error[i, j] = float((sub["empirical"] - sub["nominal"]).abs().mean())
    return theta0, alpha, error


def load_loss_tail_mean(run_dir: str) -> float:
    """Mean of the persisted loss_history_tail from model.pt."""
    ck = load_checkpoint(run_dir)
    tail = ck.loss_history_tail
    if tail is None:
        return float(ck.final_loss)
    return float(np.mean(np.asarray(tail)))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_figure_data_loaders.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/analysis/figures/data_io/figure_data.py tests/figures_fixtures/__init__.py tests/unit/test_figure_data_loaders.py
git commit -m "feat(figures): figure_data loaders (PIT, jacobian, coverage curve/tile, loss tail)"
```

---

## Task A5: Regenerate figure data into stable paths

**Files:**
- Create: `tools/regen_figure_data.py`

This task runs actual training. It re-runs the four CDSBI replication configs into **stable** `outputs/figure_data/<id>/` dirs (so the manifest can pin them) and regenerates the catastrophic-folding trajectory. The `run_dir=` override pins the path; `seed=0`, medium budget.

- [ ] **Step 1: Write the regen orchestration script**

`tools/regen_figure_data.py`:

```python
"""Regenerate the run-dirs the PIT/Jacobian/calibration figures source from.

Runs four CDSBI replication configs into STABLE outputs/figure_data/<id>/ paths
(via the run_dir= override) so the figure manifest can pin them, and regenerates
the catastrophic-folding loss trajectory for E7.

Usage:  python tools/regen_figure_data.py
(Each CDSBI run is a few minutes on GPU; longer on CPU.)
"""
from __future__ import annotations

import subprocess
import sys

# (experiment, extra overrides) -> stable run_dir
RUNS = [
    ("8_1_replication", [], "outputs/figure_data/8_1_cdsbi"),
    ("8_2_replication", [], "outputs/figure_data/8_2_cdsbi"),
    ("8_3_replication", [], "outputs/figure_data/8_3_cdsbi"),
    ("8_4_replication", [], "outputs/figure_data/8_4_cdsbi"),
]


def main():
    for exp, extra, run_dir in RUNS:
        cmd = [sys.executable, "-m", "cdsbi.experiments.run",
               f"experiment={exp}", "method=cd_sbi", "seed=0",
               f"run_dir={run_dir}", *extra]
        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    # E7 folding trajectory: reuse the recipe documented in
    # tests/ablation/test_trained_folding.py, capturing the FULL loss list,
    # and write it to outputs/figure_data/8_4_folding/folding_trajectory.parquet.
    # (Implementer: import/replicate that test's training setup; this is a
    # regen utility, not production code.)
    from tools._folding_regen import regen_folding
    regen_folding("outputs/figure_data/8_4_folding/folding_trajectory.parquet")
    print("done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the folding regen helper**

Read `tests/ablation/test_trained_folding.py` to recover the R1-only flow + longer training recipe it uses. Create `tools/_folding_regen.py` with a `regen_folding(out_path)` function that builds the same R1-only `JointUMNN1DFlow` (or the flow that test uses), trains it under the same longer recipe **capturing the full per-step loss list**, and writes a parquet with columns `step, loss`. Mirror the test's construction exactly so the trajectory matches the regression case (dips below the entropy floor ~0.99). Use the seed the test documents as folding-prone.

- [ ] **Step 3: Run the regen**

Run:
```bash
python tools/regen_figure_data.py
```
Expected: four `outputs/figure_data/8_{1,2,3,4}_cdsbi/` run-dirs exist, each with `diagnostics/coverage.parquet`, `diagnostics/marginal_pit_raw.parquet` (and for 8_2/8_3: `joint_mahalanobis_raw.parquet`; for 8_3: `jacobian_recovery_raw.parquet`), plus `model.pt`; and `outputs/figure_data/8_4_folding/folding_trajectory.parquet`.

- [ ] **Step 4: Verify the raw companions exist and are well-formed**

Run:
```bash
python -c "
import pandas as pd, glob
for p in sorted(glob.glob('outputs/figure_data/8_*_cdsbi/diagnostics/*_raw.parquet')):
    print(p, '->', list(pd.read_parquet(p).columns), len(pd.read_parquet(p)))
print('folding:', pd.read_parquet('outputs/figure_data/8_4_folding/folding_trajectory.parquet').shape)
"
```
Expected: PIT-raw under 8_1/8_2/8_3/8_4; jacobian-raw under 8_3 (9 rows for 3×3? no — d=2 → 4 rows); joint-mahalanobis-raw under 8_2/8_3; folding trajectory with `step,loss`.

- [ ] **Step 5: Commit the regen tooling**

```bash
git add tools/regen_figure_data.py tools/_folding_regen.py
git commit -m "feat(figures): regen tooling for stable figure-data run-dirs + folding trajectory"
```

(The `outputs/figure_data/` artifacts themselves are gitignored, like all of `outputs/`. The rendered figures committed in Phase B are what the manuscript needs.)

---

# PHASE B — Figure builders (E1–E10)

Each Phase-B task follows the same shape: write a smoke test (builder returns a Figure with the expected number of Axes from a fixture), implement the builder, add its manifest entry, render it against the **real** regenerated data, controller visual-acceptance, commit. The smoke test uses fixtures (no real run-dirs); the render + visual acceptance uses the Phase-A `outputs/figure_data/` paths.

**Manifest entries** are added to `configs/figures/manifest.yaml` per figure. **Smoke tests** all live in `tests/integration/test_e_figures_smoke.py` (append one test per builder).

## Task B1: E1 — §8.1 CDSBI calibration

**Files:**
- Create: `src/cdsbi/analysis/figures/figures/e1_loc_normal_calibration.py`
- Modify: `configs/figures/manifest.yaml`, `tests/integration/test_e_figures_smoke.py`

E1 = a two-panel figure: PIT histogram (left) + coverage curve with diagonal (right), CDSBI at §8.1.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/integration/test_e_figures_smoke.py`:

```python
"""Smoke tests for the E1–E10 figure builders, driven by synthetic fixtures."""
from __future__ import annotations

import matplotlib
import numpy as np

from cdsbi.analysis.figures.manifest import FigureSpec
from tests.figures_fixtures import (
    make_run_dir, make_sweep, write_marginal_pit_raw, write_coverage,
    write_jacobian_raw,
)


def _spec(**kw):
    base = dict(id="x", description="d", section="8", builder="m:render",
                output_pdf="figures/x.pdf", output_png="figures/x.png",
                source_runs=[], checkpoint_runs=[])
    base.update(kw)
    return FigureSpec(**base)


def test_e1_returns_two_panel_figure(tmp_path):
    from cdsbi.analysis.figures.figures.e1_loc_normal_calibration import render
    rd = tmp_path / "8_1"; rd.mkdir()
    write_marginal_pit_raw(rd, np.random.default_rng(0).uniform(0, 1, 500))
    write_coverage(rd, [-3.0, 0.0, 3.0], [0.5, 0.68, 0.9, 0.95])
    fig = render(_spec(source_runs=[str(rd)]))
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_e_figures_smoke.py::test_e1_returns_two_panel_figure -v`
Expected: FAIL — `ModuleNotFoundError: ...e1_loc_normal_calibration`.

- [ ] **Step 3: Write the builder**

`src/cdsbi/analysis/figures/figures/e1_loc_normal_calibration.py`:

```python
"""E1 — §8.1 CDSBI calibration: marginal PIT histogram + coverage curve."""
from __future__ import annotations

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures.manifest import FigureSpec
from cdsbi.analysis.figures import panels
from cdsbi.analysis.figures.data_io.figure_data import (
    load_pit_values, load_coverage_curve,
)


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    run = spec.source_runs[0]
    u = load_pit_values(run)
    nominal, empirical = load_coverage_curve(run)

    fig, (ax_pit, ax_cov) = plt.subplots(1, 2, figsize=style.SIZES["double_column"])
    panels.pit_histogram(ax_pit, u, bins=20)
    ax_pit.set_title("Marginal PIT")
    panels.diagonal_reference(ax_cov, lo=float(nominal.min()), hi=1.0)
    panels.coverage_curve(ax_cov, nominal, empirical,
                          color=style.METHOD_STYLE["cd_sbi"]["color"], marker="o",
                          label="CD-SBI")
    ax_cov.set_title("Coverage")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `pytest tests/integration/test_e_figures_smoke.py::test_e1_returns_two_panel_figure -v`
Expected: PASS

- [ ] **Step 5: Add the manifest entry**

Append to `configs/figures/manifest.yaml`:

```yaml
e1_loc_normal_calibration:
  description: "§8.1 CDSBI 1-D calibration: marginal PIT + coverage curve"
  section: "8.1"
  source_runs:
    - outputs/figure_data/8_1_cdsbi
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures.e1_loc_normal_calibration:render"
  output:
    pdf: figures/e1_loc_normal_calibration.pdf
    png: figures/e1_loc_normal_calibration.png
```

- [ ] **Step 6: Render against real data**

Run: `python -m cdsbi.analysis.figures.render --fig e1_loc_normal_calibration`
Expected: writes `figures/e1_loc_normal_calibration.{pdf,png}` (nonzero).

- [ ] **Step 7: Visual acceptance (controller)**

The controller `Read`s `figures/e1_loc_normal_calibration.png` and confirms: PIT histogram is ~flat at density 1 (calibrated); coverage curve hugs the diagonal; mathtext renders; CDSBI navy. **Message check:** "CDSBI is calibrated at d=1." If it fails, fix and re-render.

- [ ] **Step 8: Commit**

```bash
git add src/cdsbi/analysis/figures/figures/e1_loc_normal_calibration.py configs/figures/manifest.yaml tests/integration/test_e_figures_smoke.py figures/e1_loc_normal_calibration.pdf figures/e1_loc_normal_calibration.png
git commit -m "feat(figures): E1 §8.1 CDSBI calibration figure

visual: OK — flat PIT, coverage on diagonal"
```

## Task B2: E2 — §8.1 cross-method coverage

**Files:** Create `src/cdsbi/analysis/figures/figures/e2_loc_normal_cross_method.py`; modify manifest + smoke test.

E2 = one panel: coverage curves for all 5 methods at §8.1 medium budget + diagonal. Sources the existing §8.1 baseline sweep (each method's medium/seed=0 `coverage.parquet`).

- [ ] **Step 1: Write the failing smoke test** — append to `tests/integration/test_e_figures_smoke.py`:

```python
def test_e2_returns_single_panel_with_method_lines(tmp_path):
    from cdsbi.analysis.figures.figures.e2_loc_normal_cross_method import render
    # Build a mini "sweep" with per-method coverage parquets.
    root = tmp_path / "sweep"
    for method in ("cd_sbi", "npe", "nle"):
        rd = root / f"method={method},budget=medium,seed=0"
        rd.mkdir(parents=True)
        (rd / "STATUS").write_text("OK")
        write_coverage(rd, [-3.0, 0.0, 3.0], [0.5, 0.68, 0.9, 0.95])
        import pandas as pd
        pd.DataFrame([{"method": method, "budget_name": "medium", "seed": 0}]).to_parquet(rd / "index_row.parquet")
    fig = render(_spec(source_runs=[str(root)], section="8.1"))
    ax = fig.axes[0]
    # 3 method curves + 1 diagonal reference.
    assert len(ax.lines) == 4
```

- [ ] **Step 2: Run to verify fail.** `pytest tests/integration/test_e_figures_smoke.py::test_e2_returns_single_panel_with_method_lines -v` → FAIL (no module).

- [ ] **Step 3: Write the builder**

`src/cdsbi/analysis/figures/figures/e2_loc_normal_cross_method.py`:

```python
"""E2 — §8.1 cross-method coverage curves (all 5 methods, medium budget)."""
from __future__ import annotations

import glob
from pathlib import Path

from cdsbi.analysis.figures import style, panels
from cdsbi.analysis.figures.manifest import FigureSpec
from cdsbi.analysis.figures.data_io.figure_data import load_coverage_curve


def _method_of(run_dir: str) -> str | None:
    name = Path(run_dir).name
    for token in name.split(","):
        if token.startswith("method="):
            return token.split("=", 1)[1]
    return None


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    # Collect medium-budget seed=0 run-dirs per method from the sweep root(s).
    by_method = {}
    for root in spec.source_runs:
        for rd in glob.glob(str(Path(root) / "*")):
            m = _method_of(rd)
            if m and "budget=medium" in Path(rd).name and "seed=0" in Path(rd).name:
                by_method.setdefault(m, rd)

    fig, ax = plt.subplots(figsize=style.SIZES["single_column"])
    # diagonal first so it sits behind the curves
    panels.diagonal_reference(ax, lo=0.5, hi=1.0)
    for m in style.CANONICAL_ORDER:
        if m not in by_method:
            continue
        nominal, empirical = load_coverage_curve(by_method[m])
        st = style.METHOD_STYLE[m]
        panels.coverage_curve(ax, nominal, empirical, color=st["color"],
                              marker=st["marker"], label=st["label"])
    ax.legend(fontsize=7)
    ax.set_title("§8.1 coverage by method")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run smoke test → PASS.** `pytest tests/integration/test_e_figures_smoke.py::test_e2_returns_single_panel_with_method_lines -v`

- [ ] **Step 5: Manifest entry** — append (source is the existing §8.1 baseline sweep dir; the implementer fills the actual timestamped path from `outputs/8_1_baseline_sweep/` — use the one in CLAUDE.md's manuscript-to-code mapping, `2026-05-27_00-57-31`):

```yaml
e2_loc_normal_cross_method:
  description: "§8.1 cross-method coverage curves (5 methods, medium budget)"
  section: "8.1"
  source_runs:
    - outputs/8_1_baseline_sweep/2026-05-27_00-57-31
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures.e2_loc_normal_cross_method:render"
  output:
    pdf: figures/e2_loc_normal_cross_method.pdf
    png: figures/e2_loc_normal_cross_method.png
```

- [ ] **Step 6: Render.** `python -m cdsbi.analysis.figures.render --fig e2_loc_normal_cross_method`. If the pinned sweep dir is absent locally, re-render after confirming the correct `outputs/8_1_baseline_sweep/<timestamp>` path and updating the manifest.

- [ ] **Step 7: Visual acceptance (controller).** Read the PNG. **Message:** "all methods are near-diagonal at d=1; CDSBI tightest." Confirm 5 curves + diagonal, colour convention, legend legible.

- [ ] **Step 8: Commit** (builder + manifest + test + the two artifacts), message `feat(figures): E2 §8.1 cross-method coverage` + `visual:` line.

## Task B3: E3 — §8.2 joint diagnostics

**Files:** Create `e3_joint_diagnostics.py`; modify manifest + smoke test. Two panels: joint-Mahalanobis PIT histogram (left, from `load_joint_mahalanobis_sq` → pooled, transformed to PIT via χ²₂ CDF) + coverage tile (right). Source `outputs/figure_data/8_2_cdsbi`.

- [ ] **Step 1: Failing smoke test** — append:

```python
def test_e3_returns_two_panels(tmp_path):
    from cdsbi.analysis.figures.figures.e3_joint_diagnostics import render
    import pandas as pd, numpy as np
    rd = tmp_path / "8_2"; (rd / "diagnostics").mkdir(parents=True)
    rng = np.random.default_rng(0)
    # joint_mahalanobis_raw: r² ~ χ²_2 at two θ_0
    frames = [pd.DataFrame({"theta_0_repr": rep, "r_sq": rng.chisquare(2, 400)})
              for rep in ("[0.0, 0.0]", "[2.0, -1.0]")]
    pd.concat(frames, ignore_index=True).to_parquet(rd / "diagnostics" / "joint_mahalanobis_raw.parquet")
    write_coverage(rd, [-2.0, 0.0, 2.0], [0.5, 0.68, 0.9, 0.95])
    fig = render(_spec(source_runs=[str(rd)], section="8.2"))
    assert len(fig.axes) >= 2
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Write the builder**

```python
"""E3 — §8.2 joint diagnostics: joint-Mahalanobis PIT + coverage-error tile."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from cdsbi.analysis.figures import style, panels
from cdsbi.analysis.figures.manifest import FigureSpec
from cdsbi.analysis.figures.data_io.figure_data import (
    load_joint_mahalanobis_sq, load_coverage_tile,
)


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    run = spec.source_runs[0]
    r_sq = load_joint_mahalanobis_sq(run)
    pooled = np.concatenate(list(r_sq.values()))
    pit = chi2(df=2).cdf(pooled)            # χ²_d PIT; flat if calibrated
    theta0, alpha, error = load_coverage_tile(run)

    fig, (ax_pit, ax_tile) = plt.subplots(1, 2, figsize=style.SIZES["double_column"])
    panels.pit_histogram(ax_pit, pit, bins=20)
    ax_pit.set_title("Joint Mahalanobis PIT")
    panels.coverage_tile(ax_tile, theta0, alpha, error)
    ax_tile.set_title("Coverage error")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run smoke → PASS.**
- [ ] **Step 5: Manifest entry** (`section: "8.2"`, source `outputs/figure_data/8_2_cdsbi`).
- [ ] **Step 6: Render** `--fig e3_joint_diagnostics`.
- [ ] **Step 7: Visual acceptance.** **Message:** "the 2-D pivot is jointly calibrated (flat χ² PIT, near-zero coverage error across the grid)."
- [ ] **Step 8: Commit** + `visual:` line.

## Task B4: E4 — §8.3 Jacobian recovery

**Files:** Create `e4_jacobian_recovery.py`; modify manifest + smoke test. One panel: `jacobian_recovery_scatter(trained, truth)`. Source `outputs/figure_data/8_3_cdsbi`.

- [ ] **Step 1: Failing smoke test** — append:

```python
def test_e4_returns_scatter_panel(tmp_path):
    from cdsbi.analysis.figures.figures.e4_jacobian_recovery import render
    import numpy as np
    rd = tmp_path / "8_3"; rd.mkdir()
    j_true = np.array([[1.0, 0.0], [-0.4, 0.9]])
    write_jacobian_raw(rd, j_true + 0.01, j_true)
    fig = render(_spec(source_runs=[str(rd)], section="8.3"))
    assert len(fig.axes) == 1
    assert len(fig.axes[0].collections) == 1   # the scatter
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Write the builder**

```python
"""E4 — §8.3 Jacobian recovery: trained E[∂r/∂θ] vs closed-form L⁻¹."""
from __future__ import annotations

from cdsbi.analysis.figures import style, panels
from cdsbi.analysis.figures.manifest import FigureSpec
from cdsbi.analysis.figures.data_io.figure_data import load_jacobian_matrices


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt
    j_emp, j_true = load_jacobian_matrices(spec.source_runs[0])
    fig, ax = plt.subplots(figsize=style.SIZES["square"])
    panels.jacobian_recovery_scatter(ax, j_emp, j_true)
    ax.set_title("§8.3 Jacobian recovery")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run smoke → PASS.**
- [ ] **Step 5: Manifest entry** (`section: "8.3"`, source `outputs/figure_data/8_3_cdsbi`).
- [ ] **Step 6: Render** `--fig e4_jacobian_recovery`.
- [ ] **Step 7: Visual acceptance.** **Message:** "trained Jacobian elements lie on y=x against L⁻¹ — KR uniqueness confirmed."
- [ ] **Step 8: Commit** + `visual:` line.

## Task B5: E5 — §8.4 doubly-monotone calibration

**Files:** Create `e5_exp_rate_calibration.py`; modify manifest + smoke test. Three panels: PIT histogram + coverage curve + a single final-loss-vs-floor bar (`loss_bar_with_floor` with one series). Source `outputs/figure_data/8_4_cdsbi`. The entropy floor is 0.99 (per the manuscript / `ExponentialRate.entropy_lower_bound()`).

- [ ] **Step 1: Failing smoke test** — append:

```python
def test_e5_returns_three_panels(tmp_path):
    from cdsbi.analysis.figures.figures.e5_exp_rate_calibration import render
    import numpy as np
    rd = make_run_dir(tmp_path, method="cd_sbi", budget_name="medium", seed=0,
                      coverage_error_max=0.03, final_loss=0.985,
                      loss_history_tail=[0.99, 0.985, 0.985])
    write_marginal_pit_raw(rd, np.random.default_rng(0).uniform(0, 1, 500))
    write_coverage(rd, [0.5, 1.5, 2.5], [0.5, 0.68, 0.9, 0.95])
    fig = render(_spec(source_runs=[str(rd)], section="8.4"))
    assert len(fig.axes) == 3
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Write the builder**

```python
"""E5 — §8.4 doubly-monotone calibration: PIT + coverage + final-loss-vs-floor."""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures import style, panels
from cdsbi.analysis.figures.manifest import FigureSpec
from cdsbi.analysis.figures.data_io.figure_data import (
    load_pit_values, load_coverage_curve, load_loss_tail_mean,
)

ENTROPY_FLOOR = 0.99


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    run = spec.source_runs[0]
    u = load_pit_values(run)
    nominal, empirical = load_coverage_curve(run)
    loss = load_loss_tail_mean(run)

    fig, (ax_pit, ax_cov, ax_loss) = plt.subplots(1, 3, figsize=(10.5, 3.0))
    panels.pit_histogram(ax_pit, u, bins=20); ax_pit.set_title("Marginal PIT")
    panels.diagonal_reference(ax_cov, lo=float(nominal.min()), hi=1.0)
    panels.coverage_curve(ax_cov, nominal, empirical,
                          color=style.METHOD_STYLE["cd_sbi"]["color"], marker="o")
    ax_cov.set_title("Coverage")
    panels.loss_bar_with_floor(ax_loss, {"CD-SBI": np.array([loss])},
                               floor=ENTROPY_FLOOR, group_labels=["medium"])
    ax_loss.set_title("Final loss vs entropy floor")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run smoke → PASS.**
- [ ] **Step 5: Manifest entry** (`section: "8.4"`, source `outputs/figure_data/8_4_cdsbi`).
- [ ] **Step 6: Render** `--fig e5_exp_rate_calibration`.
- [ ] **Step 7: Visual acceptance.** **Message:** "the doubly-monotone flow is calibrated AND sits at the entropy floor in the non-additive case."
- [ ] **Step 8: Commit** + `visual:` line.

## Task B6: E6 — §8.4 (R2) ablation tail-mean bars

**Files:** Create `e6_r2_ablation_bars.py`; modify manifest + smoke test. `loss_bar_with_floor` comparing R1+R2 vs R1-only tail-mean loss across the four budgets, floor at 0.99. Source: the existing `outputs/8_4_ablation/2026-05-27_23-55-59/` sweep (run-dirs carry `model.pt` with `loss_history_tail`; the two arms are distinguished by flow in the run-dir name / index_row).

- [ ] **Step 1: Failing smoke test** — append (fixture: two arms × 4 budgets of run-dirs with loss tails). Assert `len(fig.axes[0].patches) == 8` and one floor line.
- [ ] **Step 2–4:** implement a builder that, for each arm (R1+R2 = `doubly_monotone`, R1-only = `joint_umnn`) and each budget, reads `load_loss_tail_mean` over the matching run-dirs and calls `loss_bar_with_floor({"R1+R2": [...4...], "R1 only": [...4...]}, floor=0.99, group_labels=["S","M","L","XL"])`. The builder maps run-dirs to (arm, budget) by parsing the run-dir name (`flow=`/`budget=` tokens) — reuse the token-parse helper pattern from E2.
- [ ] **Step 5:** Manifest entry (`section: "8.4"`, source the ablation sweep dir).
- [ ] **Step 6–7:** Render + visual acceptance. **Message:** "R1+R2 sits at the floor at every budget; R1-only stays ~0.4 above — (R2) is load-bearing."
- [ ] **Step 8:** Commit + `visual:` line.

## Task B7: E7 — §8.4 catastrophic folding

**Files:** Create `e7_catastrophic_folding.py`; modify manifest + smoke test. `loss_trajectory_with_floor` of the R1-only longer-recipe run, dipping below the floor. Source: `outputs/figure_data/8_4_folding/folding_trajectory.parquet` (Task A5).

- [ ] **Step 1: Failing smoke test** — append (fixture writes a `folding_trajectory.parquet` with `step,loss` dipping below 0.99). Assert single panel, `len(ax.lines) == 2` (trajectory + floor).
- [ ] **Step 2–4:** builder reads the parquet (`pd.read_parquet`, columns `step,loss`) and calls `loss_trajectory_with_floor(ax, {"R1 only (long recipe)": loss}, floor=0.99)`. NOTE: the source is a single parquet file, not a run-dir; the builder reads `spec.source_runs[0]` as a parquet path. Document this in the builder docstring.
- [ ] **Step 5:** Manifest entry (`section: "8.4"`, source `outputs/figure_data/8_4_folding/folding_trajectory.parquet`).
- [ ] **Step 6–7:** Render + visual acceptance. **Message:** "without architectural (R2), training folds the density and the loss drops below the entropy floor — impossible for a valid density."
- [ ] **Step 8:** Commit + `visual:` line.

## Task B8: E8 — headline cross-method summary

**Files:** Create `e8_headline_summary.py`; modify manifest + smoke test. `cross_method_summary_log_y`: `coverage_error_max` per method × {8.1, 8.2, 8.3, 8.4}, log-y, floor band at 0.02. Source: the four existing baseline sweeps (via F0 `load_aggregates`, medium budget, mean over seeds).

- [ ] **Step 1: Failing smoke test** — append: build 4 mini-sweeps (one per experiment label) via `make_sweep`; assert single panel, log-y, ≥1 line per method.
- [ ] **Step 2–4:** builder takes `spec.source_runs` = the four sweep roots in §8.1→8.4 order (and `spec.metadata`-free); for each, `load_aggregates([root])`, filter `budget_name=="medium"`, group by method → mean `coverage_error_max`; assemble `{method: {label: value}}` and call `cross_method_summary_log_y(ax, data, floor=0.02)`. Experiment labels come from a fixed list `["8.1","8.2","8.3","8.4"]` aligned to source order. (If F0 `load_aggregates` import is needed: `from cdsbi.analysis.figures.data_io.aggregates import load_aggregates`.)
- [ ] **Step 5:** Manifest entry (`section: "8.5"`, four `source_runs` = the baseline sweep dirs from CLAUDE.md's mapping).
- [ ] **Step 6–7:** Render + visual acceptance. **Message:** "the CDSBI→others gap widens monotonically 8.1→8.4; CDSBI alone stays at the floor."
- [ ] **Step 8:** Commit + `visual:` line.

## Task B9: E9 — budget saturation

**Files:** Create `e9_budget_saturation.py`; modify manifest + smoke test. `metric_vs_budget`: `coverage_error_max` vs `actual_params_total` per method, for one representative experiment (use §8.2). Source: the §8.2 baseline sweep (all budgets, mean over seeds).

- [ ] **Step 1: Failing smoke test** — append: mini-sweep across budgets; assert single panel, log-x, ≥1 line/method.
- [ ] **Step 2–4:** builder `load_aggregates([root])`, group by (method, budget) → mean `coverage_error_max` and mean `actual_params_total`; assemble `{method: (params_array, metric_array)}` sorted by params; call `metric_vs_budget(ax, series)`.
- [ ] **Step 5:** Manifest entry (`section: "8.5"`, source the §8.2 baseline sweep dir).
- [ ] **Step 6–7:** Render + visual acceptance. **Message:** "CDSBI saturates at the floor by the smallest budget; others don't improve much with scale."
- [ ] **Step 8:** Commit + `visual:` line.

## Task B10: E10 — per-experiment boxplots

**Files:** Create `e10_per_experiment_boxplots.py`; modify manifest + smoke test. 2×2 = 4 panels, one per §8.x, each a `boxplot_per_method` of `coverage_error_max` pooled over seeds × budgets. Source: the four baseline sweeps.

- [ ] **Step 1: Failing smoke test** — append: 4 mini-sweeps; assert `len(fig.axes) == 4`.
- [ ] **Step 2–4:** builder takes four `source_runs`; for each, `load_aggregates([root])` → `{method: coverage_error_max values}` pooled; one `boxplot_per_method` per subplot; titles "§8.1"…"§8.4".
- [ ] **Step 5:** Manifest entry (`section: "8.5"`, four `source_runs`).
- [ ] **Step 6–7:** Render + visual acceptance. **Message:** "CDSBI's distribution sits at the floor with low variance in every experiment."
- [ ] **Step 8:** Commit + `visual:` line.

---

## Task B11: Regenerate gallery + figure-suite verification

- [ ] **Step 1:** `python -m cdsbi.analysis.figures.render --gallery` — regenerate `figures/README.md` (now lists e1–e10 + _hello, grouped by section).
- [ ] **Step 2:** `pytest -q` — full fast suite green (adds the A* diagnostic tests, the figure_data loader tests, and the 10 E smoke tests). Report counts.
- [ ] **Step 3:** `du -sh figures/` — confirm under the 9 MB budget; if any single figure PDF > 400 KB or PNG > 150 KB, reduce DPI/decimate per the spec.
- [ ] **Step 4:** Controller views `figures/README.md` to confirm the gallery embeds all 11 PNGs correctly.
- [ ] **Step 5:** Commit the regenerated gallery: `git add figures/README.md && git commit -m "docs(figures): regenerate gallery with E1–E10"`.

---

# PHASE C — Manuscript integration

The manuscript `cd_sbi_v7.tex` is tables-only today. Insert the 10 figures with captions near their result sections. Section anchors (verified): §8.1 `\label{subsec:8.1}` (~line 2449), §8.2 `subsec:8.2` (~2580), §8.3 `subsec:8.3` (~2738), §8.4 `subsec:8.4` (~2922), §8.5 `subsec:8.5` (~3174). Build with the CLAUDE.md LaTeX cycle (absolute `/usr/bin/pdflatex`).

## Task C1: Insert per-section figures E1–E7 (§8.1–§8.4)

**Files:** Modify `cd_sbi_v7.tex`.

- [ ] **Step 1:** Insert E1 + E2 in §8.1. After the §8.1 cross-method coverage table, add:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/e1_loc_normal_calibration.pdf}
\caption{\S\ref{subsec:8.1} CDSBI calibration in the 1-D location-normal model:
the marginal PIT is uniform (left) and empirical coverage tracks nominal
along the diagonal (right), at the medium budget.}
\label{fig:e1}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.6\linewidth]{figures/e2_loc_normal_cross_method.pdf}
\caption{\S\ref{subsec:8.1} coverage by method (medium budget). CDSBI sits on
the diagonal; the Bayesian and ratio baselines deviate.}
\label{fig:e2}
\end{figure}
```

- [ ] **Step 2:** Insert E3 in §8.2 (after the §8.2 joint-diagnostics discussion), `\includegraphics[width=\linewidth]{figures/e3_joint_diagnostics.pdf}`, caption referencing joint Mahalanobis PIT + coverage tile, `\label{fig:e3}`.
- [ ] **Step 3:** Insert E4 in §8.3, `\includegraphics[width=0.55\linewidth]{figures/e4_jacobian_recovery.pdf}`, caption referencing Theorem A-d / KR uniqueness, `\label{fig:e4}`.
- [ ] **Step 4:** Insert E5, E6, E7 in §8.4 — E5 after the calibration table, E6 + E7 in the ablation paragraph. Captions: E5 doubly-monotone calibration + entropy floor; E6 R1+R2 vs R1-only across budgets; E7 catastrophic folding below the floor. Labels `fig:e5`, `fig:e6`, `fig:e7`.
- [ ] **Step 5:** Add a `\ref{fig:eN}` sentence-level cross-reference in each section's prose so figures aren't orphaned (LaTeX warns on unreferenced floats only if `\label` unused, but a textual reference aids the reader). E.g. in §8.1: "(Fig.~\ref{fig:e1})".
- [ ] **Step 6:** Build: `/usr/bin/pdflatex cd_sbi_v7 && /usr/bin/bibtex cd_sbi_v7 && /usr/bin/pdflatex cd_sbi_v7 && /usr/bin/pdflatex cd_sbi_v7`. Expected: clean build, no missing-figure errors, no undefined `\ref`.
- [ ] **Step 7:** Commit: `git add cd_sbi_v7.tex && git commit -m "manuscript(8.1-8.4): insert empirical figures E1-E7"`.

## Task C2: Insert synthesis figures E8–E10 (§8.5) + final build

**Files:** Modify `cd_sbi_v7.tex`.

- [ ] **Step 1:** In §8.5 (synthesis), after the claims/summary tables, insert E8, E9, E10:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.7\linewidth]{figures/e8_headline_summary.pdf}
\caption{Worst-case coverage error (log scale) by method across the four
experiments. The CDSBI--baseline gap widens monotonically
\S\ref{subsec:8.1}$\to$\S\ref{subsec:8.4}; only CDSBI stays at the
Monte-Carlo floor (band).}
\label{fig:e8}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.6\linewidth]{figures/e9_budget_saturation.pdf}
\caption{Coverage error vs parameter budget (\S\ref{subsec:8.2}). CDSBI
saturates at the floor by the smallest budget tested.}
\label{fig:e9}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/e10_per_experiment_boxplots.pdf}
\caption{Per-experiment distribution of worst-case coverage error across
seeds and budgets. CDSBI is at the floor with low variance throughout.}
\label{fig:e10}
\end{figure}
```

- [ ] **Step 2:** Add textual `\ref{fig:e8}`/`e9`/`e10` references in the §8.5 prose.
- [ ] **Step 3:** Build the full LaTeX cycle (as in C1 Step 6). Confirm clean, the PDF page count grew, and `grep -i "undefined\|warning: .*float\|LaTeX Warning: Reference" cd_sbi_v7.log` shows no undefined references.
- [ ] **Step 4:** Commit: `git add cd_sbi_v7.tex && git commit -m "manuscript(8.5): insert synthesis figures E8-E10"`.

---

## Known data-sourcing dependencies (carried context)

- E2/E8/E9/E10 pin **existing** baseline-sweep run-dirs (gitignored). If those timestamped dirs are absent on the execution machine, re-run the relevant `8_x_baseline_sweep` multirun (CLAUDE.md has the command) or re-point the manifest. The figures themselves (committed PDFs/PNGs) don't need the dirs once rendered.
- E1/E3/E4/E5 depend on Phase-A's `outputs/figure_data/` re-runs.
- E7 depends on the folding-trajectory regen (A5 Step 2).
- C1/C3 conceptual figures and the model-weights gap remain F3 scope — untouched here.

## Self-review notes

- **Spec coverage:** E1–E10 all built (B1–B10) and inserted into the manuscript (C1–C2); the data-sourcing prerequisite for the PIT/Jacobian figures is resolved (A1–A5). Matches the spec's F2 = catalogue E1–E10 + per-section/synthesis manuscript integration.
- **Builder contract:** every E-builder is `render(spec) -> Figure`, creates its Figure, composes F1 panels, returns it; the render CLI saves. No `savefig` in any builder.
- **Type consistency:** `data_io.figure_data` loader names (`load_pit_values`, `load_joint_mahalanobis_sq`, `load_jacobian_matrices`, `load_coverage_curve`, `load_coverage_tile`, `load_loss_tail_mean`) are used identically in their tests and the builders that import them. Manifest entries' `builder` strings match the `module:render` of each file. Companion-parquet names (`marginal_pit_raw`, `joint_mahalanobis_raw`, `jacobian_recovery_raw`) are identical across the diagnostic meta keys, the `_write_raw_companion` writer, the fixtures, and the loaders.
- **Visual acceptance:** every figure (B1–B10) has an explicit controller `Read`-the-PNG step with a figure-specific message check, per the spec protocol; B11 + the gallery view close the loop.
- **Placeholder caution:** Tasks B6–B10 give the algorithm + exact panel calls + manifest/section but compress the boilerplate test/commit steps (identical in shape to B1–B5's fully-spelled versions) to keep the plan readable; the builder logic and data sources are concrete. A2/A3 reuse A1's `_write_raw_companion`. A5 Step 2 (folding regen) requires reading `test_trained_folding.py` to mirror its recipe — flagged explicitly as a regen utility, not invented code.
- **F2 scope fidelity:** no F3 conceptual figures (C1–C6), no model-weights work; only the empirical catalogue + its data + manuscript insertion.
