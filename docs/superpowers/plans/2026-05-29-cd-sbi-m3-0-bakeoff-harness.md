# CD-SBI M3.0 — Stage-B bake-off shared harness

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared evaluation harness that judges every Stage-B device arm (I-A/I-B/II-A) uniformly against the Stage-A oracle control: a `SufficiencyRecovery` diagnostic, an arm-aware `FloorIntegrity` diagnostic, the plumbing to expose a trained model's learned features to diagnostics, and the conversion of the M2 cheat-smoke into a green cheat-capture regression test.

**Architecture:** Diagnostics receive `(trained, simulator, ...)`. To measure whether a *learned* summary kept the sufficient information, a diagnostic needs the trained summary's features — so we add an optional `encode_fn` closure to `PivotBasedProcedure` (set by `CDSBIRunner.fit`) and an `oracle_summary(x)` method to the simulator (the ground-truth `(log s², X̄)`). `SufficiencyRecovery` compares them by Spearman (monotone-invariant). `FloorIntegrity` is arm-aware: for an NF-MLE loss it records `final_loss − H` (cheat = sinking below the conditional-entropy floor `H`); for a non-NF-MLE loss (II-A's energy loss) it no-ops the floor comparison. Both write a parquet and reduce to scalar index-row columns, mirroring `MarginalCDRecovery`. No new arm is built here.

**Tech Stack:** PyTorch, numpy/scipy (`spearmanr`, `pearsonr`), pandas (diagnostic parquet → index-row), pytest.

**Spec:** `docs/superpowers/specs/2026-05-29-cd-sbi-stage-b-device-bakeoff-design.md` §2 (shared harness) + §1 (the cheat the regression test captures).

---

## Conventions / decisions locked

- **Diagnostic protocol** (`src/cdsbi/diagnostics/base.py`): `name: str`; `__call__(self, trained, simulator, eval_data=None, x_per_theta=None) -> DiagnosticResult`; `DiagnosticResult(name, value, passed, noise_floor, n_samples, meta)`.
- **Trained-feature access:** `trained.procedure.encode_fn` is `Callable[[Tensor], Tensor] | None`. When present, `encode_fn(x) -> feats (n, d_out)` runs the trained conditioner in eval mode (no grad). `None` for procedures without a learned summary (or non-pivot procedures). Diagnostics that need learned features gate on `encode_fn is not None`.
- **Oracle summary:** `simulator.oracle_summary(x) -> Tensor (n, d_oracle)` returns the ground-truth sufficient statistic in the SAME order/scale the flow expects: for `NormalUnknownMeanVar`, `(log s², X̄)`. Diagnostics gate on `hasattr(simulator, "oracle_summary")`.
- **`SufficiencyRecovery`** value is a 1-row DataFrame (scalar summary across a single eval draw), columns `sufficiency_min_spearman`, plus per-oracle-coordinate `spearman_*` / `pearson_*`. `passed = sufficiency_min_spearman > 0.9`.
- **`FloorIntegrity`** value is a 1-row DataFrame, columns `final_loss`, `entropy_floor` (NaN if not applicable), `floor_margin = final_loss − H` (NaN if N/A), `cheats` (bool). For NF-MLE: `passed = not cheats` (i.e. `final_loss > H − tol`, tol=0.10). For non-NF-MLE losses: `cheats=False`, `passed=True`, `meta` notes "floor N/A for <loss>".
- **Floor tolerance** `tol = 0.10` (matches the §8.4 / M1 floor checks).
- **No-op contract** (matches `MarginalCDRecovery`/`JacobianRecovery`): return an empty-DataFrame `DiagnosticResult(passed=True, meta={"reason": ...})` when inapplicable; the empty DataFrame survives `to_parquet`, and the index-row reducer's `len(df)` guard keeps the columns absent for non-applicable runs.

---

## File structure

```
NEW
  src/cdsbi/diagnostics/sufficiency_recovery.py     # SufficiencyRecovery
  src/cdsbi/diagnostics/floor_integrity.py          # FloorIntegrity (arm-aware)
  tests/unit/test_sufficiency_recovery.py
  tests/unit/test_floor_integrity.py

MODIFY
  src/cdsbi/confidence_set/procedures.py            # PivotBasedProcedure: optional encode_fn
  src/cdsbi/methods/cd_sbi.py                        # fit(): set procedure.encode_fn (eval-mode conditioner)
  src/cdsbi/simulators/normal_unknown_mean_var.py    # oracle_summary(x) -> (log s², X̄)
  src/cdsbi/experiments/run.py                        # diagnostics list + index-row reduction
  tests/integration/test_mu_sigma_stage_b_smoke.py    # convert to cheat-capture regression test
```

No new arm code (I-A/I-B/II-A are M3.1–M3.3). No change to `SingleIndexMonotoneFlow`, the loss, or the existing diagnostics.

---

## Task 1: expose trained features + oracle summary

**Files:**
- Modify: `src/cdsbi/confidence_set/procedures.py` (`PivotBasedProcedure.__init__`)
- Modify: `src/cdsbi/methods/cd_sbi.py` (`fit()`)
- Modify: `src/cdsbi/simulators/normal_unknown_mean_var.py`
- Test: `tests/unit/test_sufficiency_recovery.py` (a plumbing test here; the diagnostic in Task 2)

READ `src/cdsbi/confidence_set/procedures.py` (`PivotBasedProcedure.__init__(self, pivot_fn, d_theta, theta_range=...)` at ~line 315) and `src/cdsbi/methods/cd_sbi.py` (the `pivot_fn` closure + `PivotBasedProcedure(...)` construction at ~line 149–162, and the conditioner's eval switch). Also `normal_unknown_mean_var.py` `_suff_stats`.

- [ ] **Step 1: Write the failing test**

```python
"""encode_fn plumbing + simulator.oracle_summary."""
import torch


def test_oracle_summary_is_log_s2_and_xbar():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    sim = NormalUnknownMeanVar()
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0]])  # mean 2, s²(ddof1)
    o = sim.oracle_summary(x)
    assert o.shape == (1, 2)
    s2 = float(x.var(dim=-1, unbiased=True))
    assert abs(float(o[0, 0]) - torch.log(torch.tensor(s2)).item()) < 1e-5   # log s²
    assert abs(float(o[0, 1]) - 2.0) < 1e-6                                   # X̄


def test_procedure_exposes_encode_fn_after_fit():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi import CDSBIRunner
    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=16)
    cond = DeepSetsConditioner(n_iid=sim.n_iid, d_out=2, hidden=16)
    trained = CDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss()).fit(
        simulator=sim, config={"lr": 1e-3, "batch_size": 64, "n_steps": 5,
                               "n_train": 500, "optimizer": "adam", "fresh_batch": False}, seed=0)
    assert trained.procedure.encode_fn is not None
    import numpy as np
    _, x = sim.sample(32, np.random.default_rng(0))
    feats = trained.procedure.encode_fn(x)
    assert feats.shape == (32, 2)
    assert not feats.requires_grad      # eval/no-grad path
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_sufficiency_recovery.py -v`
Expected: FAIL (`oracle_summary` missing; `encode_fn` attribute missing).

- [ ] **Step 3: Implement**

(3a) `PivotBasedProcedure.__init__` — add an optional `encode_fn` (default `None`). The current body sets `self.pivot_fn`, `self.d_theta`, `self.theta_range` (and any cache init) — keep ALL existing lines, just add the new param + assignment:
```python
    def __init__(self, pivot_fn: Callable, d_theta: int, theta_range: tuple = (-20.0, 20.0),
                 encode_fn: Callable | None = None):
        self.pivot_fn = pivot_fn
        self.d_theta = d_theta
        self.theta_range = theta_range          # <-- preserve existing line(s)
        self.encode_fn = encode_fn              # <-- new
```
(`from __future__ import annotations` and `Callable` are already imported in this file, so `Callable | None` needs no new import. Do NOT drop `self.theta_range` or any other existing assignment.)

(3b) `cd_sbi.py` `fit()` — after the `pivot_fn` closure and after `self.conditioner.eval()` (the conditioner is already switched to eval for nn.Module conditioners in M2), add an `encode_fn` closure and pass it to the procedure:
```python
        def encode_fn(x: torch.Tensor) -> torch.Tensor:
            x = x.to(device)
            with torch.no_grad():
                feats, _ = conditioner.encode(x)
            return feats

        procedure = PivotBasedProcedure(pivot_fn=pivot_fn, d_theta=simulator.d_theta,
                                        encode_fn=encode_fn)
```
(Replace the existing `procedure = PivotBasedProcedure(pivot_fn=pivot_fn, d_theta=simulator.d_theta)` line.)

(3c) `normal_unknown_mean_var.py` — add `oracle_summary` (reuses `_suff_stats`; same `(log s², X̄)` order/scale as `SufficientStatConditioner`):
```python
    def oracle_summary(self, x: torch.Tensor) -> torch.Tensor:
        """Ground-truth sufficient statistic (log s², X̄), shape (n, 2) — the order
        and scale the flow's features use. Used by the SufficiencyRecovery diagnostic."""
        xbar, s2 = self._suff_stats(x)
        return torch.cat([torch.log(s2.clamp_min(1e-12)), xbar], dim=-1)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_sufficiency_recovery.py -v`
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/confidence_set/procedures.py src/cdsbi/methods/cd_sbi.py \
        src/cdsbi/simulators/normal_unknown_mean_var.py tests/unit/test_sufficiency_recovery.py
git commit -m "feat(harness): expose trained encode_fn on procedure + simulator.oracle_summary"
```

---

## Task 2: `SufficiencyRecovery` diagnostic

**Files:**
- Create: `src/cdsbi/diagnostics/sufficiency_recovery.py`
- Test: `tests/unit/test_sufficiency_recovery.py` (append)

Measures whether the trained summary retained the sufficient information: Spearman (monotone-invariant) of each oracle coordinate against the learned feature set; reports the min across oracle coordinates as the headline. Gated to no-op without `encode_fn` or `oracle_summary`.

- [ ] **Step 1: Write the failing tests (append)**

```python
def test_sufficiency_recovery_perfect_when_encode_is_oracle():
    # encode_fn = the oracle itself ⇒ Spearman ≈ 1 on both coords.
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.diagnostics.sufficiency_recovery import SufficiencyRecovery
    sim = NormalUnknownMeanVar()

    class _Proc:
        d_theta = 2
        def __init__(self, sim): self.encode_fn = lambda x: sim.oracle_summary(x)
        def pivot(self, theta, x): return sim.r_star(theta, x)
    class _Trained:
        def __init__(self, sim): self.procedure = _Proc(sim)

    res = SufficiencyRecovery(n_eval=3000)(_Trained(sim), sim)
    df = res.value
    assert {"sufficiency_min_spearman", "spearman_log_s2", "spearman_xbar"}.issubset(df.columns)
    assert df["sufficiency_min_spearman"].iloc[0] > 0.98
    assert res.passed


def test_sufficiency_recovery_detects_collapse():
    # encode_fn that drops σ²-info (returns only X̄ duplicated) ⇒ low Spearman on log s².
    import torch
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.diagnostics.sufficiency_recovery import SufficiencyRecovery
    sim = NormalUnknownMeanVar()

    class _Proc:
        d_theta = 2
        def __init__(self, sim):
            self.encode_fn = lambda x: x.mean(dim=-1, keepdim=True).repeat(1, 2)  # only X̄
        def pivot(self, theta, x): return sim.r_star(theta, x)
    class _Trained:
        def __init__(self, sim): self.procedure = _Proc(sim)

    res = SufficiencyRecovery(n_eval=3000)(_Trained(sim), sim)
    assert res.value["spearman_log_s2"].iloc[0] < 0.5     # σ²-info lost
    assert not res.passed


def test_sufficiency_recovery_noop_without_encode_fn():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.diagnostics.sufficiency_recovery import SufficiencyRecovery
    sim = NormalUnknownMeanVar()
    class _Proc:
        d_theta = 2; encode_fn = None
        def pivot(self, theta, x): return x
    class _Trained:
        procedure = _Proc()
    res = SufficiencyRecovery(n_eval=10)(_Trained(), sim)
    assert res.passed and "reason" in res.meta
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_sufficiency_recovery.py::test_sufficiency_recovery_perfect_when_encode_is_oracle -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `src/cdsbi/diagnostics/sufficiency_recovery.py`**

```python
"""SufficiencyRecovery: does a trained learned summary retain the sufficient info?

Spearman (rank, monotone-invariant) of each oracle-summary coordinate against the
trained learned features. Monotone-invariant so a curved-but-equivalent feature
(e.g. ∝ s² rather than log s²) still scores ~1. No-ops unless the procedure
exposes encode_fn (a learned summary) and the simulator exposes oracle_summary.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr, ConstantInputWarning

from cdsbi.diagnostics.base import DiagnosticResult

# oracle-coordinate display names for NormalUnknownMeanVar's (log s², X̄)
_ORACLE_NAMES = ["log_s2", "xbar"]


class SufficiencyRecovery:
    name = "sufficiency_recovery"

    def __init__(self, n_eval: int = 4000, pass_threshold: float = 0.9, seed: int = 123):
        self.n_eval = n_eval
        self.pass_threshold = pass_threshold
        self.seed = seed

    def _noop(self, reason: str) -> DiagnosticResult:
        return DiagnosticResult(self.name, value=pd.DataFrame(), passed=True,
                                noise_floor=0.0, n_samples=0, meta={"reason": reason})

    def __call__(self, trained, simulator, eval_data=None, x_per_theta=None) -> DiagnosticResult:
        proc = getattr(trained, "procedure", None)
        encode_fn = getattr(proc, "encode_fn", None)
        if encode_fn is None:
            return self._noop("procedure has no encode_fn (no learned summary)")
        if not hasattr(simulator, "oracle_summary"):
            return self._noop("simulator has no oracle_summary")
        rng = np.random.default_rng(self.seed)
        _, x = simulator.sample(self.n_eval, rng)
        with torch.no_grad():
            feats = encode_fn(x).detach().cpu().numpy()             # (n, d_out)
            oracle = simulator.oracle_summary(x).detach().cpu().numpy()  # (n, d_oracle)
        row = {}
        spearmans = []
        # A collapsed learned feature is constant → spearmanr/pearsonr return NaN.
        # Use np.nanmax (NOT max(), which is order-dependent with NaN) so a real
        # recovery on the OTHER feature index isn't masked, and suppress the warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            for k in range(oracle.shape[1]):
                name = _ORACLE_NAMES[k] if k < len(_ORACLE_NAMES) else f"coord{k}"
                sp = np.nanmax([abs(spearmanr(oracle[:, k], feats[:, j]).statistic)
                                for j in range(feats.shape[1])])
                pe = np.nanmax([abs(pearsonr(oracle[:, k], feats[:, j])[0])
                                for j in range(feats.shape[1])])
                row[f"spearman_{name}"] = float(sp)
                row[f"pearson_{name}"] = float(pe)
                spearmans.append(sp)
        row["sufficiency_min_spearman"] = float(np.nanmin(spearmans))
        df = pd.DataFrame([row])
        passed = bool(row["sufficiency_min_spearman"] > self.pass_threshold)
        return DiagnosticResult(self.name, value=df, passed=passed,
                                noise_floor=0.0, n_samples=self.n_eval,
                                meta={"pass_threshold": self.pass_threshold})
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_sufficiency_recovery.py -v`
Expected: all pass (2 plumbing from Task 1 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/diagnostics/sufficiency_recovery.py tests/unit/test_sufficiency_recovery.py
git commit -m "feat(diag): SufficiencyRecovery — Spearman of learned features vs oracle stat"
```

---

## Task 3: `FloorIntegrity` diagnostic (arm-aware)

**Files:**
- Create: `src/cdsbi/diagnostics/floor_integrity.py`
- Test: `tests/unit/test_floor_integrity.py`

Records whether the trained objective respected its lower bound. For an NF-MLE loss it compares `final_loss` to the simulator's conditional-entropy floor `H` (cheat = sinking below `H − tol`). For any other loss (II-A's energy loss) the floor comparison is N/A → no cheat, pass.

- [ ] **Step 1: Write the failing tests**

```python
"""FloorIntegrity: arm-aware floor check (NF-MLE vs other losses)."""
import math
import pandas as pd


class _Trained:
    def __init__(self, final_loss, loss_class):
        self.final_loss = final_loss
        self.arch_metadata = {"loss_class": loss_class}
        self.procedure = object()


class _SimWithFloor:
    def entropy_lower_bound(self, **kw): return 0.92


def test_floor_integrity_passes_at_floor():
    from cdsbi.diagnostics.floor_integrity import FloorIntegrity
    res = FloorIntegrity()(_Trained(0.90, "NFMLELoss"), _SimWithFloor())
    row = res.value.iloc[0]
    assert not bool(row["cheats"])
    assert res.passed
    assert abs(float(row["entropy_floor"]) - 0.92) < 1e-9


def test_floor_integrity_flags_cheat_below_floor():
    from cdsbi.diagnostics.floor_integrity import FloorIntegrity
    res = FloorIntegrity()(_Trained(-5.29, "NFMLELoss"), _SimWithFloor())
    assert bool(res.value.iloc[0]["cheats"])
    assert not res.passed


def test_floor_integrity_noop_for_non_nfmle_loss():
    from cdsbi.diagnostics.floor_integrity import FloorIntegrity
    res = FloorIntegrity()(_Trained(-100.0, "EnergyCalibrationLoss"), _SimWithFloor())
    row = res.value.iloc[0]
    assert not bool(row["cheats"])          # floor N/A for a non-likelihood loss
    assert res.passed
    assert math.isnan(float(row["entropy_floor"]))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_floor_integrity.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `src/cdsbi/diagnostics/floor_integrity.py`**

```python
"""FloorIntegrity: did the trained objective respect its lower bound, or cheat?

Arm-aware. For an NF-MLE loss the conditional-entropy floor H = E[loss at r*]
applies: cheating = final_loss < H − tol (the §3.5 folding / Stage-B collapse
pathology). For a non-likelihood loss (e.g. the energy calibration loss) there is
no H to undercut, so the floor comparison is N/A (cheats=False, pass).
"""
from __future__ import annotations

import math

import pandas as pd

from cdsbi.diagnostics.base import DiagnosticResult

_NFMLE_LOSSES = {"NFMLELoss"}


class FloorIntegrity:
    name = "floor_integrity"

    def __init__(self, tol: float = 0.10):
        self.tol = tol

    def __call__(self, trained, simulator, eval_data=None, x_per_theta=None) -> DiagnosticResult:
        loss_class = getattr(trained, "arch_metadata", {}).get("loss_class", "")
        final_loss = float(getattr(trained, "final_loss", float("nan")))
        applicable = loss_class in _NFMLE_LOSSES and hasattr(simulator, "entropy_lower_bound")
        if applicable:
            H = float(simulator.entropy_lower_bound())
            margin = final_loss - H
            cheats = bool(margin < -self.tol)
            meta = {"loss_class": loss_class, "tol": self.tol}
        else:
            H = float("nan")
            margin = float("nan")
            cheats = False
            meta = {"reason": f"floor N/A for loss {loss_class!r}"}
        df = pd.DataFrame([{
            "final_loss": final_loss, "entropy_floor": H,
            "floor_margin": margin, "cheats": cheats,
        }])
        return DiagnosticResult(self.name, value=df, passed=(not cheats),
                                noise_floor=0.0, n_samples=0, meta=meta)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/unit/test_floor_integrity.py -v`
Expected: 3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/diagnostics/floor_integrity.py tests/unit/test_floor_integrity.py
git commit -m "feat(diag): FloorIntegrity — arm-aware entropy-floor cheat check"
```

---

## Task 4: wire both diagnostics into the runner + index-row reduction

**Files:**
- Modify: `src/cdsbi/experiments/run.py`

Both diagnostics self-gate (no-op when inapplicable), so adding them to the global list is safe for all experiments. READ the diagnostics-list block (~line 319–365) and the `_write_index_row` reduction blocks (~line 460–470, the `jacobian_recovery` / `marginal_cd_recovery` patterns).

- [ ] **Step 1: Add imports + list entries**

Imports (with the other diagnostics):
```python
from cdsbi.diagnostics.sufficiency_recovery import SufficiencyRecovery
from cdsbi.diagnostics.floor_integrity import FloorIntegrity
```
List entries (alongside `("marginal_cd_recovery", ...)`):
```python
        ("sufficiency_recovery", SufficiencyRecovery(
            n_eval=int(OmegaConf.select(cfg, "experiment.n_eval", default=4000)))),
        ("floor_integrity", FloorIntegrity()),
```
(`SufficiencyRecovery` does its own sampling, so it does NOT need `x_per_theta`; do NOT add it to `x_sharing_names`.)

- [ ] **Step 2: Add index-row reduction**

In `_write_index_row`, after the `marginal_cd_recovery` block:
```python
    sr_path = rd.path / "diagnostics" / "sufficiency_recovery.parquet"
    if sr_path.exists():
        sr_df = pd.read_parquet(sr_path)
        if "sufficiency_min_spearman" in sr_df.columns and len(sr_df):
            row["sufficiency_min_spearman"] = float(sr_df["sufficiency_min_spearman"].iloc[0])
    fi_path = rd.path / "diagnostics" / "floor_integrity.parquet"
    if fi_path.exists():
        fi_df = pd.read_parquet(fi_path)
        if "floor_margin" in fi_df.columns and len(fi_df):
            row["floor_margin"] = float(fi_df["floor_margin"].iloc[0])
            row["floor_cheats"] = bool(fi_df["cheats"].iloc[0])
```

- [ ] **Step 3: Plumbing smoke (tiny budget; checks the diagnostics write + reduce)**

Run:
```bash
rm -rf /tmp/m30_smoke
python -m cdsbi.experiments.run experiment=mu_sigma_stage_b seed=0 \
  budget=small training.n_steps=50 hydra.run.dir=/tmp/m30_smoke 2>&1 | tail -5
python -c "
import pandas as pd
sr = pd.read_parquet('/tmp/m30_smoke/diagnostics/sufficiency_recovery.parquet')
fi = pd.read_parquet('/tmp/m30_smoke/diagnostics/floor_integrity.parquet')
print('sufficiency cols:', sorted(sr.columns))
print('floor cols:', sorted(fi.columns), '| cheats:', bool(fi['cheats'].iloc[0]))
row = pd.read_parquet('/tmp/m30_smoke/index_row.parquet')
print('index_row harness cols:', [c for c in row.columns if c in
      ('sufficiency_min_spearman','floor_margin','floor_cheats')])
"
```
Expected: `sufficiency_recovery.parquet` has `sufficiency_min_spearman`, `spearman_log_s2`, `spearman_xbar`, ...; `floor_integrity.parquet` has `final_loss`, `entropy_floor`, `floor_margin`, `cheats`; index_row has the three harness columns. (At 50 steps the *values* are meaningless — `mu_sigma_stage_b` uses the learned DeepSets summary, so `cheats` may already be True; this step only checks plumbing.)

- [ ] **Step 4: Confirm fast unit suite green + no import breakage**

Run: `pytest tests/unit/test_sufficiency_recovery.py tests/unit/test_floor_integrity.py -q && python -c "import cdsbi.experiments.run"`
Expected: all pass; import clean.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/experiments/run.py
git commit -m "feat(run): wire SufficiencyRecovery + FloorIntegrity into harness + index_row"
```

---

## Task 5: convert the M2 smoke into a cheat-capture regression test

**Files:**
- Modify: `tests/integration/test_mu_sigma_stage_b_smoke.py`

The M2 smoke currently asserts the learned summary *calibrates* (and fails — device-1 cheats). Convert it to assert the **cheat is present** under the naive device-1 DeepSets, documenting the §1 pathology and turning the suite green. Precedent: `tests/ablation/test_trained_folding.py` captures the §3.5 folding failure the same way.

READ the current `tests/integration/test_mu_sigma_stage_b_smoke.py` to reuse its setup (the training config that reproduces the cheat).

- [ ] **Step 1: Replace the test body**

```python
"""Regression capture: the naive device-1 learned summary (DeepSets, log_det=0 +
normalization) CHEATS the NF-MLE loss — final loss sinks below the entropy floor
and σ²-information collapses. This documents the Stage-B pathology that motivates
the device bake-off (see specs/2026-05-29-cd-sbi-stage-b-device-bakeoff-design.md
§1). It asserts the cheat is PRESENT; the fixes are the bake-off arms (M3.1–M3.3).
Precedent: tests/ablation/test_trained_folding.py captures the §3.5 folding failure."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.intensive
def test_device1_learned_summary_cheats_below_floor():
    from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar
    from cdsbi.conditioners.deep_sets import DeepSetsConditioner
    from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
    from cdsbi.losses.nfmle import NFMLELoss
    from cdsbi.methods.cd_sbi import CDSBIRunner
    from scipy.stats import spearmanr

    torch.manual_seed(0)
    sim = NormalUnknownMeanVar()
    flow = SingleIndexMonotoneFlow(d=2, theta_signs=list(sim.theta_signs),
                                   feat_signs=list(sim.feat_signs), hidden=32, depth=2)
    cond = DeepSetsConditioner(n_iid=sim.n_iid, d_out=2, hidden=32, depth=2)
    runner = CDSBIRunner(flow=flow, conditioner=cond, loss=NFMLELoss())
    config = {"lr": 3e-3, "batch_size": 256, "n_steps": 6000, "n_train": 10000,
              "optimizer": "adamw", "lr_schedule": "warmup_cosine",
              "warmup_steps": 300, "lr_min_ratio": 0.01, "fresh_batch": False}
    trained = runner.fit(simulator=sim, config=config, seed=0)

    H = sim.entropy_lower_bound()
    # (1) the cheat: final NF-MLE loss sinks well below the conditional-entropy floor
    assert trained.final_loss < H - 1.0, (
        f"expected device-1 cheat (loss ≪ floor {H:.2f}); got {trained.final_loss:.3f}"
    )

    # (2) the harm: σ²-information collapses (X̄ survives)
    rng = np.random.default_rng(123)
    _, x = sim.sample(4000, rng)
    cond.eval()
    with torch.no_grad():
        feats = cond.encode(x.to(runner.device))[0].cpu().numpy()
    oracle = sim.oracle_summary(x).numpy()
    sp_logs2 = max(abs(spearmanr(oracle[:, 0], feats[:, j]).statistic) for j in range(2))
    sp_xbar = max(abs(spearmanr(oracle[:, 1], feats[:, j]).statistic) for j in range(2))
    print(f"device-1 cheat: final_loss={trained.final_loss:.3f} floor={H:.3f} "
          f"| σ²-Spearman={sp_logs2:.3f} X̄-Spearman={sp_xbar:.3f}")
    assert sp_logs2 < 0.9, f"expected σ²-collapse; got Spearman {sp_logs2:.2f}"
    assert sp_xbar > 0.9, f"expected X̄ to survive; got Spearman {sp_xbar:.2f}"
```

- [ ] **Step 2: Run it (intensive; trains one model, ~minutes on GPU)**

Run: `pytest tests/integration/test_mu_sigma_stage_b_smoke.py -v -s -m intensive`
Expected: PASS — it asserts the cheat (final_loss ≪ floor) and the σ²-collapse (Spearman < 0.9) are present, matching the M2 observation (loss ≈ −5.3, σ²-Spearman ≈ 0.47). If the cheat does NOT reproduce (loss ≥ floor − 1, or σ²-Spearman ≥ 0.9), STOP and report — that would mean the device-1 pathology is seed/recipe-sensitive and the audit needs revisiting.

- [ ] **Step 3: Confirm fast suite green**

Run: `pytest -q`
Expected: all fast tests pass; this intensive test deselected.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_mu_sigma_stage_b_smoke.py
git commit -m "test(regression): capture the device-1 learned-summary cheat (Stage-B motivation)"
```

---

## Self-review

- **Spec coverage (§2):** `SufficiencyRecovery` (Task 2) ✓; arm-aware `FloorIntegrity` (Task 3) ✓; expose trained features to diagnostics (Task 1: `encode_fn` + `oracle_summary`) ✓; wired into runner/index-row (Task 4) ✓; M2 smoke → cheat-capture regression test (Task 5) ✓. No arm code (correct — arms are M3.1–M3.3).
- **Placeholder scan:** every step has runnable code + exact commands + expected output; no TBD.
- **Type consistency:** `encode_fn: Callable | None` defined on `PivotBasedProcedure` (Task 1) and read by `SufficiencyRecovery` (Task 2) + the Task-5 test; `oracle_summary(x) -> (n,2)` defined (Task 1), read by `SufficiencyRecovery` and the Task-5 test; `DiagnosticResult` fields match the protocol; index-row columns (`sufficiency_min_spearman`, `floor_margin`, `floor_cheats`) written in Task 4 match the parquet columns produced in Tasks 2–3. The `arch_metadata["loss_class"]` key read by `FloorIntegrity` is the one `cd_sbi.py` already writes (verified in M2).
- **Risk — `entropy_lower_bound` cost in `FloorIntegrity`.** It runs a 50k-MC estimate per diagnostic invocation (~seconds). Acceptable for the one-per-run diagnostic pass; if it ever dominates, cache on the simulator. Not addressed here (premature).
- **Risk — `SufficiencyRecovery` oracle-coordinate names.** `_ORACLE_NAMES=["log_s2","xbar"]` is specific to `NormalUnknownMeanVar`'s 2-D oracle; for a future target with a different oracle the names fall back to `coord{k}`. Fine for this milestone (single target).
- **Known follow-on:** the bake-off arms (I-B/II-A/I-A) consume this harness in M3.1–M3.3; the cross-arm verdict table is M3.4.
