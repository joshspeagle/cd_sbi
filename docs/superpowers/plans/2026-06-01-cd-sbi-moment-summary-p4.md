# CD-SBI Moment-Summary — Phase 4 (uniform coverage + auto-selection) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the μ₂-style *pointwise* (sup-θ) coverage gap with an S2b
least-favorable reweighting of the Stage-2 pivot calibration, and make the suite
self-driving by auto-selecting the target/R1 knobs with the P2 modality detector.

**Architecture:** Adds a least-favorable **reweighting** to `TwoStageCDSBIRunner`'s
Stage 2 (the summary is *frozen*, so reweighting by the floor-free per-θ coverage
discrepancy `D(θ)` is **safe** — Thm 7's collapse needs a *learned* summary). The
reweighting is exponentiated-gradient on a θ-grid (Thms 8–9). Then wires P2's
`select_target_and_r1` into `run.py` for auto-configuration. **S2b's tractable
estimator is theory-OPEN (note §9½/§13.4)** — this phase *builds and measures* it;
a null result (no sup-θ improvement) is an informative finding, not a failure.

**Tech stack:** Python, PyTorch, Hydra, pytest. Depends on P1 (runner) and P2
(modality detector). P3-independent.

**Spec:** `docs/superpowers/specs/2026-05-31-cd-sbi-moment-summary-design.md` (Phase 4).
**Theory:** note §6½ (Thm S2), §9½ (Thms 8–9 closed form), §13.4 (Prop 16).

---

## File structure (Phase 4)

- Create `src/cdsbi/reweighting/least_favorable.py` — EG weight updater over a θ-grid.
- Modify `src/cdsbi/methods/cd_sbi_two_stage.py` — optional reweighted Stage 2.
- Modify `src/cdsbi/experiments/run.py` — optional auto-select of target/R1 (P2 detector).
- Tests: `tests/unit/test_least_favorable.py`,
  `tests/integration/test_reweighted_uniform_coverage.py`,
  `tests/theory/test_coverage_decomposition.py` (un-skip `test_s2_minimax_trained`).

---

### Task 1: Least-favorable EG weight updater

**Files:**
- Create: `src/cdsbi/reweighting/least_favorable.py`
- Test: `tests/unit/test_least_favorable.py`

A pure, testable component: given per-cell discrepancies `D` over a θ-grid, maintain
exponentiated-gradient weights `w ∝ w·exp(γ D)` (→ concentrate on the highest-`D`
cells, Thm 9's `π_LF` on the regret-extremes), and assign θ-samples to cells.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_least_favorable.py
import numpy as np
import torch
from cdsbi.reweighting.least_favorable import GridReweighter


def test_grid_assignment_and_uniform_init():
    rw = GridReweighter(lows=[-1.0], highs=[1.0], n_per_dim=4)
    assert rw.n_cells == 4
    assert np.allclose(rw.weights, np.full(4, 0.25))
    cells = rw.assign(torch.tensor([[-0.9], [0.9], [0.0]]))   # left, right, middle
    assert cells[0] == 0 and cells[1] == 3


def test_eg_update_concentrates_on_high_discrepancy_cells():
    rw = GridReweighter(lows=[-1.0], highs=[1.0], n_per_dim=4, gamma=5.0)
    D = np.array([0.0, 0.0, 0.0, 0.3])      # cell 3 is the worst
    for _ in range(10):
        rw.update(D)
    assert rw.weights.argmax() == 3 and rw.weights[3] > 0.7


def test_importance_weights_unit_mean_and_nonuniform():
    rw = GridReweighter(lows=[-1.0], highs=[1.0], n_per_dim=4, gamma=5.0)
    rw.update(np.array([0.0, 0.0, 0.0, 0.5]))         # cell 3 upweighted
    th = (torch.arange(2000).float() / 1000.0 - 1.0).unsqueeze(1)  # spans [-1,1]
    iw = rw.importance_weights(th)
    assert abs(float(iw.mean()) - 1.0) < 0.05        # mean-1 (unbiased loss scale)
    assert float(iw.std()) > 0.1                     # genuinely NON-uniform (the point)
    assert float(iw[th.squeeze() > 0.5].mean()) > float(iw[th.squeeze() < -0.5].mean())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_least_favorable.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cdsbi/reweighting/least_favorable.py
"""GridReweighter: exponentiated-gradient least-favorable weights over a θ-grid
(Thms 8–9, §13.4). Concentrates training on the highest per-θ discrepancy cells →
π_LF on the regret-extremes. Used to reweight Stage-2 pivot calibration toward
uniform (sup-θ) coverage. Safe with a FROZEN summary (Thm 7 collapse needs a
learned summary)."""
from __future__ import annotations

import itertools
from typing import List, Sequence

import numpy as np
import torch


class GridReweighter:
    def __init__(self, lows: Sequence[float], highs: Sequence[float],
                 n_per_dim: int = 5, gamma: float = 5.0):
        self.lows = np.asarray(lows, dtype=float)
        self.highs = np.asarray(highs, dtype=float)
        self.d = len(lows)
        self.n_per_dim = n_per_dim
        self.gamma = gamma
        self.edges = [np.linspace(lows[i], highs[i], n_per_dim + 1) for i in range(self.d)]
        self.centers = np.array(list(itertools.product(
            *[0.5 * (e[:-1] + e[1:]) for e in self.edges])))      # (n_cells, d)
        self.n_cells = self.centers.shape[0]
        self.weights = np.full(self.n_cells, 1.0 / self.n_cells)

    def assign(self, theta: torch.Tensor) -> np.ndarray:
        """Map each θ (n,d) to a flat cell index."""
        th = theta.detach().cpu().numpy()
        idx = np.zeros(th.shape[0], dtype=int)
        for i in range(self.d):
            b = np.clip(np.digitize(th[:, i], self.edges[i]) - 1, 0, self.n_per_dim - 1)
            idx = idx * self.n_per_dim + b
        return idx

    def update(self, discrepancy: np.ndarray) -> None:
        """EG step: w ← w·exp(γ D), renormalized."""
        w = self.weights * np.exp(self.gamma * np.asarray(discrepancy))
        self.weights = w / w.sum()

    def importance_weights(self, theta: torch.Tensor) -> torch.Tensor:
        """Per-sample weights ∝ (cell weight)/(uniform), normalized to mean 1."""
        cells = self.assign(theta)
        raw = self.weights[cells] * self.n_cells          # /uniform
        raw = raw / raw.mean()
        return torch.tensor(raw, dtype=torch.float32, device=theta.device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_least_favorable.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/reweighting/least_favorable.py tests/unit/test_least_favorable.py
git commit -m "feat(reweighting): GridReweighter (least-favorable EG weights, Thms 8-9)"
```

---

### Task 2: Reweighted Stage 2 in `TwoStageCDSBIRunner`

**Files:**
- Modify: `src/cdsbi/methods/cd_sbi_two_stage.py` (`_fit_pivot`)
- Test: covered by Task 3.

Add an optional reweighting path to Stage 2: importance-weight the NF-MLE loss by the
reweighter, and every `reweight_every` steps re-estimate per-cell coverage
discrepancy `D` (sample X at cell centers, push through the pivot) and EG-update.

- [ ] **Step 0: Add a per-dim `theta_upper` to `NormalUnknownMeanVar`** (it has
`theta_lower` but not the matching upper bound the reweighter needs at d=2):

```python
# in src/cdsbi/simulators/normal_unknown_mean_var.py, next to theta_lower:
@property
def theta_upper(self) -> Tuple[float, float]:
    """Per-coordinate prior upper bounds (log σ_max, μ_max) — pairs with theta_lower."""
    return (self.log_sigma_range[1], self.mu_range[1])
```
(`SignNormal1D` is d=1, so its scalar `theta_range` is already the per-coord box —
the `getattr(..., None) or` fallback handles it; no change needed there.)

- [ ] **Step 1: Add the discrepancy estimator + reweighted loop**

In `cd_sbi_two_stage.py`, replace `_fit_pivot` with a version that branches on
`config.get("reweight", False)`:

```python
    def _fit_pivot(self, simulator, config, rngs, dev, n_steps):
        import numpy as np
        from scipy.stats import chi2
        self.flow.train()
        opt = torch.optim.Adam(self.flow.parameters(), lr=float(config["lr"]))
        grad_clip = float(config.get("grad_clip_norm", 5.0))
        d = int(simulator.d_theta)
        reweight = bool(config.get("reweight", False))
        rw = None
        if reweight:
            from cdsbi.reweighting.least_favorable import GridReweighter
            # PER-DIM prior box — NOT simulator.theta_range (that is the SCALAR bounding
            # box; for μσ it spans σ∈[0.007,148], ~76% outside the prior). Use the
            # simulator's per-coord bounds; fall back to the scalar box for d=1.
            lows = list(getattr(simulator, "theta_lower", None) or (simulator.theta_range[0],) * d)
            highs = list(getattr(simulator, "theta_upper", None) or (simulator.theta_range[1],) * d)
            rw = GridReweighter(lows=lows, highs=highs,
                                n_per_dim=int(config.get("reweight_grid", 5)),
                                gamma=float(config.get("reweight_gamma", 5.0)))
        every = int(config.get("reweight_every", 200))
        losses = []; import time; t0 = time.time()
        loss_fn = self.loss
        for step in range(n_steps):
            theta, x = simulator.sample(int(config["batch_size"]), rngs.train)
            theta, x = theta.to(dev), x.to(dev)
            with torch.no_grad():
                feats, _ = self.conditioner.encode(x)
            r, log_det = self.flow.forward(theta, context=feats)
            # per-sample NF-MLE (matches NFMLELoss: ½‖r‖²+½d·log2π−log|∂r/∂feat|)
            per = (0.5 * r.pow(2).sum(-1) + 0.5 * d * np.log(2 * np.pi) - log_det)
            if reweight:
                iw = rw.importance_weights(theta).to(dev)
                loss_val = (iw * per).mean()
            else:
                loss_val = per.mean()
            opt.zero_grad(); loss_val.backward()
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=grad_clip)
            opt.step()
            losses.append(float(loss_val.item()))
            if reweight and step > 0 and step % every == 0:
                D = self._cell_discrepancy(simulator, rw, dev,
                                           n=int(config.get("reweight_eval_n", 2000)),
                                           rng=rngs.train)
                rw.update(D)
        return losses, time.time() - t0

    def _cell_discrepancy(self, simulator, rw, dev, n, rng):
        import numpy as np
        from scipy.stats import chi2
        d = int(simulator.d_theta)
        self.flow.eval()
        D = np.zeros(rw.n_cells)
        with torch.no_grad():
            for c in range(rw.n_cells):
                theta0 = rw.centers[c]
                x = simulator.sample_x_given_theta(theta0, n, rng).to(dev)
                feats, _ = self.conditioner.encode(x)
                th = torch.tensor(theta0, dtype=torch.float32, device=dev).expand(n, d)
                r, _ = self.flow.forward(th, context=feats)
                sq = (r ** 2).sum(-1).cpu().numpy()
                worst = 0.0
                for a in (0.5, 0.8, 0.9, 0.95):
                    worst = max(worst, abs(float((sq <= chi2.ppf(a, df=d)).mean()) - a))
                D[c] = worst
        self.flow.train()
        return D
```

- [ ] **Step 2: Run the existing P1 integration tests to confirm no regression**

Run: `pytest tests/integration/test_two_stage_runner.py -v`
Expected: still pass (reweight defaults False → identical to P1's loop).

- [ ] **Step 3: Commit**

```bash
git add src/cdsbi/methods/cd_sbi_two_stage.py
git commit -m "feat(method): optional least-favorable reweighting in Stage-2 calibration"
```

---

### Task 3: Integration — reweighting reduces sup-θ coverage error (crippled capacity)

**Files:**
- Create: `tests/integration/test_reweighted_uniform_coverage.py`

To exhibit a sup-θ gap to close, cripple the pivot's σ-context capacity (so ρ-average
calibration leaves a tail). Reweighting should reduce the worst-θ coverage error.

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_reweighted_uniform_coverage.py
import math
import numpy as np
import torch
import pytest
from scipy.stats import chi2
from cdsbi.conditioners.sufficient_stat import SufficientStatConditioner
from cdsbi.flows.single_index_monotone import SingleIndexMonotoneFlow
from cdsbi.methods.cd_sbi_two_stage import TwoStageCDSBIRunner
from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar


def _sup_cov_err(proc, sim, seed):
    rng = np.random.default_rng(seed); worst = 0.0
    for ls in (math.log(0.3), 0.0, math.log(3.0)):       # span the σ box incl. tails
        x = sim.sample_x_given_theta(np.array([ls, 0.0]), 8000, rng)
        th = torch.tensor([ls, 0.0], dtype=torch.float32).expand(8000, 2)
        sq = (proc.pivot_fn(th, x) ** 2).sum(-1).numpy()
        for a in (0.8, 0.9, 0.95):
            worst = max(worst, abs(float((sq <= chi2.ppf(a, 2)).mean()) - a))
    return worst


@pytest.mark.intensive
def test_reweighting_reduces_sup_theta_error():
    sim = NormalUnknownMeanVar()
    base = {"lr": 2e-3, "stage1_steps": 0, "stage2_steps": 4000,
            "batch_size": 512, "fresh_batch": True}

    def run(reweight):
        torch.manual_seed(0)
        # crippled σ-context: hidden=4 so ρ-average leaves a tail gap
        flow = SingleIndexMonotoneFlow(d=2, theta_signs=sim.theta_signs,
                                       feat_signs=sim.feat_signs, hidden=4)
        cond = SufficientStatConditioner(n_iid=sim.n_iid)   # oracle summary (isolate the pivot)
        runner = TwoStageCDSBIRunner(flow=flow, conditioner=cond, device="cpu")
        cfg = dict(base); cfg["reweight"] = reweight
        return runner.fit(sim, cfg, seed=0).procedure

    sup_avg = _sup_cov_err(run(False), sim, seed=1)
    sup_rw = _sup_cov_err(run(True), sim, seed=1)
    print(f"sup-θ coverage err: ρ-average={sup_avg:.3f}  reweighted={sup_rw:.3f}")
    # S2b's estimator is theory-open: assert improvement, not a guarantee. If it does
    # NOT improve, that is an informative null result — record it (do not force-pass).
    assert sup_rw <= sup_avg + 0.005, "reweighting should not WORSEN sup-θ coverage"
    assert sup_rw < sup_avg, "expected reweighting to reduce the worst-θ coverage error"
```

- [ ] **Step 2: Run it (opt-in)** — `pytest tests/integration/test_reweighted_uniform_coverage.py -m intensive -v -s`
Expected: prints both; reweighted ≤ ρ-average. If the second assertion fails (no
improvement) but the first holds (no harm), that is the documented **null result**
for the open S2b estimator — weaken to the no-harm assertion and record it.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_reweighted_uniform_coverage.py
git commit -m "test(integration): least-favorable reweighting reduces sup-θ coverage error"
```

---

### Task 4: Fill the scaffolded `test_s2_minimax_trained`

**Files:**
- Modify: `tests/theory/test_coverage_decomposition.py` (replace the skip stub)

The theory note left `test_s2_minimax_trained` as a scaffold (the trained analog of
the closed-form Thm 8 witness). Implement it as a thin wrapper over Task 3's result.

- [ ] **Step 1: Replace the skip stub with the real test**

In `tests/theory/test_coverage_decomposition.py`, replace the
`@pytest.mark.skip(reason="SCAFFOLD (S2 trained)...")` stub with:

```python
@pytest.mark.intensive
def test_s2_minimax_trained():
    """Trained analog of Thm 8 (closed-form minimax beats ρ-average on sup-θ): a
    crippled-capacity pivot under ρ-average leaves a tail gap that least-favorable
    reweighting reduces. Delegates to the P4 integration check."""
    from tests.integration.test_reweighted_uniform_coverage import (
        test_reweighting_reduces_sup_theta_error as _check)
    _check()
```

- [ ] **Step 2: Run it (opt-in)** — `pytest tests/theory/test_coverage_decomposition.py::test_s2_minimax_trained -m intensive -v`
Expected: PASS (delegates to Task 3). If Task 3 recorded a null result, mirror that
framing here (no-harm assertion + recorded note).

- [ ] **Step 3: Confirm fast theory suite still green** — `pytest tests/theory/ -q -m "not intensive"`
Expected: same count as before minus the now-unskipped intensive test.

- [ ] **Step 4: Commit**

```bash
git add tests/theory/test_coverage_decomposition.py
git commit -m "test(theory): fill test_s2_minimax_trained (trained minimax-vs-average)"
```

---

### Task 5: Auto-select target/R1 via the modality detector (self-driving suite)

**Files:**
- Modify: `src/cdsbi/experiments/run.py` (optional auto-select before building the method)
- Test: `tests/unit/test_auto_select_wiring.py`

When `method.auto_select: true`, call P2's `select_target_and_r1(simulator)` and set
`conditioner.target` + the flow (`single_index_monotone` if R1 else
`non_monotone_pivot`) + `allow_ablation` accordingly — so one experiment config drives
the right knobs per target.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_auto_select_wiring.py
from cdsbi.methods.modality_detector import select_target_and_r1
from cdsbi.simulators.sign_normal_1d import SignNormal1D
from cdsbi.simulators.normal_unknown_mean_var import NormalUnknownMeanVar


def test_auto_select_maps_to_flow_and_target():
    # the helper run.py uses: map a detector decision to (flow, target, allow_ablation)
    from cdsbi.experiments.run import _auto_select_knobs
    sign = _auto_select_knobs(SignNormal1D())
    assert sign == {"flow": "non_monotone_pivot", "target": "theta_sq", "allow_ablation": True}
    reg = _auto_select_knobs(NormalUnknownMeanVar())
    assert reg == {"flow": "single_index_monotone", "target": "theta", "allow_ablation": False}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_auto_select_wiring.py -v`
Expected: FAIL with `ImportError: cannot import name '_auto_select_knobs'`.

- [ ] **Step 3: Add `_auto_select_knobs` + wire it into `_build_method`**

```python
# add to src/cdsbi/experiments/run.py
def _auto_select_knobs(simulator) -> dict:
    from cdsbi.methods.modality_detector import select_target_and_r1
    sel = select_target_and_r1(simulator)
    return {
        "flow": "single_index_monotone" if sel["r1"] else "non_monotone_pivot",
        "target": sel["target"],
        "allow_ablation": not sel["r1"],
    }
```
Then wire it into `_build_method`'s `name == "cd_sbi"` branch. Three correctness
requirements (per review):
- **declare the key:** add `auto_select: false` to `cd_sbi_moment*.yaml` so the cfg
  is struct-valid;
- **order:** apply the overrides **before** the `flow = _build_flow(cfg, simulator)`
  call (else `_build_flow` reads the stale `method.flow`);
- **struct mutation:** use `open_dict` to set the (existing) keys.

```python
from omegaconf import open_dict
# at the TOP of the `if m.name == "cd_sbi":` branch, before _build_flow:
if bool(OmegaConf.select(cfg, "method.auto_select", default=False)):
    knobs = _auto_select_knobs(simulator)
    with open_dict(cfg):
        cfg.method.flow = knobs["flow"]
        cfg.method.allow_ablation = knobs["allow_ablation"]
        if OmegaConf.select(cfg, "conditioner.target", default=None) is not None:
            cfg.conditioner.target = knobs["target"]   # only if the conditioner has it
```

- [ ] **Step 4: Add an integration test of the full wiring (not just the pure fn)**

```python
# append to tests/unit/test_auto_select_wiring.py
import subprocess, sys, torch
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]

def test_auto_select_builds_non_monotone_flow_for_sign(tmp_path):
    out = tmp_path / "auto"
    cmd = [sys.executable, "-m", "cdsbi.experiments.run",
           f"hydra.run.dir={out}", "experiment=moment_sign", "seed=0",
           "method.auto_select=true", "training.stage1_steps=100", "training.stage2_steps=100"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr[-2000:]
    meta = torch.load(out / "model.pt", weights_only=False)["arch_metadata"]
    assert meta["flow_class"] == "NonMonotonePivotFlow"   # detector picked R1-off
```
(Requires P2's `moment_sign` experiment + `non_monotone_pivot` flow. If P2 is not
yet landed, mark this test `@pytest.mark.skip(reason="needs P2")` and rely on the
pure-function test until P2 merges.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_auto_select_wiring.py -v`
Expected: 2 passed (or 1 passed + 1 skipped if P2 not yet landed).

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/experiments/run.py tests/unit/test_auto_select_wiring.py \
        configs/method/cd_sbi_moment.yaml
git commit -m "feat(experiments): auto-select target/R1 from the modality detector"
```

---

## Self-review (checklist)

- **Spec coverage (Phase 4):** S2b least-favorable reweighting (Tasks 1–2),
  sup-θ improvement validation (Task 3), the scaffolded `test_s2_minimax_trained`
  filled (Task 4), modality auto-selection (Task 5). ✓
- **Theory-open honesty:** Task 3/4 assert *improvement or no-harm*, with the null
  result explicitly recordable — matches note §9½/§13.4 (the estimator is open). ✓
- **Frozen-summary safety:** reweighting uses the floor-free `D(θ)` signal, which is
  safe because the summary is frozen in Stage 2 (Thm 7 collapse needs a learned
  summary). Stated in Task 1/2 docstrings. ✓
- **Backward-compatible:** `reweight`/`auto_select` default off ⟹ P1/P2/P3 unchanged.
