# CD-SBI Visualizations — F3 Conceptual Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 6 conceptual figures (C1–C6) that illustrate the theory in Parts I–IV + §10, and insert them into the manuscript — completing the 16-figure visualization suite.

**Architecture:** All 6 are **synthetic-driver** figures: each `render(spec) -> Figure` builds its illustration from closed-form pivots / analytic constructions with `numpy`+`matplotlib` (and the F1 `noise_floor_band` / `position_table_as_axes` panels where they fit). None reads run-dir data (`source_runs: []`). This deliberately resolves the spec's C1/C3 "needs trained weights" gap: conceptual figures are clearer and more controlled built from the closed-form pivot (`r* = θ − X`) and an analytic folded `r` than from a noisy trained checkpoint — and `model.pt` has no `state_dict` anyway.

**Tech Stack:** matplotlib 3.10.5 (F0 `style` + F1 `panels`), numpy, scipy.stats (normal pdf), pytest, pdflatex. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-28-cd-sbi-visualizations-design.md` (catalogue C1–C6). **Prereqs:** F0+F1+F2 merged.

---

## Conventions (read first)

- **Builder contract** (same as F2): `def render(spec: FigureSpec) -> matplotlib.figure.Figure` — calls `style.apply_style()` first, builds the Figure, returns it, NEVER calls `savefig`. All 6 are synthetic, so they ignore `spec.source_runs` (which is `[]`).
- **Determinism:** synthetic figures that sample use `np.random.default_rng(0)` so renders are byte-stable (the render CLI also strips timestamp metadata).
- **Visual acceptance (mandatory, controller):** F3 figures are pure pedagogy — the *message* is the whole point. After each renders, the controller `Read`s the PNG and confirms the message lands AND mathtext/labels are legible. Each builder commit records a one-line `visual:` verdict.
- **Smoke tests** all live in `tests/integration/test_c_figures_smoke.py` (one per builder), using the same `_spec(**kw)` helper pattern as the E-figures.

---

## File structure

```
NEW
  src/cdsbi/analysis/figures/figures/c1_pivot_picture.py
  src/cdsbi/analysis/figures/figures/c2_r1_failure.py
  src/cdsbi/analysis/figures/figures/c3_r2_folding.py
  src/cdsbi/analysis/figures/figures/c4_entropy_floor.py
  src/cdsbi/analysis/figures/figures/c5_kr_structure.py
  src/cdsbi/analysis/figures/figures/c6_sbi_position.py
  tests/integration/test_c_figures_smoke.py

MODIFY
  configs/figures/manifest.yaml      # add c1..c6 entries
  cd_sbi_v7.tex                       # insert 6 \includegraphics (§2.2, §2.3, §2.4, §3.2, §6.1, §10)
```

No changes to F0/F1/F2 code. F3 only adds conceptual builders + manifest entries + manuscript figures.

---

## Task C1: Pivot picture (§2.2)

**Files:** Create `src/cdsbi/analysis/figures/figures/c1_pivot_picture.py`; modify `configs/figures/manifest.yaml`, create `tests/integration/test_c_figures_smoke.py`.

Two panels: `X | θ₀` histograms (shifted) → `r*(θ₀; X) = θ₀ − X` histograms all collapsing onto `N(0,1)`.

- [ ] **Step 1: Create the smoke-test file + C1 test**

`tests/integration/test_c_figures_smoke.py`:

```python
"""Smoke tests for the C1–C6 conceptual figure builders (synthetic drivers)."""
from __future__ import annotations

import matplotlib

from cdsbi.analysis.figures.manifest import FigureSpec


def _spec(**kw):
    base = dict(id="x", description="d", section="theory", builder="m:render",
                output_pdf="figures/x.pdf", output_png="figures/x.png",
                source_runs=[], checkpoint_runs=[])
    base.update(kw)
    return FigureSpec(**base)


def test_c1_two_panels_with_normal_overlay():
    from cdsbi.analysis.figures.figures.c1_pivot_picture import render
    fig = render(_spec())
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2
    # right panel overlays the N(0,1) pdf as a line on top of the histograms
    assert len(fig.axes[1].lines) >= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/integration/test_c_figures_smoke.py::test_c1_two_panels_with_normal_overlay -v`
Expected: FAIL — `ModuleNotFoundError: ...c1_pivot_picture`.

- [ ] **Step 3: Write the builder**

`src/cdsbi/analysis/figures/figures/c1_pivot_picture.py`:

```python
"""C1 — the pivot picture: every X | θ₀ collapses to N(0,1) under r* = θ₀ − X."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures.manifest import FigureSpec

THETAS = [-2.0, 0.0, 2.0]
N = 4000


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, (ax_x, ax_r) = plt.subplots(1, 2, figsize=style.SIZES["double_column"])
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(THETAS)))
    for t, c in zip(THETAS, colors):
        x = rng.normal(t, 1.0, N)
        ax_x.hist(x, bins=40, density=True, alpha=0.5, color=c,
                  label=fr"$\theta_0={t:g}$")
        ax_r.hist(t - x, bins=40, density=True, alpha=0.5, color=c)
    ax_x.set_title(r"Data $X \mid \theta_0$")
    ax_x.set_xlabel("$X$"); ax_x.set_ylabel("density"); ax_x.legend(fontsize=7)
    grid = np.linspace(-4, 4, 200)
    ax_r.plot(grid, norm.pdf(grid), color="k", lw=1.5, label=r"$\mathcal{N}(0,1)$")
    ax_r.set_title(r"Pivot $r^*(\theta_0; X) = \theta_0 - X$")
    ax_r.set_xlabel("$r$"); ax_r.set_ylabel("density"); ax_r.legend(fontsize=7)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/integration/test_c_figures_smoke.py::test_c1_two_panels_with_normal_overlay -v`
Expected: PASS

- [ ] **Step 5: Add the manifest entry** — append to `configs/figures/manifest.yaml`:

```yaml
c1_pivot_picture:
  description: "§2.2 the pivot collapses X|θ₀ onto N(0,1) via r*=θ₀-X"
  section: "2.2"
  source_runs: []
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures.c1_pivot_picture:render"
  output:
    pdf: figures/c1_pivot_picture.pdf
    png: figures/c1_pivot_picture.png
```

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/analysis/figures/figures/c1_pivot_picture.py configs/figures/manifest.yaml tests/integration/test_c_figures_smoke.py
git commit -m "feat(figures): C1 pivot picture (conceptual, closed-form r*)"
```

(Render + visual acceptance happen in Task C7.)

---

## Task C2: (R1) failure schematic (§2.3)

**Files:** Create `c2_r1_failure.py`; modify manifest + smoke test.

One panel: a non-monotone-in-θ pivot `r(θ)` where a level `r = c` is hit at two θ values → multi-valued CD inverse.

- [ ] **Step 1: Append the smoke test**

```python
def test_c2_single_panel_marks_two_roots():
    from cdsbi.analysis.figures.figures.c2_r1_failure import render
    fig = render(_spec())
    assert len(fig.axes) == 1
    # r-curve + horizontal level line + the two-root marker series
    assert len(fig.axes[0].lines) >= 3
```

- [ ] **Step 2: Run to verify it fails.** `pytest tests/integration/test_c_figures_smoke.py::test_c2_single_panel_marks_two_roots -v` → FAIL.

- [ ] **Step 3: Write the builder**

`src/cdsbi/analysis/figures/figures/c2_r1_failure.py`:

```python
"""C2 — (R1) failure: a non-monotone pivot in θ gives a multi-valued CD inverse."""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures.manifest import FigureSpec


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    theta = np.linspace(-3.0, 3.0, 400)
    r = (theta - 0.7) ** 2 - 2.0           # non-monotone in θ
    level = 0.5
    roots = 0.7 + np.array([-1.0, 1.0]) * np.sqrt(level + 2.0)

    fig, ax = plt.subplots(figsize=style.SIZES["single_column"])
    ax.plot(theta, r, color=style.METHOD_STYLE["cd_sbi"]["color"],
            label=r"$r(\theta; X)$ (non-monotone)")
    ax.axhline(level, ls="--", color="0.5", label=r"$r = c$")
    ax.plot(roots, [level, level], "o", color="#e8743b", zorder=5,
            label="two solutions")
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$r(\theta; X)$")
    ax.set_title(r"(R1) violated: two $\theta$ map to one $r$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Manifest entry:**

```yaml
c2_r1_failure:
  description: "§2.3 (R1) failure: non-monotone r ⇒ multi-valued CD inverse"
  section: "2.3"
  source_runs: []
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures.c2_r1_failure:render"
  output:
    pdf: figures/c2_r1_failure.pdf
    png: figures/c2_r1_failure.png
```

- [ ] **Step 6: Commit** — `git add` the builder + manifest + test; `git commit -m "feat(figures): C2 (R1) failure schematic"`.

---

## Task C3: (R2) folding mechanism (§2.4)

**Files:** Create `c3_r2_folding.py`; modify manifest + smoke test.

One panel: a folded-in-X pivot `r(X)` with a region of negative slope (`∂r/∂X < 0`) shaded — the folding that makes `Z(θ) > 1`.

- [ ] **Step 1: Append the smoke test**

```python
def test_c3_single_panel_with_folded_region():
    from cdsbi.analysis.figures.figures.c3_r2_folding import render
    fig = render(_spec())
    assert len(fig.axes) == 1
    # the r-curve (line) + the shaded folded region (a fill_between collection)
    assert len(fig.axes[0].collections) >= 1
```

- [ ] **Step 2: Run to verify it fails.** → FAIL.

- [ ] **Step 3: Write the builder**

`src/cdsbi/analysis/figures/figures/c3_r2_folding.py`:

```python
"""C3 — (R2) failure: a pivot folded in X (∂r/∂X < 0 region) ⇒ Z(θ) > 1."""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures.manifest import FigureSpec


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    x = np.linspace(-3.0, 3.0, 400)
    r = x - 1.6 * np.sin(x)                 # ∂r/∂X = 1 - 1.6 cos(X) < 0 near X=0
    folded = (1.0 - 1.6 * np.cos(x)) < 0.0

    fig, ax = plt.subplots(figsize=style.SIZES["single_column"])
    ax.plot(x, r, color=style.METHOD_STYLE["cd_sbi"]["color"], label=r"$r(\theta; X)$")
    ax.fill_between(x, r.min(), r.max(), where=folded, color="#e8743b", alpha=0.2,
                    label=r"$\partial r/\partial X < 0$ (folded)")
    ax.set_xlabel("$X$"); ax.set_ylabel(r"$r(\theta; X)$")
    ax.set_title(r"(R2) violated: folding $\Rightarrow Z(\theta) > 1$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Manifest entry:**

```yaml
c3_r2_folding:
  description: "§2.4 (R2) failure: folding in X ⇒ Z(θ)>1 (unnormalized density)"
  section: "2.4"
  source_runs: []
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures.c3_r2_folding:render"
  output:
    pdf: figures/c3_r2_folding.pdf
    png: figures/c3_r2_folding.png
```

- [ ] **Step 6: Commit** — `git commit -m "feat(figures): C3 (R2) folding mechanism"`.

---

## Task C4: NF-MLE entropy floor (§3.2)

**Files:** Create `c4_entropy_floor.py`; modify manifest + smoke test.

One panel: NF-MLE loss vs a density-misfit parameter δ, a convex curve sitting on the conditional-entropy floor with equality at the truth (δ=0). Uses `panels.noise_floor_band` for the floor line.

- [ ] **Step 1: Append the smoke test**

```python
def test_c4_loss_curve_with_floor():
    from cdsbi.analysis.figures.figures.c4_entropy_floor import render
    fig = render(_spec())
    ax = fig.axes[0]
    # loss curve + floor line (noise_floor_band axhline) + truth marker
    assert len(ax.lines) >= 3
    # every loss value sits at/above the floor (convex bowl with min at the floor)
    line = ax.lines[0]
    assert float(min(line.get_ydata())) >= 0.99   # floor is 1.0; min ≈ floor
```

- [ ] **Step 2: Run to verify it fails.** → FAIL.

- [ ] **Step 3: Write the builder**

`src/cdsbi/analysis/figures/figures/c4_entropy_floor.py`:

```python
"""C4 — NF-MLE strict propriety: loss ≥ conditional entropy, equality at truth."""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures import style, panels
from cdsbi.analysis.figures.manifest import FigureSpec

FLOOR = 1.0   # illustrative conditional-entropy floor E_ρ[H(X|θ)]


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    delta = np.linspace(-2.0, 2.0, 400)
    loss = FLOOR + 0.5 * delta ** 2         # loss = floor + KL(δ) ≥ floor
    fig, ax = plt.subplots(figsize=style.SIZES["single_column"])
    ax.plot(delta, loss, color=style.METHOD_STYLE["cd_sbi"]["color"],
            label="NF-MLE loss")
    panels.noise_floor_band(ax, FLOOR,
                            label=r"entropy floor $\mathbb{E}_\rho[H(X\mid\theta)]$")
    ax.plot([0.0], [FLOOR], "o", color="#e8743b", zorder=5)
    ax.annotate("truth: loss = floor", xy=(0.0, FLOOR), xytext=(0.4, FLOOR + 0.6),
                arrowprops=dict(arrowstyle="->", color="#e8743b"), fontsize=7)
    ax.set_xlabel(r"density misfit $\delta$"); ax.set_ylabel("NF-MLE loss")
    ax.set_title(r"Strict propriety: loss $\geq$ floor")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Manifest entry:**

```yaml
c4_entropy_floor:
  description: "§3.2 strict propriety: NF-MLE loss ≥ conditional-entropy floor"
  section: "3.2"
  source_runs: []
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures.c4_entropy_floor:render"
  output:
    pdf: figures/c4_entropy_floor.pdf
    png: figures/c4_entropy_floor.png
```

- [ ] **Step 6: Commit** — `git commit -m "feat(figures): C4 NF-MLE entropy floor"`.

---

## Task C5: Triangular / KR structure (§6.1)

**Files:** Create `c5_kr_structure.py`; modify manifest + smoke test.

One panel: a lower-triangular dependency map — `r_k` depends on `θ_{1..k}` (and all on X), shown as a `tril` heatmap with check marks.

- [ ] **Step 1: Append the smoke test**

```python
def test_c5_triangular_heatmap():
    from cdsbi.analysis.figures.figures.c5_kr_structure import render
    fig = render(_spec())
    ax = fig.axes[0]
    assert len(ax.images) == 1          # the tril heatmap (imshow)
```

- [ ] **Step 2: Run to verify it fails.** → FAIL.

- [ ] **Step 3: Write the builder**

`src/cdsbi/analysis/figures/figures/c5_kr_structure.py`:

```python
"""C5 — Knothe–Rosenblatt triangular structure: r_k = r_k(θ_{1:k}; X)."""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures.manifest import FigureSpec

D = 4


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    mask = np.tril(np.ones((D, D)))
    fig, ax = plt.subplots(figsize=style.SIZES["square"])
    ax.imshow(mask, cmap="Blues", vmin=0.0, vmax=1.5, aspect="equal")
    for i in range(D):
        for j in range(D):
            if mask[i, j]:
                ax.text(j, i, "✓", ha="center", va="center", color="white", fontsize=10)
    ax.set_xticks(range(D)); ax.set_xticklabels([fr"$\theta_{{{j+1}}}$" for j in range(D)])
    ax.set_yticks(range(D)); ax.set_yticklabels([fr"$r_{{{i+1}}}$" for i in range(D)])
    ax.set_title(r"Triangular dependence: $r_k = r_k(\theta_{1:k};\, X)$")
    ax.set_xlabel(r"depends on $\theta_j$  (every $r_k$ also depends on $X$)")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Manifest entry:**

```yaml
c5_kr_structure:
  description: "§6.1 triangular autoregressive (Knothe–Rosenblatt) dependence"
  section: "6.1"
  source_runs: []
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures.c5_kr_structure:render"
  output:
    pdf: figures/c5_kr_structure.pdf
    png: figures/c5_kr_structure.png
```

- [ ] **Step 6: Commit** — `git commit -m "feat(figures): C5 KR triangular structure"`.

---

## Task C6: Position in SBI literature (§10)

**Files:** Create `c6_sbi_position.py`; modify manifest + smoke test. Uses the F1 `position_table_as_axes` panel.

- [ ] **Step 1: Append the smoke test**

```python
def test_c6_renders_one_table():
    from cdsbi.analysis.figures.figures.c6_sbi_position import render
    fig = render(_spec())
    ax = fig.axes[0]
    assert len(ax.tables) == 1
    assert ax.axison is False
```

- [ ] **Step 2: Run to verify it fails.** → FAIL.

- [ ] **Step 3: Write the builder**

`src/cdsbi/analysis/figures/figures/c6_sbi_position.py`:

```python
"""C6 — position in the SBI literature: target, single-stage, coverage."""
from __future__ import annotations

from cdsbi.analysis.figures import style, panels
from cdsbi.analysis.figures.manifest import FigureSpec

COLUMNS = ["Target", "Single-stage", "Coverage by\nconstruction", r"Finite-$d$ guar."]
ROWS = ["CD-SBI", "NPE", "NLE", "NRE", "LF2I"]
CELLS = [
    ["confidence dist.", "yes", "yes", "yes"],
    ["posterior", "yes", "no", "no"],
    ["likelihood", "yes", "no", "no"],
    ["likelihood ratio", "yes", "no", "no"],
    ["confidence set", "no", "yes", "partial"],
]


def render(spec: FigureSpec):
    style.apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=style.SIZES["double_column"])
    panels.position_table_as_axes(ax, COLUMNS, ROWS, CELLS)
    ax.set_title("CD-SBI vs other SBI methods")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Manifest entry:**

```yaml
c6_sbi_position:
  description: "§10 CD-SBI vs NPE/NLE/NRE/LF2I (target, single-stage, coverage)"
  section: "10"
  source_runs: []
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures.c6_sbi_position:render"
  output:
    pdf: figures/c6_sbi_position.pdf
    png: figures/c6_sbi_position.png
```

- [ ] **Step 6: Run the whole C-smoke file + commit**

Run: `pytest tests/integration/test_c_figures_smoke.py -v` (expect 6 passed).
```bash
git add src/cdsbi/analysis/figures/figures/c6_sbi_position.py configs/figures/manifest.yaml tests/integration/test_c_figures_smoke.py
git commit -m "feat(figures): C6 SBI-position table"
```

---

## Task C7: Render all + visual acceptance + gallery + full suite

**Files:** none (render + verify only).

- [ ] **Step 1: Render the 6 conceptual figures**

Run: `python -m cdsbi.analysis.figures.render --section 2.2 && python -m cdsbi.analysis.figures.render --section 2.3 && python -m cdsbi.analysis.figures.render --section 2.4 && python -m cdsbi.analysis.figures.render --section 3.2 && python -m cdsbi.analysis.figures.render --section 6.1 && python -m cdsbi.analysis.figures.render --section 10`
(Or simply `python -m cdsbi.analysis.figures.render --all` to (re)render everything.)
Expected: `figures/c1_pivot_picture.{pdf,png}` … `figures/c6_sbi_position.{pdf,png}` all written, nonzero.

- [ ] **Step 2: Visual acceptance (controller views each PNG)**

`Read` each of `figures/c1_pivot_picture.png` … `figures/c6_sbi_position.png` and confirm the message + legibility:
- **C1:** left histograms are shifted by θ₀; right histograms all collapse onto the `N(0,1)` curve. "the pivot standardizes every conditional."
- **C2:** the parabola-like `r(θ)` is hit at two θ by the dashed `r=c` line, both marked. "(R1) ⇒ multi-valued inverse."
- **C3:** the folded region (negative slope) is shaded. "(R2) violation ⇒ Z(θ)>1."
- **C4:** convex loss bowl sitting on the floor line, truth marked at the minimum. "loss ≥ floor, equality at truth."
- **C5:** lower-triangular check-mark heatmap, `r_k` rows vs `θ_j` cols. "triangular dependence."
- **C6:** legible 5×4 method table; CD-SBI row all "yes". "CD-SBI is the single-stage coverage-by-construction method."
- All: mathtext renders (no tofu); style applied.
Fix and re-render any that fail.

- [ ] **Step 3: Regenerate the gallery**

Run: `python -m cdsbi.analysis.figures.render --gallery`
Expected: `figures/README.md` now lists c1–c6 (grouped under sections 2.2/2.3/2.4/3.2/6.1/10) plus the e1–e10 and `_hello`.

- [ ] **Step 4: Full fast suite + size budget**

Run: `pytest -q` (expect all prior + 6 new C smoke tests; no failures).
Run: `du -sh figures/` (confirm under the 9 MB budget).

- [ ] **Step 5: Commit the rendered artifacts + gallery**

```bash
git add figures/c1_pivot_picture.pdf figures/c1_pivot_picture.png figures/c2_r1_failure.pdf figures/c2_r1_failure.png figures/c3_r2_folding.pdf figures/c3_r2_folding.png figures/c4_entropy_floor.pdf figures/c4_entropy_floor.png figures/c5_kr_structure.pdf figures/c5_kr_structure.png figures/c6_sbi_position.pdf figures/c6_sbi_position.png figures/README.md
git commit -m "feat(figures): render C1-C6 conceptual figures + gallery (visual-accepted)"
```

---

## Task C8: Manuscript integration (Parts I–IV + §10)

**Files:** Modify `cd_sbi_v7.tex`.

Insert each conceptual figure near its theory section. Anchors (verified): `subsec:2.2` (~L491), `subsec:2.3` (~L599), `subsec:2.4` (~L632), `subsec:3.2` (~L733), `subsec:6.1` (~L1808), `sec:10` (~L3483). Insert each `[htbp]` figure at the END of its (sub)section's content (just before the next `\subsection`/`\section`). Build with the CLAUDE.md LaTeX cycle (absolute `/usr/bin/pdflatex`).

- [ ] **Step 1: Insert C1 at end of §2.2**

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/c1_pivot_picture.pdf}
\caption{The pivot picture. Left: the data \(X \mid \theta_0\) shift with
\(\theta_0\). Right: the location-normal pivot \(r^*(\theta_0; X) = \theta_0 - X\)
maps every conditional onto the same \(\mathcal{N}(0, 1)\) (solid curve) ---
pointwise calibration with no prior.}
\label{fig:c1}
\end{figure}
```

- [ ] **Step 2: Insert C2 at end of §2.3**

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.6\linewidth]{figures/c2_r1_failure.pdf}
\caption{Why (R1) is needed. A pivot non-monotone in \(\theta\) is hit by a
level \(r = c\) at two parameter values, so the implied confidence
distribution's inverse is multi-valued and the confidence set is ill-defined.}
\label{fig:c2}
\end{figure}
```

- [ ] **Step 3: Insert C3 at end of §2.4**

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.6\linewidth]{figures/c3_r2_folding.pdf}
\caption{Why (R2) is needed. Where the pivot folds in \(X\)
(\(\partial r / \partial X < 0\), shaded), the change of variables
double-counts mass and \(Z(\theta) > 1\): the induced surrogate is
unnormalized. Architectural monotonicity in \(X\) rules this out by
construction.}
\label{fig:c3}
\end{figure}
```

- [ ] **Step 4: Insert C4 at end of §3.2**

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.6\linewidth]{figures/c4_entropy_floor.pdf}
\caption{Strict propriety. By Gibbs's inequality the population NF-MLE loss is
bounded below by the conditional entropy \(\mathbb{E}_\rho[H(X \mid \theta)]\)
(dashed), with equality exactly at the true density --- so minimizing the loss
drives the surrogate onto the calibration manifold.}
\label{fig:c4}
\end{figure}
```

- [ ] **Step 5: Insert C5 at end of §6.1**

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.5\linewidth]{figures/c5_kr_structure.pdf}
\caption{The triangular (Knothe--Rosenblatt) architectural class: component
\(r_k\) depends on \(\theta_1, \dots, \theta_k\) (and on \(X\)), giving a
lower-triangular Jacobian in \(\theta\) and a tractable, invertible map.}
\label{fig:c5}
\end{figure}
```

- [ ] **Step 6: Insert C6 at end of §10**

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{figures/c6_sbi_position.pdf}
\caption{CD-SBI in the SBI landscape. It shares the NF-MLE loss with SNL and
the confidence-set target with LF2I/WALDO, but is the single-stage method whose
output has frequentist coverage by construction.}
\label{fig:c6}
\end{figure}
```

- [ ] **Step 7: Add textual cross-references**

Add a short parenthetical near each section's relevant prose: `(Fig.~\ref{fig:c1})` in §2.2, `(Fig.~\ref{fig:c2})` in §2.3, `(Fig.~\ref{fig:c3})` in §2.4, `(Fig.~\ref{fig:c4})` in §3.2, `(Fig.~\ref{fig:c5})` in §6.1, `(Fig.~\ref{fig:c6})` in §10. One short parenthetical each; keep edits minimal.

- [ ] **Step 8: Build + verify**

Run: `/usr/bin/pdflatex -interaction=nonstopmode cd_sbi_v7 && /usr/bin/bibtex cd_sbi_v7 && /usr/bin/pdflatex -interaction=nonstopmode cd_sbi_v7 && /usr/bin/pdflatex -interaction=nonstopmode cd_sbi_v7`
Then: `grep -i "LaTeX Warning: Reference .* undefined\|! LaTeX Error\|File .figures/c.* not found\|cannot determine size" cd_sbi_v7.log | head`
Expected: no missing-figure errors, no undefined `fig:c*` references. Confirm `grep -c "newlabel{fig:c" cd_sbi_v7.aux` is 6.

- [ ] **Step 9: Commit**

```bash
git add cd_sbi_v7.tex
git commit -m "manuscript(I-IV,10): insert conceptual figures C1-C6"
```

---

## Self-review notes

- **Spec coverage:** all six catalogue conceptual figures built (C1–C6, Tasks C1–C6), rendered + visually accepted (C7), and inserted into the manuscript at their theory sections (C8). Completes the 16-figure suite (10 empirical + 6 conceptual).
- **C1/C3 weights gap resolved by design:** both are synthetic/closed-form (no `model.pt` dependency), which is the right call for conceptual figures and removes the only open F3 blocker the F1/F2 plans flagged.
- **Builder contract:** every C-builder is `render(spec) -> Figure`, synthetic (`source_runs: []`), no `savefig`; C4 reuses `noise_floor_band`, C6 reuses `position_table_as_axes`.
- **Placeholder scan:** every step has complete runnable code + exact commands; no TBD.
- **Type consistency:** manifest `builder` strings match each module's `render`; smoke-test helper `_spec(**kw)` matches `FigureSpec`'s fields; section labels (`subsec:2.2`…`sec:10`) verified to exist; `noise_floor_band`/`position_table_as_axes` signatures match their F1 definitions.
- **F3 scope fidelity:** only conceptual builders + manifest + manuscript; no changes to F0/F1/F2 code or to diagnostics/run.py.
