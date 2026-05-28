# CD-SBI Visualizations — F1 Empirical Panels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable Axes-level chart primitives (panels) that F2's empirical figures and F3's conceptual figures will compose — each a pure function driven by synthetic data in its unit test, with no sweep dependency yet.

**Architecture:** Panels live in `src/cdsbi/analysis/figures/panels/`, grouped into thematic modules (primitives / calibration / comparison / loss / recovery / tables). Every panel is an **Axes-level function**: first positional arg `ax`, draws onto it, returns `ax` (the seaborn idiom). Panels never create Figures, never call `savefig`, and assume the caller has already called `style.apply_style()` — keeping them composable inside F2 figure builders and unit-testable in isolation. Unit tests assert exact counts of plotted artists (Lines / Patches / Collections) and the method-colour convention; a final contact-sheet task renders every panel with synthetic data into one PNG for a single visual-acceptance pass.

**Tech Stack:** matplotlib 3.10.5 (Agg backend via `style.apply_style()`), numpy, pytest. No new dependencies. Builds directly on F0's `style` module (`METHOD_STYLE`, `CANONICAL_ORDER`, `SIZES`, `apply_style`).

**Spec:** `docs/superpowers/specs/2026-05-28-cd-sbi-visualizations-design.md` (F1 milestone).

---

## Panel contract (read first — applies to every task)

Every panel function obeys this contract:

1. **Signature:** `def <name>(ax, <data...>, *, <styling kwargs>) -> ax`. First positional arg is a matplotlib `Axes`. Returns the same `ax`.
2. **Pure-drawing:** draws onto `ax` only. NEVER creates a `Figure`, NEVER calls `savefig`, NEVER calls `plt.show`. (Exception: `coverage_tile` calls `ax.figure.colorbar(...)`, which legitimately adds a colorbar axes to the parent figure — that is the one allowed figure-level touch, and it is required for a heatmap to be readable.)
3. **Style assumed applied:** the caller (F2 builder, or the test) calls `style.apply_style()` first. Panels read colours/markers from `cdsbi.analysis.figures.style.METHOD_STYLE` and never hard-code method colours.
4. **Method ordering:** any panel that plots multiple methods iterates `style.CANONICAL_ORDER`, filtered to the methods present in the data, so CDSBI is always drawn first/foregrounded.

**Shared test helper.** Each test module defines this at the top (DRY within the file):

```python
def _ax():
    from cdsbi.analysis.figures import style
    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    return ax
```

**Why exact-count assertions.** matplotlib silently accepts almost anything, so "it ran" proves little. Tests assert the precise number of artists a correct call produces (e.g. a histogram with `bins=20` adds exactly 20 `Patch` objects; a 2-method line plot adds exactly 2 `Line2D`). This catches "drew nothing", "drew the diagonal N times", and "off-by-one in grouping" regressions.

---

## File structure

```
NEW
  src/cdsbi/analysis/figures/panels/primitives.py    # noise_floor_band, diagonal_reference
  src/cdsbi/analysis/figures/panels/calibration.py   # pit_histogram, coverage_curve, coverage_tile
  src/cdsbi/analysis/figures/panels/comparison.py    # boxplot_per_method, cross_method_summary_log_y, metric_vs_budget
  src/cdsbi/analysis/figures/panels/loss.py          # loss_trajectory_with_floor, loss_bar_with_floor
  src/cdsbi/analysis/figures/panels/recovery.py      # jacobian_recovery_scatter
  src/cdsbi/analysis/figures/panels/tables.py        # position_table_as_axes
  tools/panel_contact_sheet.py                        # dev script: render all panels w/ synthetic data -> one PNG

  tests/unit/test_panels_primitives.py
  tests/unit/test_panels_calibration.py
  tests/unit/test_panels_comparison.py
  tests/unit/test_panels_loss.py
  tests/unit/test_panels_recovery.py
  tests/unit/test_panels_tables.py
  tests/unit/test_panels_exports.py

MODIFY
  src/cdsbi/analysis/figures/panels/__init__.py      # re-export all panels (currently a one-line placeholder)
  .gitignore                                          # ignore the scratch contact-sheet PNG
```

No changes to F0 modules (`style`, `data_io`, `manifest`, `render`, `gallery`, `figures/_hello.py`). F1 only adds to `panels/`.

---

## Task 1: `panels/primitives.py` — shared reference lines

**Files:**
- Create: `src/cdsbi/analysis/figures/panels/primitives.py`
- Test: `tests/unit/test_panels_primitives.py`

These two tiny reference-drawing helpers are reused by several later panels (`noise_floor_band` by loss + comparison; `diagonal_reference` by calibration + recovery). Build them first so later tasks can import them.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_panels_primitives.py`:

```python
"""Shared reference primitives: noise_floor_band, diagonal_reference."""
from __future__ import annotations


def _ax():
    from cdsbi.analysis.figures import style
    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    return ax


def test_noise_floor_band_scalar_draws_one_line():
    from cdsbi.analysis.figures.panels.primitives import noise_floor_band
    ax = _ax()
    out = noise_floor_band(ax, 0.99)
    assert out is ax
    assert len(ax.lines) == 1
    assert len(ax.patches) == 0
    # The line is horizontal at y=0.99.
    ydata = ax.lines[0].get_ydata()
    assert ydata[0] == 0.99 and ydata[1] == 0.99


def test_noise_floor_band_halfwidth_draws_one_patch():
    from cdsbi.analysis.figures.panels.primitives import noise_floor_band
    ax = _ax()
    noise_floor_band(ax, 0.5, halfwidth=0.02)
    assert len(ax.patches) == 1
    assert len(ax.lines) == 0


def test_diagonal_reference_draws_unit_slope_line():
    from cdsbi.analysis.figures.panels.primitives import diagonal_reference
    ax = _ax()
    out = diagonal_reference(ax, lo=0.0, hi=1.0)
    assert out is ax
    assert len(ax.lines) == 1
    line = ax.lines[0]
    assert list(line.get_xdata()) == [0.0, 1.0]
    assert list(line.get_ydata()) == [0.0, 1.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_panels_primitives.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cdsbi.analysis.figures.panels.primitives'`

- [ ] **Step 3: Write the implementation**

`src/cdsbi/analysis/figures/panels/primitives.py`:

```python
"""Shared reference-line primitives reused across panels.

Axes-level helpers: take an Axes, draw a reference, return the Axes.
"""
from __future__ import annotations


def noise_floor_band(ax, floor, *, halfwidth=None, label="noise floor", color="0.5"):
    """Draw a Monte-Carlo noise floor on `ax`.

    Scalar floor -> a dashed horizontal line at y=floor.
    With `halfwidth` -> a shaded horizontal band [floor-halfwidth, floor+halfwidth].
    """
    if halfwidth is None:
        ax.axhline(floor, ls="--", lw=1.0, color=color, label=label)
    else:
        ax.axhspan(floor - halfwidth, floor + halfwidth, color=color, alpha=0.15, label=label)
    return ax


def diagonal_reference(ax, *, lo=0.0, hi=1.0, label=None, color="0.5"):
    """Draw the y=x identity line from (lo, lo) to (hi, hi)."""
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=color, label=label)
    return ax
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_panels_primitives.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/analysis/figures/panels/primitives.py tests/unit/test_panels_primitives.py
git commit -m "feat(figures): panel primitives (noise_floor_band, diagonal_reference)"
```

---

## Task 2: `panels/calibration.py` — PIT, coverage curve, coverage tile

**Files:**
- Create: `src/cdsbi/analysis/figures/panels/calibration.py`
- Test: `tests/unit/test_panels_calibration.py`

Three calibration-diagnostic panels. `pit_histogram` (the marginal/joint PIT view for E1/E3/E5), `coverage_curve` (single-series nominal-vs-empirical, called once per method by E1/E2/E5), `coverage_tile` (the 2D coverage-error heatmap for E3).

- [ ] **Step 1: Write the failing test**

`tests/unit/test_panels_calibration.py`:

```python
"""Calibration panels: pit_histogram, coverage_curve, coverage_tile."""
from __future__ import annotations

import numpy as np


def _ax():
    from cdsbi.analysis.figures import style
    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    return ax


def test_pit_histogram_draws_bins_patches_plus_uniform_line():
    from cdsbi.analysis.figures.panels.calibration import pit_histogram
    ax = _ax()
    rng = np.random.default_rng(0)
    out = pit_histogram(ax, rng.uniform(0, 1, size=500), bins=20)
    assert out is ax
    assert len(ax.patches) == 20            # one Rectangle per bin
    assert len(ax.lines) == 1               # uniform-density reference line
    assert ax.get_xlim() == (0.0, 1.0)


def test_pit_histogram_uses_cdsbi_navy_by_default():
    from cdsbi.analysis.figures.panels.calibration import pit_histogram
    from cdsbi.analysis.figures import style
    ax = _ax()
    pit_histogram(ax, np.linspace(0, 1, 100), bins=10)
    facecolor = ax.patches[0].get_facecolor()
    import matplotlib.colors as mcolors
    assert mcolors.to_hex(facecolor) == style.METHOD_STYLE["cd_sbi"]["color"]


def test_coverage_curve_draws_single_series():
    from cdsbi.analysis.figures.panels.calibration import coverage_curve
    ax = _ax()
    nominal = np.array([0.5, 0.68, 0.9, 0.95])
    empirical = np.array([0.49, 0.67, 0.91, 0.95])
    out = coverage_curve(ax, nominal, empirical, color="#1f2d5a", marker="o")
    assert out is ax
    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_xdata()) == list(nominal)


def test_coverage_tile_draws_one_quadmesh_and_a_colorbar():
    from cdsbi.analysis.figures.panels.calibration import coverage_tile
    ax = _ax()                              # fig starts with exactly 1 axes
    theta0 = np.array([-2.0, 0.0, 2.0])
    alpha = np.array([0.5, 0.68, 0.9, 0.95])
    error = np.abs(np.random.default_rng(1).normal(0, 0.02, size=(3, 4)))
    out = coverage_tile(ax, theta0, alpha, error)
    assert out is ax
    assert len(ax.collections) == 1         # the QuadMesh
    # The colorbar is required for a readable heatmap; it adds a second axes
    # to the parent figure. Assert it so TDD actually drives that requirement
    # (deleting the ax.figure.colorbar call must break this test).
    assert len(ax.figure.axes) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_panels_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cdsbi.analysis.figures.panels.calibration'`

- [ ] **Step 3: Write the implementation**

`src/cdsbi/analysis/figures/panels/calibration.py`:

```python
"""Calibration-diagnostic panels: PIT histogram, coverage curve, coverage tile."""
from __future__ import annotations

from cdsbi.analysis.figures import style


def pit_histogram(ax, pit_values, *, bins=20, color=None, label=None):
    """Histogram of PIT values on [0, 1] with a uniform-density reference line.

    A perfectly calibrated pivot has PIT ~ Uniform(0, 1), i.e. a flat histogram
    at density 1. `pit_values` is the raw 1-D array of PIT transforms.
    """
    color = color or style.METHOD_STYLE["cd_sbi"]["color"]
    ax.hist(pit_values, bins=bins, range=(0.0, 1.0), density=True,
            color=color, alpha=0.85, label=label)
    ax.axhline(1.0, ls="--", lw=1.0, color="0.5")
    ax.set_xlabel("PIT")
    ax.set_ylabel("density")
    ax.set_xlim(0.0, 1.0)
    return ax


def coverage_curve(ax, nominal, empirical, *, color=None, marker=None, label=None):
    """Plot one method's empirical coverage against nominal coverage.

    Single-series: F2 calls this once per method and draws the y=x reference
    separately (panels.primitives.diagonal_reference) so the diagonal appears once.
    """
    ax.plot(nominal, empirical, marker=marker, color=color, label=label)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    return ax


def coverage_tile(ax, theta0, alpha, error, *, cmap="magma"):
    """Heatmap of coverage error |empirical - nominal| over (theta_0 x alpha).

    `error` has shape (len(theta0), len(alpha)). Adds a colorbar to the parent
    figure (the one allowed figure-level touch — a heatmap is unreadable without it).
    """
    im = ax.pcolormesh(alpha, theta0, error, cmap=cmap, shading="auto")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\theta_0$")
    ax.figure.colorbar(im, ax=ax, label="coverage error")
    return ax
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_panels_calibration.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/analysis/figures/panels/calibration.py tests/unit/test_panels_calibration.py
git commit -m "feat(figures): calibration panels (pit_histogram, coverage_curve, coverage_tile)"
```

---

## Task 3: `panels/comparison.py` — cross-method comparisons

**Files:**
- Create: `src/cdsbi/analysis/figures/panels/comparison.py`
- Test: `tests/unit/test_panels_comparison.py`

Three cross-method panels: `boxplot_per_method` (seed×budget distribution, E10), `cross_method_summary_log_y` (scalar headline across experiments, E8), `metric_vs_budget` (saturation lines, E9). All iterate `style.CANONICAL_ORDER` so CDSBI leads.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_panels_comparison.py`:

```python
"""Cross-method comparison panels: boxplot, log-y summary, metric-vs-budget."""
from __future__ import annotations

import numpy as np


def _ax():
    from cdsbi.analysis.figures import style
    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    return ax


def test_boxplot_per_method_one_box_per_method_in_canonical_order():
    from cdsbi.analysis.figures.panels.comparison import boxplot_per_method
    ax = _ax()
    data = {
        "npe": np.array([0.06, 0.07, 0.05]),
        "cd_sbi": np.array([0.025, 0.024, 0.026]),
        "nle": np.array([0.10, 0.11, 0.09]),
    }
    out = boxplot_per_method(ax, data)
    assert out is ax
    # One box per method present (3); x tick labels follow CANONICAL_ORDER filtered.
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["CD-SBI", "NLE", "NPE"]


def test_cross_method_summary_log_y_one_line_per_method_plus_floor():
    from cdsbi.analysis.figures.panels.comparison import cross_method_summary_log_y
    ax = _ax()
    data = {
        "cd_sbi": {"8.1": 0.025, "8.2": 0.025, "8.3": 0.025, "8.4": 0.031},
        "nle":    {"8.1": 0.025, "8.2": 0.08, "8.3": 0.11, "8.4": 0.21},
    }
    out = cross_method_summary_log_y(ax, data, floor=0.02)
    assert out is ax
    assert ax.get_yscale() == "log"
    assert len(ax.lines) == 3               # 2 method lines + 1 floor line


def test_metric_vs_budget_log_x_one_line_per_method():
    from cdsbi.analysis.figures.panels.comparison import metric_vs_budget
    ax = _ax()
    series = {
        "cd_sbi": (np.array([1000, 5000, 25000]), np.array([0.03, 0.025, 0.025])),
        "npe":    (np.array([1000, 5000, 25000]), np.array([0.07, 0.06, 0.05])),
    }
    out = metric_vs_budget(ax, series)
    assert out is ax
    assert ax.get_xscale() == "log"
    assert len(ax.lines) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_panels_comparison.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cdsbi.analysis.figures.panels.comparison'`

- [ ] **Step 3: Write the implementation**

`src/cdsbi/analysis/figures/panels/comparison.py`:

```python
"""Cross-method comparison panels."""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures import style


def _ordered(methods_present):
    """Methods present, in canonical best->worst order."""
    return [m for m in style.CANONICAL_ORDER if m in methods_present]


def boxplot_per_method(ax, data, *, ylabel="coverage error (max)"):
    """Boxplot of a per-method metric distribution (across seeds/budgets).

    `data` maps method -> 1-D array of values. Boxes appear in canonical order,
    each filled with its method colour.
    """
    methods = _ordered(data)
    values = [np.asarray(data[m]) for m in methods]
    bp = ax.boxplot(values, patch_artist=True,
                    tick_labels=[style.METHOD_STYLE[m]["label"] for m in methods])
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(style.METHOD_STYLE[m]["color"])
        patch.set_alpha(0.7)
    ax.set_ylabel(ylabel)
    return ax


def cross_method_summary_log_y(ax, data, *, floor=None, ylabel="coverage error (max)"):
    """One log-y line per method across experiments.

    `data` maps method -> {experiment_label: scalar metric}. All methods must
    share the same experiment labels (insertion order of the first method sets
    the x-axis order).
    """
    methods = _ordered(data)
    experiments = list(next(iter(data.values())).keys())
    x = np.arange(len(experiments))
    for m in methods:
        ys = [data[m][e] for e in experiments]
        st = style.METHOD_STYLE[m]
        ax.plot(x, ys, marker=st["marker"], color=st["color"], label=st["label"])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.set_ylabel(ylabel)
    if floor is not None:
        ax.axhline(floor, ls="--", lw=1.0, color="0.5", label="noise floor")
    return ax


def metric_vs_budget(ax, series, *, xlabel="parameters", ylabel="coverage error (max)"):
    """One log-x line per method: metric vs total parameter budget.

    `series` maps method -> (budgets_array, metric_array).
    """
    methods = _ordered(series)
    for m in methods:
        budgets, metric = series[m]
        st = style.METHOD_STYLE[m]
        ax.plot(budgets, metric, marker=st["marker"], color=st["color"], label=st["label"])
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_panels_comparison.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/analysis/figures/panels/comparison.py tests/unit/test_panels_comparison.py
git commit -m "feat(figures): comparison panels (boxplot, log-y summary, metric-vs-budget)"
```

---

## Task 4: `panels/loss.py` — training-loss panels

**Files:**
- Create: `src/cdsbi/analysis/figures/panels/loss.py`
- Test: `tests/unit/test_panels_loss.py`

Two loss panels driving the §8.4 ablation figures: `loss_trajectory_with_floor` (full-trajectory line, E7) and `loss_bar_with_floor` (tail-mean grouped bars across budgets, E6). Both draw the entropy floor via `noise_floor_band`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_panels_loss.py`:

```python
"""Training-loss panels: loss_trajectory_with_floor, loss_bar_with_floor."""
from __future__ import annotations

import numpy as np


def _ax():
    from cdsbi.analysis.figures import style
    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    return ax


def test_loss_trajectory_one_line_per_series_plus_floor():
    from cdsbi.analysis.figures.panels.loss import loss_trajectory_with_floor
    ax = _ax()
    trajectories = {
        "R1+R2": np.linspace(1.4, 0.99, 50),
        "R1 only": np.linspace(1.4, 1.40, 50),
    }
    out = loss_trajectory_with_floor(ax, trajectories, floor=0.99)
    assert out is ax
    assert len(ax.lines) == 3               # 2 trajectories + 1 floor line


def test_loss_trajectory_without_floor_has_no_extra_line():
    from cdsbi.analysis.figures.panels.loss import loss_trajectory_with_floor
    ax = _ax()
    out = loss_trajectory_with_floor(ax, {"a": np.array([1.0, 0.9])})
    assert len(ax.lines) == 1


def test_loss_bar_with_floor_one_bar_per_series_group():
    from cdsbi.analysis.figures.panels.loss import loss_bar_with_floor
    ax = _ax()
    bars = {
        "R1+R2": np.array([0.986, 0.985, 0.983, 0.983]),
        "R1 only": np.array([1.419, 1.406, 1.391, 1.372]),
    }
    out = loss_bar_with_floor(ax, bars, floor=0.99,
                              group_labels=["S", "M", "L", "XL"])
    assert out is ax
    assert len(ax.patches) == 8             # 2 series x 4 budget groups
    assert len(ax.lines) == 1               # floor line
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_panels_loss.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cdsbi.analysis.figures.panels.loss'`

- [ ] **Step 3: Write the implementation**

`src/cdsbi/analysis/figures/panels/loss.py`:

```python
"""Training-loss panels for the (R2) ablation figures."""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures.panels.primitives import noise_floor_band


def loss_trajectory_with_floor(ax, trajectories, *, floor=None):
    """Plot NF-MLE loss vs training step for one or more runs.

    `trajectories` maps label -> 1-D loss array. With `floor`, draws the
    entropy lower bound as a dashed horizontal line.
    """
    for label, losses in trajectories.items():
        losses = np.asarray(losses)
        ax.plot(np.arange(len(losses)), losses, label=label)
    ax.set_xlabel("training step")
    ax.set_ylabel("NF-MLE loss")
    if floor is not None:
        noise_floor_band(ax, floor, label="entropy floor")
    return ax


def loss_bar_with_floor(ax, bars, *, floor=None, group_labels=None):
    """Grouped bar chart of a (tail-mean) loss per series across budget groups.

    `bars` maps series_label -> 1-D array, one value per group. With `floor`,
    draws the entropy lower bound as a dashed horizontal line.
    """
    series = list(bars.keys())
    n_groups = len(next(iter(bars.values())))
    x = np.arange(n_groups)
    width = 0.8 / len(series)
    for i, s in enumerate(series):
        ax.bar(x + i * width, np.asarray(bars[s]), width=width, label=s)
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    if group_labels is not None:
        ax.set_xticklabels(group_labels)
    ax.set_ylabel("final NF-MLE loss")
    if floor is not None:
        noise_floor_band(ax, floor, label="entropy floor")
    return ax
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_panels_loss.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/analysis/figures/panels/loss.py tests/unit/test_panels_loss.py
git commit -m "feat(figures): loss panels (trajectory + tail-mean bars with floor)"
```

---

## Task 5: `panels/recovery.py` — Jacobian recovery scatter

**Files:**
- Create: `src/cdsbi/analysis/figures/panels/recovery.py`
- Test: `tests/unit/test_panels_recovery.py`

One panel for E4: scatter of trained Jacobian elements against the closed-form truth (`L⁻¹`), with a y=x reference. Element-wise scatter works for any dimension and shows recovery at a glance.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_panels_recovery.py`:

```python
"""Jacobian recovery panel: trained-vs-truth element scatter."""
from __future__ import annotations

import numpy as np


def _ax():
    from cdsbi.analysis.figures import style
    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    return ax


def test_jacobian_recovery_scatter_one_collection_plus_diagonal():
    from cdsbi.analysis.figures.panels.recovery import jacobian_recovery_scatter
    ax = _ax()
    truth = np.array([[1.0, 0.0], [-0.5, 0.8]])
    trained = truth + np.array([[0.01, 0.0], [-0.02, 0.015]])
    out = jacobian_recovery_scatter(ax, trained, truth)
    assert out is ax
    assert len(ax.collections) == 1         # the scatter PathCollection
    assert len(ax.lines) == 1               # the y=x reference


def test_jacobian_recovery_scatter_plots_all_elements():
    from cdsbi.analysis.figures.panels.recovery import jacobian_recovery_scatter
    ax = _ax()
    truth = np.zeros((3, 3))
    trained = np.zeros((3, 3))
    jacobian_recovery_scatter(ax, trained, truth)
    offsets = ax.collections[0].get_offsets()
    assert offsets.shape[0] == 9            # all 3x3 elements scattered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_panels_recovery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cdsbi.analysis.figures.panels.recovery'`

- [ ] **Step 3: Write the implementation**

`src/cdsbi/analysis/figures/panels/recovery.py`:

```python
"""Jacobian recovery panel: trained E[dr/dtheta] vs closed-form L^-1."""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures.panels.primitives import diagonal_reference


def jacobian_recovery_scatter(ax, trained, truth, *, color=None):
    """Scatter trained Jacobian elements against truth (L^-1), with a y=x line.

    `trained` and `truth` are same-shape arrays of Jacobian-matrix elements;
    they are flattened and plotted element-wise. Points on the diagonal mean
    perfect recovery (Theorem A-d).
    """
    trained = np.asarray(trained).ravel()
    truth = np.asarray(truth).ravel()
    color = color or style.METHOD_STYLE["cd_sbi"]["color"]
    ax.scatter(truth, trained, color=color, s=30, zorder=3)
    lo = float(min(truth.min(), trained.min()))
    hi = float(max(truth.max(), trained.max()))
    diagonal_reference(ax, lo=lo, hi=hi, label="$y=x$")
    ax.set_xlabel(r"truth $L^{-1}$ element")
    ax.set_ylabel(r"trained $\mathbb{E}[\partial r/\partial\theta]$ element")
    return ax
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_panels_recovery.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/analysis/figures/panels/recovery.py tests/unit/test_panels_recovery.py
git commit -m "feat(figures): jacobian recovery scatter panel"
```

---

## Task 6: `panels/tables.py` — position-table-as-axes

**Files:**
- Create: `src/cdsbi/analysis/figures/panels/tables.py`
- Test: `tests/unit/test_panels_tables.py`

One panel for C6 (position in the SBI literature): render a hand-curated table onto an Axes so its styling matches the other figures.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_panels_tables.py`:

```python
"""Table-as-axes panel: position_table_as_axes."""
from __future__ import annotations


def _ax():
    from cdsbi.analysis.figures import style
    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    return ax


def test_position_table_renders_one_table_with_axis_off():
    from cdsbi.analysis.figures.panels.tables import position_table_as_axes
    ax = _ax()
    columns = ["Target", "Single-stage", "Coverage by construction"]
    rows = ["CD-SBI", "NPE", "LF2I"]
    cells = [
        ["CD", "yes", "yes"],
        ["posterior", "yes", "no"],
        ["confidence set", "no", "yes"],
    ]
    out = position_table_as_axes(ax, columns, rows, cells)
    assert out is ax
    assert len(ax.tables) == 1
    assert ax.axison is False               # axis turned off for a clean table
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_panels_tables.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cdsbi.analysis.figures.panels.tables'`

- [ ] **Step 3: Write the implementation**

`src/cdsbi/analysis/figures/panels/tables.py`:

```python
"""Render a curated comparison table onto an Axes (C6 position figure)."""
from __future__ import annotations


def position_table_as_axes(ax, columns, rows, cells):
    """Draw a table on `ax` with the figure suite's styling.

    `columns`: list of column headers.
    `rows`: list of row labels.
    `cells`: list of rows, each a list of cell strings; shape len(rows) x len(columns).
    """
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        rowLabels=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    return ax
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_panels_tables.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/analysis/figures/panels/tables.py tests/unit/test_panels_tables.py
git commit -m "feat(figures): position-table-as-axes panel"
```

---

## Task 7: Panel exports + contact sheet + visual acceptance + verification

**Files:**
- Modify: `src/cdsbi/analysis/figures/panels/__init__.py`
- Create: `tools/panel_contact_sheet.py`
- Modify: `.gitignore`
- Test: `tests/unit/test_panels_exports.py`

Wire up convenience re-exports, build a synthetic contact sheet that renders every panel onto one figure for a single visual-acceptance pass, and verify the full suite.

- [ ] **Step 1: Write the failing exports test**

`tests/unit/test_panels_exports.py`:

```python
"""All panels are importable from the panels package root."""
from __future__ import annotations


def test_all_panels_re_exported():
    from cdsbi.analysis.figures import panels
    for name in (
        "noise_floor_band", "diagonal_reference",
        "pit_histogram", "coverage_curve", "coverage_tile",
        "boxplot_per_method", "cross_method_summary_log_y", "metric_vs_budget",
        "loss_trajectory_with_floor", "loss_bar_with_floor",
        "jacobian_recovery_scatter",
        "position_table_as_axes",
    ):
        assert callable(getattr(panels, name)), f"{name} not re-exported"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_panels_exports.py -v`
Expected: FAIL with `AttributeError: module 'cdsbi.analysis.figures.panels' has no attribute 'noise_floor_band'`

- [ ] **Step 3: Write the re-export `__init__.py`**

Replace `src/cdsbi/analysis/figures/panels/__init__.py` with:

```python
"""Reusable Axes-level chart primitives (panels).

Every panel takes an Axes as its first argument, draws onto it, and returns it.
Callers must have applied the shared style (style.apply_style()) first.
"""
from cdsbi.analysis.figures.panels.calibration import (
    coverage_curve,
    coverage_tile,
    pit_histogram,
)
from cdsbi.analysis.figures.panels.comparison import (
    boxplot_per_method,
    cross_method_summary_log_y,
    metric_vs_budget,
)
from cdsbi.analysis.figures.panels.loss import (
    loss_bar_with_floor,
    loss_trajectory_with_floor,
)
from cdsbi.analysis.figures.panels.primitives import (
    diagonal_reference,
    noise_floor_band,
)
from cdsbi.analysis.figures.panels.recovery import jacobian_recovery_scatter
from cdsbi.analysis.figures.panels.tables import position_table_as_axes

__all__ = [
    "noise_floor_band",
    "diagonal_reference",
    "pit_histogram",
    "coverage_curve",
    "coverage_tile",
    "boxplot_per_method",
    "cross_method_summary_log_y",
    "metric_vs_budget",
    "loss_trajectory_with_floor",
    "loss_bar_with_floor",
    "jacobian_recovery_scatter",
    "position_table_as_axes",
]
```

- [ ] **Step 4: Run the exports test to verify it passes**

Run: `pytest tests/unit/test_panels_exports.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Write the contact-sheet dev script**

`tools/panel_contact_sheet.py`:

```python
"""Render every F1 panel with synthetic data onto one contact sheet.

A developer/agent verification aid (NOT a manuscript figure): run it, then
view the PNG to confirm each primitive renders legibly with the shared style.

    python tools/panel_contact_sheet.py [output_path]

Defaults to ./panel_contact_sheet.png (gitignored).
"""
from __future__ import annotations

import sys

import numpy as np

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures import panels


def build():
    style.apply_style()
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes = axes.ravel()

    panels.pit_histogram(axes[0], rng.uniform(0, 1, 1000), bins=20)
    axes[0].set_title("pit_histogram")

    nominal = np.array([0.5, 0.68, 0.9, 0.95, 0.99])
    panels.diagonal_reference(axes[1], lo=0.5, hi=1.0)
    panels.coverage_curve(axes[1], nominal, nominal + rng.normal(0, 0.01, 5),
                          color=style.METHOD_STYLE["cd_sbi"]["color"], marker="o",
                          label="CD-SBI")
    axes[1].set_title("coverage_curve + diagonal")

    theta0 = np.linspace(-2, 2, 5)
    alpha = np.array([0.5, 0.68, 0.9, 0.95, 0.99])
    panels.coverage_tile(axes[2], theta0, alpha, np.abs(rng.normal(0, 0.02, (5, 5))))
    axes[2].set_title("coverage_tile")

    panels.boxplot_per_method(axes[3], {
        "cd_sbi": rng.normal(0.025, 0.003, 20),
        "lf2i_bff": rng.normal(0.07, 0.01, 20),
        "nle": rng.normal(0.11, 0.02, 20),
        "npe": rng.normal(0.08, 0.02, 20),
        "nre": rng.normal(0.15, 0.03, 20),
    })
    axes[3].set_title("boxplot_per_method")

    panels.cross_method_summary_log_y(axes[4], {
        "cd_sbi": {"8.1": 0.025, "8.2": 0.025, "8.3": 0.025, "8.4": 0.031},
        "lf2i_bff": {"8.1": 0.06, "8.2": 0.11, "8.3": 0.12, "8.4": 0.078},
        "nle": {"8.1": 0.025, "8.2": 0.08, "8.3": 0.11, "8.4": 0.21},
    }, floor=0.02)
    axes[4].legend(fontsize=6)
    axes[4].set_title("cross_method_summary_log_y")

    panels.metric_vs_budget(axes[5], {
        "cd_sbi": (np.array([1e3, 5e3, 25e3, 1e5]), np.array([0.03, 0.025, 0.025, 0.025])),
        "npe": (np.array([1e3, 5e3, 25e3, 1e5]), np.array([0.07, 0.06, 0.05, 0.05])),
    })
    axes[5].set_title("metric_vs_budget")

    steps = np.arange(200)
    panels.loss_trajectory_with_floor(axes[6], {
        "R1+R2": 0.99 + 0.4 * np.exp(-steps / 40),
        "R1 only": 1.40 + 0.0 * steps,
    }, floor=0.99)
    axes[6].legend(fontsize=6)
    axes[6].set_title("loss_trajectory_with_floor")

    panels.loss_bar_with_floor(axes[7], {
        "R1+R2": np.array([0.986, 0.985, 0.983, 0.983]),
        "R1 only": np.array([1.419, 1.406, 1.391, 1.372]),
    }, floor=0.99, group_labels=["S", "M", "L", "XL"])
    axes[7].legend(fontsize=6)
    axes[7].set_title("loss_bar_with_floor")

    L_inv = np.array([[1.0, 0.0], [-0.4, 0.9]])
    panels.jacobian_recovery_scatter(axes[8], L_inv + rng.normal(0, 0.02, (2, 2)), L_inv)
    axes[8].set_title("jacobian_recovery_scatter")

    panels.position_table_as_axes(axes[9],
        ["Target", "1-stage", "Coverage"],
        ["CD-SBI", "NPE", "LF2I"],
        [["CD", "yes", "yes"], ["posterior", "yes", "no"], ["set", "no", "yes"]])
    axes[9].set_title("position_table_as_axes")

    axes[10].axis("off")
    axes[11].axis("off")
    fig.tight_layout()
    return fig


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "panel_contact_sheet.png"
    fig = build()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Ignore the scratch contact-sheet artifact**

Append to `.gitignore`:

```gitignore

# F1 panel contact sheet is a local verification artifact, not committed.
/panel_contact_sheet.png
```

- [ ] **Step 7: Generate the contact sheet**

Run:
```bash
python tools/panel_contact_sheet.py panel_contact_sheet.png
ls -la panel_contact_sheet.png
```
Expected: `Wrote panel_contact_sheet.png` and a nonzero file. If any panel raises, fix that panel and re-run.

- [ ] **Step 8: Visual acceptance — actually view the contact sheet**

Per the spec's Visual acceptance protocol, `Read` the contact sheet so the pixels enter context:

```
Read panel_contact_sheet.png
```

Confirm against the checklist and REPORT what you see, panel by panel:
- Every one of the 10 panels renders something (none blank or error-boxed).
- Style is applied (top/right spines absent, faint grid, consistent fonts).
- Math labels (`$\theta_0$`, `$\alpha$`, `$L^{-1}$`, `$\mathbb{E}[\partial r/\partial\theta]$`, `$y=x$`) render as real glyphs, not tofu boxes.
- Method colours follow the convention (CD-SBI navy leading; box/line colours match `METHOD_STYLE`).
- `coverage_tile` shows a colorbar; `cross_method_summary` is on a log y-axis; `metric_vs_budget` is on a log x-axis; `loss_bar` shows the floor line; the position table is legible.

If any panel looks wrong, fix it and re-render before committing.

- [ ] **Step 9: Run the full fast suite**

Run: `pytest -q`
Expected: all prior tests still pass plus the new panel tests (17 new: primitives 3, calibration 4, comparison 3, loss 3, recovery 2, tables 1, exports 1). No failures. `intensive`/`ablation` stay deselected.

- [ ] **Step 10: Commit**

```bash
git add src/cdsbi/analysis/figures/panels/__init__.py tools/panel_contact_sheet.py tests/unit/test_panels_exports.py .gitignore
git commit -m "feat(figures): panel re-exports + synthetic contact sheet (F1 visual acceptance)

visual: OK — all 10 panels render, style applied, mathtext legible, colours per convention"
```

---

## Known data-sourcing gaps for F2 (record now — do NOT solve in F1)

F1 panels are pure functions of synthetic input data, so they have no sourcing dependency. F2 figures must feed them real data, and three inputs are **not currently persisted** by `cdsbi.experiments.run`:

1. **Raw PIT values (E1, E3, E5 → `pit_histogram`).** `diagnostics/marginal_pit.parquet` and `joint_mahalanobis.parquet` store only the KS scalar + noise floor, not the underlying PIT array. F2 must either (a) extend the marginal/joint PIT diagnostic writers to dump the raw PIT samples to parquet and re-run the relevant sweeps, or (b) recompute PIT from a trained model — which is blocked because `model.pt` has no `state_dict` (the C1/C3 weights gap). Option (a) is the recommended path.
2. **Jacobian matrices (E4 → `jacobian_recovery_scatter`).** `diagnostics/jacobian_recovery.parquet` stores only `max_residual`/`norm_residual` scalars, not the trained `E[∂r/∂θ]` matrix or the truth `L⁻¹`. F2 must extend the `JacobianRecovery` diagnostic to persist both matrices, then re-run §8.3.
3. **Full loss trajectory (E7 → `loss_trajectory_with_floor`).** `model.pt` stores only `loss_history_tail` (last 100 steps). E7's catastrophic-folding trajectory must be regenerated in-process from the `tests/ablation/test_trained_folding.py` recipe (already noted in the spec); the tail-100 window suffices for E6's `loss_bar_with_floor`. **Verified present:** the `model.pt` written by `run.py` (lines ~489–492) stores `{"arch_metadata": trained.arch_metadata, "final_loss": ...}`, and `arch_metadata["loss_history_tail"]` is confirmed populated (length 100) in the §8.4 ablation run-dirs — so E6's tail-mean source exists; only E7's *full* trajectory needs regeneration.

The F2 plan should open by resolving (1) and (2) — extend the diagnostic writers + a targeted re-run — before composing E1/E3/E4/E5. This is the single biggest F2 risk and is why it is flagged here at F1 close.

**F2 call-site notes (from the final F1 review):**
- `coverage_tile(ax, theta0, alpha, error)` expects `error` with shape `(len(theta0), len(alpha))` — rows are the y-axis (θ₀), columns the x-axis (α), matching `pcolormesh`. F2 must slice the coverage parquet into that orientation.
- `cross_method_summary_log_y` draws the floor via `noise_floor_band` with its default `label="noise floor"`; if an F2 legend needs a different label there, pass it at the builder level. `loss_*` panels already pass `label="entropy floor"`.
- `loss_bar_with_floor` groups series in caller dict-insertion order (intentional — the R1+R2 vs R1-only ablation has no canonical method order); F2 passes the dict in display order.

---

## Self-review notes

- **Spec coverage (F1 panels):** the spec's F1 panel list — `pit_histogram`, `coverage_curve`, `boxplot_per_method`, `jacobian_recovery_scatter`, `loss_bar_with_floor` (tail-mean), `loss_trajectory_with_floor`, `cross_method_summary_log_y`, `position_table_as_axes`, `noise_floor_band` — is fully covered (Tasks 1–6). The spec's "and a few more" is realized as `diagonal_reference` (Task 1), `coverage_tile` (Task 2, needed by E3), and `metric_vs_budget` (Task 3, needed by E9). Every catalogue figure's panel needs are now satisfiable: E1/E5 (pit+coverage), E2 (coverage), E3 (pit+tile), E4 (recovery), E6 (loss bar), E7 (loss trajectory), E8 (summary), E9 (metric-vs-budget), E10 (boxplot), C6 (table).
- **Visual acceptance:** Task 7 Steps 7–8 render and `Read` a synthetic contact sheet — one pass covering all 10 panels — honouring the spec's mandatory visual-acceptance protocol, scaled appropriately to primitives (correctness/legibility, since "message lands" is an F2 figure-level concern).
- **Placeholder scan:** every step has complete runnable code and exact commands; no TBD/TODO.
- **Type consistency:** the panel contract (`(ax, data..., *, kwargs) -> ax`) is uniform across all 11 functions and their tests. `noise_floor_band` (Task 1) and `diagonal_reference` (Task 1) are imported by `loss.py` (Task 4) and `recovery.py` (Task 5) at the exact paths created in Task 1. The `__init__.py` re-export names (Task 7) match the function names defined in Tasks 1–6 exactly, and `test_panels_exports.py` asserts that list. `boxplot_per_method` returns `ax` (not a tuple) consistently.
- **F1 scope fidelity:** no figure builders (F2), no manifest entries, no manuscript integration. Only `panels/` is touched plus the contact-sheet dev tool. The data-sourcing gaps are documented, not solved.

### Dual-review outcome (2026-05-28)

Self-review + an independent agent review of this plan. Outcome:
- **Reviewer Issue 1 (axhline ydata) — rejected after verification.** The reviewer claimed `ax.axhline(0.99).get_ydata()` returns `[0, 1]` (axes-coord transform), which would break `test_noise_floor_band_scalar_draws_one_line`. Verified empirically: `get_ydata()` returns `[0.99, 0.99]` (it is `get_xdata()` that returns `[0, 1]`). The test assertion is correct; no change.
- **Reviewer Issue 2 (coverage_tile colorbar) — accepted.** Strengthened `test_coverage_tile_*` to also assert `len(ax.figure.axes) == 2`, so deleting the colorbar call now breaks the test (closes a TDD gap). Verified colorbar → `fig.axes == 2`.
- **Reviewer Issue 5 (loss_history_tail unverified) — closed with evidence.** Confirmed `arch_metadata["loss_history_tail"]` is present (length 100) in the §8.4 ablation run-dirs; recorded in gap note 3. E6's source exists; only E7 needs regeneration.
- Decomposition, panel contract, spec/figure coverage, and the data-sourcing gap analysis were all confirmed sound. Verified the matplotlib 3.10.5 artist-count facts the tests assert (hist/bar/scatter/pcolormesh/axhline/axhspan counts, `axison`, hex round-trip).
