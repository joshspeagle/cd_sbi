# CD-SBI Visualizations — F0 Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the plumbing for the CD-SBI visualization suite — a `src/cdsbi/analysis/figures/` subpackage with shared style, two data-loading paths, a manifest schema, a render CLI, a gallery generator, and one synthetic-driver "hello-world" figure that exercises the whole pipeline end-to-end. No real figures yet (those are F1–F3).

**Architecture:** Five-layer subpackage (style / data_io / panels / figures / render). Figure builders are *pure* — each `render(spec)` returns a matplotlib `Figure`; the render CLI owns all disk IO (savefig). The manifest (`configs/figures/manifest.yaml`) is the single source of truth binding each figure id to its source run-dirs, builder import path, and output paths. Outputs land in a git-tracked `figures/` directory.

**Tech Stack:** matplotlib (Agg backend, shared `.mplstyle`), pandas + pyarrow (parquet aggregates), PyYAML (already a transitive dep via Hydra/omegaconf — confirmed in Task 5), pytest. No new third-party dependencies.

**Spec:** `docs/superpowers/specs/2026-05-28-cd-sbi-visualizations-design.md` (F0 milestone).

---

## File structure

```
NEW
  src/cdsbi/analysis/figures/__init__.py            # subpackage marker + public exports
  src/cdsbi/analysis/figures/style.py               # METHOD_STYLE, CANONICAL_ORDER, SIZES, apply_style()
  src/cdsbi/analysis/figures/cdsbi.mplstyle         # rcParams (packaged, loaded via importlib.resources)
  src/cdsbi/analysis/figures/data_io/__init__.py
  src/cdsbi/analysis/figures/data_io/aggregates.py  # load_aggregates(roots) -> tidy DataFrame
  src/cdsbi/analysis/figures/data_io/checkpoints.py # load_checkpoint(run_dir) -> CheckpointData
  src/cdsbi/analysis/figures/manifest.py            # FigureSpec dataclass + load_manifest()
  src/cdsbi/analysis/figures/render.py              # CLI: --fig / --all / --section / --gallery
  src/cdsbi/analysis/figures/panels/__init__.py     # empty in F0 (populated in F1)
  src/cdsbi/analysis/figures/figures/__init__.py
  src/cdsbi/analysis/figures/figures/_hello.py      # synthetic-driver hello-world figure
  configs/figures/manifest.yaml                     # one entry: _hello
  figures/.gitkeep                                  # ensure tracked dir exists pre-first-render

  tests/unit/test_figures_style.py
  tests/unit/test_figures_aggregates.py
  tests/unit/test_figures_checkpoints.py
  tests/unit/test_figures_manifest.py
  tests/unit/test_figures_hello.py
  tests/integration/test_figures_render_cli.py
  tests/figures_fixtures/__init__.py                # synthetic mini-run-dir builders for tests

MODIFY
  .gitignore                                        # negate global *.pdf for figures/
```

No changes to existing `src/cdsbi/analysis/{loaders,paper_tables}.py` — `data_io/aggregates.py` *wraps* `loaders.load_runs`, it does not replace it.

---

## Conventions locked for this milestone

- **Builder contract:** every figure module exposes `def render(spec: FigureSpec) -> matplotlib.figure.Figure`. It loads its own data (via `data_io`) from the resolved paths on `spec`, builds the figure, and returns it **without touching disk**. The render CLI saves and closes it. This keeps builders unit-testable without a filesystem.
- **Method colour/marker convention** (keyed by the method names that appear in the parquet `method` column):

  | method key | colour | marker | role |
  |---|---|---|---|
  | `cd_sbi` | `#1f2d5a` (navy) | `o` | protagonist |
  | `lf2i_bff` | `#e8743b` (orange) | `s` | |
  | `nre` | `#7f7f7f` (grey) | `X` | |
  | `nle` | `#2ca02c` (green) | `D` | |
  | `npe` | `#6baed6` (light blue) | `^` | |

  Canonical narrative order (best → worst, used for legend/category ordering): `["cd_sbi", "lf2i_bff", "nre", "nle", "npe"]`.

- **Backend:** `style.apply_style()` calls `matplotlib.use("Agg")` before importing pyplot, so everything is headless. All figure/panel tests import through `style` (or set Agg themselves).

- **`model.pt` reality (verified):** the checkpoint dict is `{"arch_metadata": {...}, "final_loss": float}` and contains **no `state_dict`/weights**. `arch_metadata` includes `loss_history_tail` (last 100 training-loss values). `data_io.checkpoints` therefore exposes `arch_metadata` + `final_loss` only. **Figures needing trained weights (C1, C3 in F3) are NOT satisfiable from current checkpoints** — see "Known gap for later milestones" at the bottom. F0 builds none of those, so this does not block F0.

## Visual acceptance protocol (applies to F0 and is inherited by F1–F3)

Automated tests verify *structure* (an Axes exists, the right number of lines/patches, the style took effect). They cannot verify a figure *reads well and conveys its intended message*. So every figure — starting with F0's hello-world — gets a mandatory **visual acceptance** step before its commit is final:

1. Render the figure to PNG (`python -m cdsbi.analysis.figures.render --fig <id>`).
2. **Actually view it.** The executing agent is multimodal — `Read` the `figures/<id>.png` file so the rendered pixels enter context. (A human executor opens it in an image viewer.)
3. Judge it against an explicit checklist:
   - **Renders at all** — not blank, not a single dot, no overlapping/clipped labels.
   - **Style applied** — top/right spines absent, grid faint, fonts legible at the target column width.
   - **Math renders** — LaTeX-style labels (e.g. `$\theta$`) show real glyphs, not `□`/tofu replacement characters.
   - **Message lands (F1–F3 only)** — a reader who knows nothing could state the figure's one-sentence takeaway from the picture alone. For cross-method figures, CDSBI is visually the protagonist (navy, foregrounded) and the colour convention is honoured.
4. If it fails any check, iterate on the builder and re-render before committing. Record the verdict in one line in the commit message or task notes (e.g. "visual: OK — spines clean, mathtext renders, sine legible").

For F0 the only "message" is "the pipeline produces a clean styled plot"; the F1–F3 plans reuse this same protocol with figure-specific takeaways.

---

## Task 1: Subpackage skeleton + gitignore for tracked figures

**Files:**
- Create: `src/cdsbi/analysis/figures/__init__.py`
- Create: `src/cdsbi/analysis/figures/data_io/__init__.py`
- Create: `src/cdsbi/analysis/figures/panels/__init__.py`
- Create: `src/cdsbi/analysis/figures/figures/__init__.py`
- Create: `figures/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create the package marker files**

`src/cdsbi/analysis/figures/__init__.py`:

```python
"""CD-SBI visualization suite: style, data loading, figure builders, render CLI.

Layers:
- style:    shared mplstyle, method colour/marker convention, size presets
- data_io:  aggregates (parquet) and checkpoints (model.pt) loaders
- panels:   reusable Axes-level chart primitives (populated in F1)
- figures:  figure-level builders, one per catalogue entry
- render:   CLI that reads the manifest and writes figures/<id>.{pdf,png}
"""
```

`src/cdsbi/analysis/figures/data_io/__init__.py`:

```python
"""Data-loading paths for figures: parquet aggregates and model.pt checkpoints."""
```

`src/cdsbi/analysis/figures/panels/__init__.py`:

```python
"""Reusable Axes-level chart primitives. Populated in milestone F1."""
```

`src/cdsbi/analysis/figures/figures/__init__.py`:

```python
"""Figure-level builders: one module per catalogue entry, each exposing render(spec)."""
```

- [ ] **Step 2: Create the tracked output directory**

```bash
mkdir -p figures
printf '' > figures/.gitkeep
```

- [ ] **Step 3: Negate the global `*.pdf` ignore for `figures/`**

The repo's `.gitignore` has a blanket `*.pdf` rule (for LaTeX build artifacts). Figure PDFs must be tracked. Append to `.gitignore`:

```gitignore

# Visualization suite outputs ARE tracked (override the global *.pdf rule above).
# Use the ** form so nested figure subdirs (e.g. figures/section_8/e1.pdf) are
# also un-ignored, not just files directly in figures/. PNGs are not globally
# ignored, but negate them too for clarity / future-proofing.
!figures/**/*.pdf
!figures/**/*.png
!figures/*.pdf
!figures/*.png
```

(Both the `**` and the single-level forms are listed: `**/*.pdf` does not match files directly in `figures/` on older Git, so the single-level line is the belt-and-suspenders for the F0 hello-world, which lives directly in `figures/`.)

- [ ] **Step 4: Verify the negation works**

Run:
```bash
touch figures/_probe.pdf && git check-ignore figures/_probe.pdf; echo "exit=$?"; rm figures/_probe.pdf
```
Expected: no path printed and `exit=1` (meaning git does NOT ignore it). If it prints `figures/_probe.pdf` with `exit=0`, the negation didn't take — check that the new lines are *after* the `*.pdf` line in `.gitignore`.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/analysis/figures figures/.gitkeep .gitignore
git commit -m "feat(figures): subpackage skeleton + track figures/ outputs"
```

---

## Task 2: Shared style — `cdsbi.mplstyle` + `style.py`

**Files:**
- Create: `src/cdsbi/analysis/figures/cdsbi.mplstyle`
- Create: `src/cdsbi/analysis/figures/style.py`
- Test: `tests/unit/test_figures_style.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_figures_style.py`:

```python
"""Style layer: method convention, size presets, headless Agg style application."""
from __future__ import annotations


def test_canonical_order_is_best_to_worst():
    from cdsbi.analysis.figures import style
    assert style.CANONICAL_ORDER == ["cd_sbi", "lf2i_bff", "nre", "nle", "npe"]


def test_every_method_has_colour_and_marker():
    from cdsbi.analysis.figures import style
    for method in style.CANONICAL_ORDER:
        entry = style.METHOD_STYLE[method]
        assert entry["color"].startswith("#")
        assert isinstance(entry["marker"], str) and len(entry["marker"]) >= 1


def test_cdsbi_is_navy():
    from cdsbi.analysis.figures import style
    assert style.METHOD_STYLE["cd_sbi"]["color"] == "#1f2d5a"


def test_sizes_presets_exist_and_are_width_height_tuples():
    from cdsbi.analysis.figures import style
    for name in ("single_column", "double_column", "square", "wide"):
        w, h = style.SIZES[name]
        assert w > 0 and h > 0


def test_apply_style_sets_agg_backend_and_returns_none():
    import matplotlib
    from cdsbi.analysis.figures import style
    assert style.apply_style() is None
    assert matplotlib.get_backend().lower() == "agg"


def test_apply_style_loads_mplstyle_rcparams():
    import matplotlib as mpl
    from cdsbi.analysis.figures import style
    style.apply_style()
    # cdsbi.mplstyle sets these; assert they took effect.
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False
    assert mpl.rcParams["mathtext.fallback"] == "cm"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_figures_style.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cdsbi.analysis.figures.style'`

- [ ] **Step 3: Write the mplstyle file**

`src/cdsbi/analysis/figures/cdsbi.mplstyle`:

```
# CD-SBI shared matplotlib style. Loaded by style.apply_style().
figure.dpi: 120
savefig.dpi: 300
savefig.bbox: tight
savefig.pad_inches: 0.02

font.family: sans-serif
font.size: 9
axes.titlesize: 10
axes.labelsize: 9
legend.fontsize: 8
xtick.labelsize: 8
ytick.labelsize: 8

mathtext.fontset: dejavusans
mathtext.fallback: cm

axes.spines.top: False
axes.spines.right: False
axes.grid: True
grid.alpha: 0.3
grid.linewidth: 0.5

lines.linewidth: 1.5
lines.markersize: 5

legend.frameon: False
```

- [ ] **Step 4: Write `style.py`**

`src/cdsbi/analysis/figures/style.py`:

```python
"""Shared figure style: method convention, size presets, headless style application."""
from __future__ import annotations

from importlib import resources

import matplotlib

# Canonical narrative order (best -> worst), used for legend and category ordering.
CANONICAL_ORDER = ["cd_sbi", "lf2i_bff", "nre", "nle", "npe"]

# Method colour/marker convention. Keys match the `method` column in index_row.parquet.
METHOD_STYLE = {
    "cd_sbi":   {"color": "#1f2d5a", "marker": "o", "label": "CD-SBI"},
    "lf2i_bff": {"color": "#e8743b", "marker": "s", "label": "LF2I-BFF"},
    "nre":      {"color": "#7f7f7f", "marker": "X", "label": "NRE"},
    "nle":      {"color": "#2ca02c", "marker": "D", "label": "NLE"},
    "npe":      {"color": "#6baed6", "marker": "^", "label": "NPE"},
}

# Figure-size presets in inches (width, height).
SIZES = {
    "single_column": (3.4, 2.6),
    "double_column": (7.0, 3.0),
    "square":        (3.4, 3.4),
    "wide":          (7.0, 2.4),
}


def apply_style() -> None:
    """Force the Agg backend and apply the packaged cdsbi.mplstyle. Idempotent."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (ensure pyplot binds to Agg)
    style_path = resources.files("cdsbi.analysis.figures") / "cdsbi.mplstyle"
    # Fail loudly and clearly if the style file isn't shipped, rather than
    # surfacing a baffling matplotlib OSError mid-figure-build.
    assert style_path.is_file(), f"cdsbi.mplstyle not found at {style_path}"
    matplotlib.style.use(str(style_path))
```

- [ ] **Step 5: Ensure the mplstyle ships as package data**

The `.mplstyle` is a non-`.py` file inside the package. Under the editable `pip install -e` used in this repo, `importlib.resources.files("cdsbi.analysis.figures")` returns a path into the live `src/` tree, so the file resolves at runtime without extra config — **for editable installs only**. A real wheel build would drop the `.mplstyle` because `[tool.setuptools.packages.find]` discovers `.py` files only. If a wheel build is ever added, declare package data:

```toml
[tool.setuptools.package-data]
"cdsbi.analysis.figures" = ["*.mplstyle"]
```

This is **out of scope for F0** (the repo only does editable installs) but is recorded so it isn't forgotten. The `assert style_path.is_file()` added above turns any future packaging regression into an immediate, legible failure.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_figures_style.py -v`
Expected: PASS (6 passed)

- [ ] **Step 7: Commit**

```bash
git add src/cdsbi/analysis/figures/style.py src/cdsbi/analysis/figures/cdsbi.mplstyle tests/unit/test_figures_style.py
git commit -m "feat(figures): shared mplstyle + method colour/marker convention"
```

---

## Task 3: Test fixtures — synthetic mini-run-dirs

**Files:**
- Create: `tests/figures_fixtures/__init__.py`

This module builds throwaway run-dirs (matching the real `index_row.parquet` + `model.pt` schema) so data_io and figure tests never depend on `outputs/` (which is gitignored and absent on fresh clones).

- [ ] **Step 1: Write the fixture builders**

`tests/figures_fixtures/__init__.py`:

```python
"""Synthetic mini-run-dir builders for figure tests.

Mirrors the real run-dir layout written by cdsbi.experiments.run._write_index_row:
a per-run directory with STATUS, index_row.parquet, and model.pt.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch


def make_run_dir(
    root: Path,
    method: str,
    budget_name: str,
    seed: int,
    coverage_error_max: float,
    final_loss: float,
    actual_params_total: int = 5000,
    loss_history_tail: list[float] | None = None,
) -> Path:
    """Create one synthetic run-dir under `root` and return its path.

    The subdir name mimics the real Hydra run-dir template
    (method=...,budget=...,seed=...) closely enough for glob-based loaders.
    """
    name = f"method={method},budget={budget_name},seed={seed}"
    rd = root / name
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "STATUS").write_text("OK")
    row = {
        "experiment": "test_exp",
        "method": method,
        "budget_name": budget_name,
        "seed": seed,
        "actual_params_total": actual_params_total,
        "coverage_error_max": coverage_error_max,
        "marginal_ks": 0.013,
        "pivot_rmse": 0.04 if method == "cd_sbi" else None,
        "final_loss": final_loss,
    }
    pd.DataFrame([row]).to_parquet(rd / "index_row.parquet")
    tail = loss_history_tail if loss_history_tail is not None else [final_loss + 0.1, final_loss]
    torch.save(
        {"arch_metadata": {"loss_history_tail": tail}, "final_loss": final_loss},
        rd / "model.pt",
    )
    return rd


def make_sweep(root: Path) -> Path:
    """Create a small multi-method, multi-seed sweep root and return it.

    Two methods x two seeds = four run-dirs. Enough to exercise grouping.
    """
    root.mkdir(parents=True, exist_ok=True)
    for method, cov in (("cd_sbi", 0.025), ("npe", 0.07)):
        for seed in (0, 1):
            make_run_dir(
                root, method=method, budget_name="medium", seed=seed,
                coverage_error_max=cov + 0.001 * seed, final_loss=1.0,
            )
    return root
```

- [ ] **Step 2: Smoke-check the fixture imports and builds**

Run (note the `PYTHONPATH=.` — the `tests` package lives outside `src/`, so a bare `python -c` from a fresh shell cannot import it; only a pytest session puts the repo root on `sys.path` automatically):
```bash
PYTHONPATH=. python -c "
import tempfile, pathlib
from tests.figures_fixtures import make_sweep
import pandas as pd, glob
d = pathlib.Path(tempfile.mkdtemp())
root = make_sweep(d / 'sweep')
runs = glob.glob(str(root / '*'))
print('n run-dirs:', len(runs))
assert len(runs) == 4
print('OK')
"
```
Expected: `n run-dirs: 4` then `OK`. (Verified: without `PYTHONPATH=.` this raises `ModuleNotFoundError: No module named 'tests.figures_fixtures'`.)

- [ ] **Step 3: Commit**

```bash
git add tests/figures_fixtures/__init__.py
git commit -m "test(figures): synthetic mini-run-dir fixtures"
```

---

## Task 4: `data_io/aggregates.py` — parquet loader

**Files:**
- Create: `src/cdsbi/analysis/figures/data_io/aggregates.py`
- Test: `tests/unit/test_figures_aggregates.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_figures_aggregates.py`:

```python
"""Aggregates loader: glob sweep roots -> one tidy DataFrame of index rows."""
from __future__ import annotations

from tests.figures_fixtures import make_sweep


def test_load_aggregates_concatenates_all_runs(tmp_path):
    from cdsbi.analysis.figures.data_io.aggregates import load_aggregates
    root = make_sweep(tmp_path / "sweep")
    df = load_aggregates([str(root)])
    assert len(df) == 4
    assert set(df["method"]) == {"cd_sbi", "npe"}
    assert "coverage_error_max" in df.columns


def test_load_aggregates_merges_multiple_roots(tmp_path):
    from cdsbi.analysis.figures.data_io.aggregates import load_aggregates
    root_a = make_sweep(tmp_path / "a")
    root_b = make_sweep(tmp_path / "b")
    df = load_aggregates([str(root_a), str(root_b)])
    assert len(df) == 8


def test_load_aggregates_empty_list_returns_empty_frame(tmp_path):
    from cdsbi.analysis.figures.data_io.aggregates import load_aggregates
    df = load_aggregates([])
    assert df.empty


def test_load_aggregates_skips_non_ok_runs(tmp_path):
    from cdsbi.analysis.figures.data_io.aggregates import load_aggregates
    root = make_sweep(tmp_path / "sweep")
    # Corrupt one run's STATUS so it is excluded. Use a default + assert so a
    # fixture-naming drift surfaces as a clear failure, not a bare StopIteration.
    bad = next(root.glob("method=cd_sbi,*seed=0*"), None)
    assert bad is not None, "fixture run-dir naming changed; update this glob"
    (bad / "STATUS").write_text("FAILED")
    df = load_aggregates([str(root)])
    assert len(df) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_figures_aggregates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cdsbi.analysis.figures.data_io.aggregates'`

- [ ] **Step 3: Write the implementation**

`src/cdsbi/analysis/figures/data_io/aggregates.py`:

```python
"""Load index_row.parquet aggregates from sweep roots into one tidy DataFrame.

Wraps the existing cdsbi.analysis.loaders.load_runs (status=OK filtering,
glob-based) so figures share the same loading semantics as paper tables.
"""
from __future__ import annotations

import os

import pandas as pd

from cdsbi.analysis.loaders import load_runs


def load_aggregates(roots: list[str]) -> pd.DataFrame:
    """Concatenate per-run index rows under each sweep root.

    `roots` are sweep directories (each containing method=...,seed=... subdirs).
    Returns an empty DataFrame if `roots` is empty or no OK runs are found.
    """
    frames = []
    for root in roots:
        pattern = os.path.join(root, "*")
        df = load_runs(pattern)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_figures_aggregates.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/analysis/figures/data_io/aggregates.py tests/unit/test_figures_aggregates.py
git commit -m "feat(figures): aggregates loader wrapping loaders.load_runs"
```

---

## Task 5: `data_io/checkpoints.py` — model.pt loader

**Files:**
- Create: `src/cdsbi/analysis/figures/data_io/checkpoints.py`
- Test: `tests/unit/test_figures_checkpoints.py`

- [ ] **Step 1: Confirm PyYAML availability for the later manifest task**

Run:
```bash
python -c "import yaml; print('pyyaml', yaml.__version__)"
```
Expected: a version string (PyYAML ships with omegaconf/Hydra, already a dependency). If this fails, add `"pyyaml>=6.0"` to `dependencies` in `pyproject.toml` and reinstall — but it should succeed.

- [ ] **Step 2: Write the failing test**

`tests/unit/test_figures_checkpoints.py`:

```python
"""Checkpoint loader: model.pt -> {arch_metadata, final_loss, loss_history_tail}."""
from __future__ import annotations

import pytest

from tests.figures_fixtures import make_run_dir


def test_load_checkpoint_returns_arch_metadata_and_final_loss(tmp_path):
    from cdsbi.analysis.figures.data_io.checkpoints import load_checkpoint
    rd = make_run_dir(
        tmp_path, method="cd_sbi", budget_name="medium", seed=0,
        coverage_error_max=0.025, final_loss=1.43,
        loss_history_tail=[1.5, 1.45, 1.43],
    )
    ck = load_checkpoint(str(rd))
    assert ck.final_loss == pytest.approx(1.43)
    assert ck.loss_history_tail == [1.5, 1.45, 1.43]
    assert isinstance(ck.arch_metadata, dict)


def test_load_checkpoint_missing_file_raises(tmp_path):
    from cdsbi.analysis.figures.data_io.checkpoints import load_checkpoint
    with pytest.raises(FileNotFoundError):
        load_checkpoint(str(tmp_path / "does_not_exist"))


def test_loss_history_tail_none_when_absent(tmp_path):
    from cdsbi.analysis.figures.data_io.checkpoints import load_checkpoint
    import torch
    rd = tmp_path / "run"
    rd.mkdir()
    torch.save({"arch_metadata": {}, "final_loss": 0.9}, rd / "model.pt")
    ck = load_checkpoint(str(rd))
    assert ck.loss_history_tail is None
    assert ck.final_loss == pytest.approx(0.9)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/unit/test_figures_checkpoints.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cdsbi.analysis.figures.data_io.checkpoints'`

- [ ] **Step 4: Write the implementation**

`src/cdsbi/analysis/figures/data_io/checkpoints.py`:

```python
"""Load model.pt checkpoints for figures that need training internals.

Verified shape of model.pt as written by cdsbi.experiments.run:
    {"arch_metadata": {... "loss_history_tail": [floats] ...}, "final_loss": float}

There is NO state_dict / weights saved. Figures needing trained weights
(C1, C3) cannot be driven from these checkpoints — see the F0 plan's
"Known gap for later milestones" note.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CheckpointData:
    """What a model.pt actually contains, plus a convenience accessor."""
    arch_metadata: dict
    final_loss: float

    @property
    def loss_history_tail(self) -> list[float] | None:
        """Last ~100 training-loss values, or None if not recorded."""
        return self.arch_metadata.get("loss_history_tail")


def load_checkpoint(run_dir: str) -> CheckpointData:
    """Load `<run_dir>/model.pt`. Raises FileNotFoundError if absent."""
    path = Path(run_dir) / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"No model.pt at {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return CheckpointData(
        arch_metadata=raw.get("arch_metadata", {}),
        final_loss=float(raw["final_loss"]),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_figures_checkpoints.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/analysis/figures/data_io/checkpoints.py tests/unit/test_figures_checkpoints.py
git commit -m "feat(figures): checkpoint loader (arch_metadata + final_loss)"
```

---

## Task 6: `manifest.py` — FigureSpec + loader

**Files:**
- Create: `src/cdsbi/analysis/figures/manifest.py`
- Create: `configs/figures/manifest.yaml`
- Test: `tests/unit/test_figures_manifest.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_figures_manifest.py`:

```python
"""Manifest schema: parse manifest.yaml into FigureSpec objects."""
from __future__ import annotations

import pytest

MANIFEST_YAML = """\
_hello:
  description: "Hello-world synthetic figure"
  section: "infra"
  source_runs: []
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures._hello:render"
  output:
    pdf: figures/_hello.pdf
    png: figures/_hello.png
e8_demo:
  description: "Demo with sources"
  section: "8.5"
  source_runs:
    - outputs/a
    - outputs/b
  builder: "cdsbi.analysis.figures.figures._hello:render"
  output:
    pdf: figures/e8_demo.pdf
    png: figures/e8_demo.png
"""


def _write(tmp_path, text):
    p = tmp_path / "manifest.yaml"
    p.write_text(text)
    return str(p)


def test_load_manifest_returns_specs_keyed_by_id(tmp_path):
    from cdsbi.analysis.figures.manifest import load_manifest
    specs = load_manifest(_write(tmp_path, MANIFEST_YAML))
    assert set(specs) == {"_hello", "e8_demo"}


def test_figurespec_fields(tmp_path):
    from cdsbi.analysis.figures.manifest import load_manifest
    spec = load_manifest(_write(tmp_path, MANIFEST_YAML))["e8_demo"]
    assert spec.id == "e8_demo"
    assert spec.section == "8.5"
    assert spec.source_runs == ["outputs/a", "outputs/b"]
    assert spec.checkpoint_runs == []   # defaults to [] when omitted
    assert spec.builder == "cdsbi.analysis.figures.figures._hello:render"
    assert spec.output_pdf == "figures/e8_demo.pdf"
    assert spec.output_png == "figures/e8_demo.png"


def test_checkpoint_runs_defaults_to_empty_when_absent(tmp_path):
    from cdsbi.analysis.figures.manifest import load_manifest
    spec = load_manifest(_write(tmp_path, MANIFEST_YAML))["e8_demo"]
    assert spec.checkpoint_runs == []


def test_resolve_builder_imports_callable(tmp_path):
    from cdsbi.analysis.figures.manifest import load_manifest
    spec = load_manifest(_write(tmp_path, MANIFEST_YAML))["_hello"]
    fn = spec.resolve_builder()
    assert callable(fn)


def test_missing_required_field_raises(tmp_path):
    from cdsbi.analysis.figures.manifest import load_manifest
    bad = "x:\n  description: no builder here\n"
    with pytest.raises(KeyError):
        load_manifest(_write(tmp_path, bad))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_figures_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cdsbi.analysis.figures.manifest'`

- [ ] **Step 3: Write the implementation**

`src/cdsbi/analysis/figures/manifest.py`:

```python
"""Manifest parsing: configs/figures/manifest.yaml -> {id: FigureSpec}."""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable

import yaml


@dataclass
class FigureSpec:
    """One figure's binding: data sources, builder, output paths."""
    id: str
    description: str
    section: str
    builder: str
    output_pdf: str
    output_png: str
    source_runs: list[str] = field(default_factory=list)
    checkpoint_runs: list[str] = field(default_factory=list)

    def resolve_builder(self) -> Callable[["FigureSpec"], object]:
        """Import the `module:function` builder reference and return the callable."""
        module_path, fn_name = self.builder.split(":")
        module = importlib.import_module(module_path)
        return getattr(module, fn_name)


def load_manifest(path: str) -> dict[str, FigureSpec]:
    """Parse a manifest YAML into FigureSpec objects keyed by figure id.

    Raises KeyError if a required field (builder/description/section/output)
    is missing for any entry.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    specs: dict[str, FigureSpec] = {}
    for fig_id, entry in raw.items():
        output = entry["output"]  # KeyError if missing -> surfaced to caller
        specs[fig_id] = FigureSpec(
            id=fig_id,
            description=entry["description"],
            section=entry["section"],
            builder=entry["builder"],
            output_pdf=output["pdf"],
            output_png=output["png"],
            source_runs=list(entry.get("source_runs") or []),
            checkpoint_runs=list(entry.get("checkpoint_runs") or []),
        )
    return specs
```

- [ ] **Step 4: Create the real manifest with the hello-world entry**

`configs/figures/manifest.yaml`:

```yaml
# CD-SBI figure manifest. Single source of truth binding figure id ->
# data sources + builder + output paths. See
# docs/superpowers/specs/2026-05-28-cd-sbi-visualizations-design.md
_hello:
  description: "Hello-world synthetic figure (F0 pipeline smoke)"
  section: "infra"
  source_runs: []
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures._hello:render"
  output:
    pdf: figures/_hello.pdf
    png: figures/_hello.png
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_figures_manifest.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/analysis/figures/manifest.py configs/figures/manifest.yaml tests/unit/test_figures_manifest.py
git commit -m "feat(figures): manifest schema (FigureSpec) + loader"
```

---

## Task 7: Hello-world figure builder

**Files:**
- Create: `src/cdsbi/analysis/figures/figures/_hello.py`
- Test: `tests/unit/test_figures_hello.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_figures_hello.py`:

```python
"""Hello-world figure: pure builder returning a styled Figure, no disk IO."""
from __future__ import annotations

import matplotlib


def _make_spec():
    from cdsbi.analysis.figures.manifest import FigureSpec
    return FigureSpec(
        id="_hello", description="hello", section="infra",
        builder="cdsbi.analysis.figures.figures._hello:render",
        output_pdf="figures/_hello.pdf", output_png="figures/_hello.png",
    )


def test_render_returns_figure_with_one_axes():
    from cdsbi.analysis.figures.figures._hello import render
    fig = render(_make_spec())
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1


def test_render_uses_agg_backend():
    from cdsbi.analysis.figures.figures import _hello
    _hello.render(_make_spec())
    assert matplotlib.get_backend().lower() == "agg"


def test_render_plots_one_line():
    from cdsbi.analysis.figures.figures._hello import render
    fig = render(_make_spec())
    ax = fig.axes[0]
    assert len(ax.lines) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_figures_hello.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cdsbi.analysis.figures.figures._hello'`

- [ ] **Step 3: Write the implementation**

`src/cdsbi/analysis/figures/figures/_hello.py`:

```python
"""Hello-world synthetic-driver figure. Exercises the F0 render pipeline.

Carries no scientific content; it is the placeholder that proves
style + builder contract + render CLI + gallery work end-to-end before
any real figure is authored in F1-F3.
"""
from __future__ import annotations

import numpy as np

from cdsbi.analysis.figures import style
from cdsbi.analysis.figures.manifest import FigureSpec


def render(spec: FigureSpec):
    """Return a styled single-panel Figure. Pure: no disk IO."""
    style.apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=style.SIZES["single_column"])
    x = np.linspace(0, 2 * np.pi, 200)
    ax.plot(x, np.sin(x), color=style.METHOD_STYLE["cd_sbi"]["color"])
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\sin(\theta)$")
    ax.set_title("CD-SBI figure pipeline — hello world")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_figures_hello.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/analysis/figures/figures/_hello.py tests/unit/test_figures_hello.py
git commit -m "feat(figures): hello-world synthetic figure (pipeline smoke)"
```

---

## Task 8: Render CLI — single figure + `--all` + `--section`

**Files:**
- Create: `src/cdsbi/analysis/figures/render.py`
- Test: `tests/integration/test_figures_render_cli.py`

The CLI owns all disk IO: it resolves the manifest, calls each builder's pure `render(spec)`, and saves the returned Figure to the spec's PDF + PNG paths. Gallery generation is added in Task 9.

- [ ] **Step 1: Write the failing test**

`tests/integration/test_figures_render_cli.py`:

```python
"""Render CLI: manifest -> builder -> saved PDF + PNG. Section/all filtering."""
from __future__ import annotations

MANIFEST = """\
_hello:
  description: "Hello-world synthetic figure"
  section: "infra"
  source_runs: []
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures._hello:render"
  output:
    pdf: {pdf}
    png: {png}
second:
  description: "Second figure, different section"
  section: "8.5"
  source_runs: []
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures._hello:render"
  output:
    pdf: {pdf2}
    png: {png2}
"""


def _manifest(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    text = MANIFEST.format(
        pdf=out / "_hello.pdf", png=out / "_hello.png",
        pdf2=out / "second.pdf", png2=out / "second.png",
    )
    mpath = tmp_path / "manifest.yaml"
    mpath.write_text(text)
    return mpath, out


def test_render_one_writes_pdf_and_png(tmp_path):
    # Import load_manifest from its home module, not via render's namespace,
    # so the test doesn't break if render.py ever switches to a lazy import.
    from cdsbi.analysis.figures.render import render_one
    from cdsbi.analysis.figures.manifest import load_manifest
    mpath, out = _manifest(tmp_path)
    specs = load_manifest(str(mpath))
    render_one(specs["_hello"])
    assert (out / "_hello.pdf").stat().st_size > 0
    assert (out / "_hello.png").stat().st_size > 0


def test_render_all_writes_every_figure(tmp_path):
    from cdsbi.analysis.figures.render import render_all
    mpath, out = _manifest(tmp_path)
    render_all(str(mpath))
    for stem in ("_hello", "second"):
        assert (out / f"{stem}.pdf").stat().st_size > 0
        assert (out / f"{stem}.png").stat().st_size > 0


def test_render_all_section_filter(tmp_path):
    from cdsbi.analysis.figures.render import render_all
    mpath, out = _manifest(tmp_path)
    render_all(str(mpath), section="8.5")
    assert (out / "second.pdf").exists()
    assert not (out / "_hello.pdf").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_figures_render_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cdsbi.analysis.figures.render'`

- [ ] **Step 3: Write the implementation**

`src/cdsbi/analysis/figures/render.py`:

```python
"""Render CLI: read the manifest, build each figure, save PDF + PNG.

Builders are pure (return a Figure); this module owns all disk IO.

Usage:
    python -m cdsbi.analysis.figures.render --fig e8_headline_summary
    python -m cdsbi.analysis.figures.render --all
    python -m cdsbi.analysis.figures.render --all --section 8.4
    python -m cdsbi.analysis.figures.render --gallery        # (added in Task 9)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cdsbi.analysis.figures.manifest import FigureSpec, load_manifest

DEFAULT_MANIFEST = "configs/figures/manifest.yaml"


def render_one(spec: FigureSpec) -> None:
    """Build one figure and save it to its PDF + PNG paths.

    Suppresses the timestamp metadata matplotlib otherwise embeds, so a
    re-render of an unchanged figure produces byte-identical output and does
    not create spurious git diffs on the tracked figures/ artifacts.
    """
    builder = spec.resolve_builder()
    fig = builder(spec)
    for out in (spec.output_pdf, spec.output_png):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    # PDF embeds a CreationDate by default; None drops it. PNG embeds a
    # Software tEXt chunk; None drops it.
    fig.savefig(spec.output_pdf, metadata={"CreationDate": None})
    fig.savefig(spec.output_png, metadata={"Software": None})
    import matplotlib.pyplot as plt
    plt.close(fig)


def render_all(manifest_path: str = DEFAULT_MANIFEST, section: str | None = None) -> list[str]:
    """Render every figure (optionally filtered by section). Returns rendered ids."""
    specs = load_manifest(manifest_path)
    rendered = []
    for fig_id, spec in specs.items():
        if section is not None and spec.section != section:
            continue
        render_one(spec)
        rendered.append(fig_id)
    return rendered


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render CD-SBI figures from the manifest.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--fig", help="Render a single figure by id.")
    parser.add_argument("--all", action="store_true", help="Render all figures.")
    parser.add_argument("--section", help="Filter --all to one manuscript section.")
    parser.add_argument("--gallery", action="store_true", help="Regenerate figures/README.md.")
    args = parser.parse_args(argv)

    if args.fig:
        specs = load_manifest(args.manifest)
        render_one(specs[args.fig])
    if args.all:
        rendered = render_all(args.manifest, section=args.section)
        print(f"Rendered {len(rendered)} figure(s): {', '.join(rendered)}")
    if args.gallery:
        from cdsbi.analysis.figures.gallery import write_gallery
        path = write_gallery(args.manifest)
        print(f"Wrote gallery: {path}")
    if not (args.fig or args.all or args.gallery):
        parser.error("nothing to do: pass --fig <id>, --all, and/or --gallery")


if __name__ == "__main__":
    main()
```

Note: the final `parser.error` guard means invoking the CLI with no action flag prints usage and exits non-zero, rather than silently doing nothing.

Note: the `--gallery` branch imports `gallery.write_gallery`, created in Task 9. The import is inside the branch, so the CLI works for `--fig`/`--all` before Task 9 lands.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/integration/test_figures_render_cli.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Render the real hello-world to disk and verify it's tracked**

Run:
```bash
python -m cdsbi.analysis.figures.render --fig _hello
git check-ignore figures/_hello.pdf; echo "ignored_exit=$?"
ls -la figures/_hello.pdf figures/_hello.png
```
Expected: both files exist with nonzero size; `ignored_exit=1` (NOT ignored — the Task 1 negation holds).

- [ ] **Step 6: Visual acceptance — actually view the rendered figure**

Per the **Visual acceptance protocol** above, `Read` the rendered PNG so the pixels enter context (a human executor opens it in an image viewer):

```
Read figures/_hello.png
```

Confirm against the checklist: renders a clean sine curve (not blank); top/right spines absent and grid faint (style applied); the `$\theta$` / `$\sin(\theta)$` axis labels show real glyphs, not tofu boxes (mathtext + fallback working). If any check fails, fix `cdsbi.mplstyle` or `_hello.py` and re-render before committing. This is the F0 proof that the whole styling pipeline produces a usable image — every F1–F3 figure repeats this step with its own message checklist.

- [ ] **Step 7: Commit (code + the rendered hello-world artifacts)**

```bash
git add src/cdsbi/analysis/figures/render.py tests/integration/test_figures_render_cli.py figures/_hello.pdf figures/_hello.png
git commit -m "feat(figures): render CLI (--fig/--all/--section) + hello-world artifact

visual: OK — sine renders, spines clean, mathtext labels legible"
```

---

## Task 9: Gallery generator — `figures/README.md`

**Files:**
- Create: `src/cdsbi/analysis/figures/gallery.py`
- Test: `tests/unit/test_figures_gallery.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_figures_gallery.py`:

```python
"""Gallery generator: manifest -> figures/README.md with PNG embeds."""
from __future__ import annotations

MANIFEST = """\
_hello:
  description: "Hello-world synthetic figure"
  section: "infra"
  source_runs: []
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures._hello:render"
  output:
    pdf: figures/_hello.pdf
    png: figures/_hello.png
e8_demo:
  description: "Cross-method summary demo"
  section: "8.5"
  source_runs:
    - outputs/8_1_baseline_sweep/run
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures._hello:render"
  output:
    pdf: figures/e8_demo.pdf
    png: figures/e8_demo.png
"""


def test_write_gallery_creates_readme(tmp_path):
    from cdsbi.analysis.figures.gallery import write_gallery
    mpath = tmp_path / "manifest.yaml"
    mpath.write_text(MANIFEST)
    out = tmp_path / "figures" / "README.md"
    path = write_gallery(str(mpath), out_path=str(out))
    text = out.read_text()
    assert "Hello-world synthetic figure" in text
    assert "Cross-method summary demo" in text
    # PNG embed for each figure
    assert "_hello.png" in text
    assert "e8_demo.png" in text
    # Source run listed for the data-bound figure
    assert "outputs/8_1_baseline_sweep/run" in text
    assert str(out) == path


def test_gallery_groups_by_section(tmp_path):
    from cdsbi.analysis.figures.gallery import write_gallery
    mpath = tmp_path / "manifest.yaml"
    mpath.write_text(MANIFEST)
    out = tmp_path / "figures" / "README.md"
    write_gallery(str(mpath), out_path=str(out))
    text = out.read_text()
    assert "## infra" in text
    assert "## 8.5" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_figures_gallery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cdsbi.analysis.figures.gallery'`

- [ ] **Step 3: Write the implementation**

`src/cdsbi/analysis/figures/gallery.py`:

```python
"""Auto-generate figures/README.md from the manifest: a browsable gallery."""
from __future__ import annotations

from pathlib import Path

from cdsbi.analysis.figures.manifest import load_manifest

DEFAULT_OUT = "figures/README.md"


def write_gallery(manifest_path: str, out_path: str = DEFAULT_OUT) -> str:
    """Render a Markdown gallery grouped by manuscript section. Returns out_path."""
    specs = load_manifest(manifest_path)

    sections: dict[str, list] = {}
    for spec in specs.values():
        sections.setdefault(spec.section, []).append(spec)

    lines = [
        "# CD-SBI Figure Gallery",
        "",
        "_Auto-generated from `configs/figures/manifest.yaml` by "
        "`python -m cdsbi.analysis.figures.render --gallery`. Do not edit by hand._",
        "",
    ]
    for section in sorted(sections):
        lines.append(f"## {section}")
        lines.append("")
        for spec in sorted(sections[section], key=lambda s: s.id):
            png_name = Path(spec.output_png).name
            lines.append(f"### {spec.id} — {spec.description}")
            lines.append("")
            lines.append(f"![{spec.id}]({png_name})")
            lines.append("")
            if spec.source_runs:
                lines.append("Source runs:")
                for run in spec.source_runs:
                    lines.append(f"- `{run}`")
                lines.append("")
    text = "\n".join(lines)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out_path
```

Note: PNG embeds use the *basename* (`_hello.png`), because `README.md` lives in `figures/` alongside the PNGs, so a relative link is just the filename.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_figures_gallery.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Generate the real gallery and verify**

Run:
```bash
python -m cdsbi.analysis.figures.render --gallery
cat figures/README.md
```
Expected: prints a gallery with an `## infra` section and the `_hello` entry embedding `_hello.png`.

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/analysis/figures/gallery.py tests/unit/test_figures_gallery.py figures/README.md
git commit -m "feat(figures): gallery generator (figures/README.md)"
```

---

## Task 10: Full-suite verification + plan close-out

**Files:** none (verification only).

- [ ] **Step 1: Run the entire fast test suite**

Run: `pytest -q`
Expected: all previously-passing tests still pass, plus **26 new figure tests** (style 6 + aggregates 4 + checkpoints 3 + manifest 5 + hello 3 + render CLI 3 + gallery 2). No failures, no errors. The `intensive` and `ablation` markers stay deselected (default `addopts`).

- [ ] **Step 2: Confirm only the figure subpackage + figures/ changed**

Run: `git status`
Expected: clean working tree (everything committed across Tasks 1–9). `figures/` contains `_hello.pdf`, `_hello.png`, `README.md`, `.gitkeep` — all tracked.

- [ ] **Step 3: Confirm the CLI end-to-end one more time**

Run:
```bash
python -m cdsbi.analysis.figures.render --all
python -m cdsbi.analysis.figures.render --gallery
git status --porcelain figures/
```
Expected: `Rendered 1 figure(s): _hello`, gallery written, and `git status --porcelain figures/` shows no changes (re-rendering is deterministic — same bytes — so nothing to re-commit). If the PNG bytes differ on re-render, that's acceptable (matplotlib metadata) — note it but don't block.

- [ ] **Step 4: Verify the figures/ size budget headroom**

Run: `du -sh figures/`
Expected: well under the 9 MB budget (the hello-world PDF+PNG are a few KB). This confirms the budget mechanism is realistic before real figures land in F1–F3.

- [ ] **Step 5: Final visual acceptance of the gallery**

`Read figures/_hello.png` one more time alongside `figures/README.md` and confirm the gallery embeds the PNG correctly (the `![_hello](_hello.png)` link resolves to the image you just viewed). This closes the F0 visual-acceptance loop: the pipeline renders a clean styled figure *and* surfaces it in the browsable gallery. No commit needed if Tasks 8–9 already committed the artifacts and the re-render produced identical bytes (the `metadata=None` suppression makes this deterministic).

---

## Known gap for later milestones (do NOT fix in F0)

**`model.pt` has no `state_dict`.** Verified: checkpoints store `{arch_metadata, final_loss}` only. Consequences for later milestones, to be resolved when those figures are planned:

- **E6 (F2b):** OK — uses `arch_metadata.loss_history_tail`, which is present.
- **E7 (F2b):** OK — regenerated in-process from the `test_trained_folding.py` recipe; needs no checkpoint.
- **C1, C3 (F3):** **NOT satisfiable** from current checkpoints — they need trained flow *weights* to evaluate `r(θ; X)`. Before F3, choose one of: (a) extend `cdsbi.experiments.run` to also save `state_dict` and re-run the one or two specific configs C1/C3 need, or (b) take the spec's documented fallback — render C1 from a fresh tiny in-process CDSBI fit, and C3 from an analytic folded `r`. Flag this in the F3 plan's opening; it is out of scope for F0.

This gap is recorded here and in `data_io/checkpoints.py`'s module docstring so the F3 plan author cannot miss it.

---

## Self-review notes

- **Spec coverage (F0 scope):** style.py + cdsbi.mplstyle (Task 2) ✓; data_io aggregates (Task 4) ✓; data_io checkpoints (Task 5) ✓; manifest schema with `section`/`source_runs`/`checkpoint_runs` (Task 6) ✓; render CLI with `--fig`/`--all`/`--section`/`--gallery` (Tasks 8–9) ✓; gallery generator (Task 9) ✓; hello-world end-to-end (Tasks 7–9) ✓; Agg backend + mathtext fallback (Task 2) ✓; smoke fixtures (Task 3) ✓; gitignore negation + size-budget check (Tasks 1, 10) ✓; panels/ placeholder for F1 (Task 1) ✓.
- **Deferred to F1–F3 by design:** real panels, the 16 catalogue figures, manuscript `\includegraphics` integration, the C1/C3 weights gap.
- **Type consistency:** `FigureSpec` fields (`source_runs`, `checkpoint_runs`, `output_pdf`, `output_png`, `resolve_builder`) are used identically in render.py and gallery.py. `CheckpointData.loss_history_tail` and `load_aggregates(roots: list[str])` signatures match their tests. Builder contract `render(spec) -> Figure` is uniform across `_hello.py`, render.py, and all tests.

### Dual-review fixes folded in (2026-05-28)

Self-review + an independent agent review of this plan produced these corrections, all applied above:
- **Visual acceptance protocol added** (the user's explicit ask): a mandatory "actually `Read` the rendered PNG and judge it" step, defined once near the top and exercised in Task 8 Step 6 + Task 10 Step 5; inherited by F1–F3.
- **Executability bug fixed:** Task 3's smoke-check now uses `PYTHONPATH=.` (verified: bare `python -c` raises `ModuleNotFoundError` for the out-of-`src` `tests` package).
- **`apply_style` asserts the `.mplstyle` exists** (clear failure vs baffling OSError); wheel-build `package-data` note recorded as out-of-scope-but-documented.
- **gitignore** uses both `**/*.pdf` and single-level `*.pdf` negations for nested-path robustness.
- **render CLI** suppresses PDF/PNG timestamp metadata so tracked artifacts re-render byte-identically (no spurious git diffs); `main()` errors on no action flag instead of silently no-op'ing.
- **Test hardening:** render-CLI test imports `load_manifest` from its home module; aggregates test uses `next(..., None)` + assert instead of a bare `next()` that could `StopIteration`.
- **Test count corrected** to 26 (was "≈23").
- Verified against matplotlib 3.10.5: `mathtext.fallback` and `axes.spines.*` rcParam keys are valid.
