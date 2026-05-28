# CD-SBI Visualizations — Design Spec

**Date:** 2026-05-28
**Status:** Approved by author; ready for implementation-plan handoff.
**Source brainstorm:** Live session 2026-05-28 following the v3 manuscript
close-out (§8.1–§8.5 complete, tables-only).

## Problem statement

`cd_sbi_v7.tex` is currently a 50-page tables-only manuscript. Tables
carry the headline empirical results well at d ≤ 2 and the §8.4
ablation, but the paper has zero figures, which weakens both pedagogy
(the architectural-monotonicity argument and the (R2) folding
mechanism want geometric pictures) and skim-readability (a reader
opening the PDF cannot tell where the results live without reading
table captions).

This spec defines a coherent visualization suite that lands in two
places: (a) inserted into the manuscript as paper figures, (b) surfaced
as a standalone `figures/` gallery for repo browsers. Targets ~16
figures (within the 12–18 budget agreed in brainstorm), balancing
conceptual (Parts I–IV) and empirical (Part V — §8.1–§8.4).

## Scope

**In scope.**

- Build a `src/cdsbi/analysis/figures/` subpackage that produces
  publication-quality PDF + gallery PNG outputs from existing
  `outputs/<sweep>/<timestamp>/` run-dirs.
- One shared style file (`cdsbi.mplstyle`) + a single method-colour /
  marker convention used everywhere.
- A `configs/figures/manifest.yaml` that ties each figure to its source
  run-dir(s) and output paths — the single source of truth for the
  paper-data binding.
- A CLI (`python -m cdsbi.analysis.figures.render`) that renders one
  figure, all figures, or the auto-generated `figures/README.md`
  gallery.
- 16 figures (6 conceptual, 10 empirical) covering the manuscript's
  theory and the four experimental sections.
- Smoke tests at both the panel (Axes-level) and figure (file-output)
  layer.
- Manuscript integration: `\includegraphics{figures/<id>.pdf}` with
  captions written in `cd_sbi_v7.tex`, landed one section at a time.

**Out of scope.**

- Interactive figures (notebooks, ipywidgets).
- Slide-deck or talk-specific variants.
- TikZ-native conceptual figures (we use matplotlib everywhere).
- Automated visual-regression testing.
- Re-running existing sweeps to regenerate source data — figures
  consume existing run-dirs; sweep re-runs are a separate concern.

## Architecture

Five-layer subpackage at `src/cdsbi/analysis/figures/`:

| Layer | Responsibility | Files |
|---|---|---|
| **style** | Shared `cdsbi.mplstyle` rcParams; method colour/marker dict; figure-size presets | `style.py`, `cdsbi.mplstyle` |
| **data_io** | Load `index_row.parquet` aggregates from one or more run-dirs into tidy pandas frames; resolve sweep paths via the manifest | `data_io.py` |
| **panels** | Reusable Axes-level chart primitives: `pit_histogram`, `coverage_curve`, `boxplot_per_method`, `jacobian_recovery_scatter`, `loss_trajectory_with_floor`, `cross_method_summary_log_y`, `position_table_as_axes`, and a few more | `panels/*.py` |
| **figures** | Figure-level builders — one per catalogue entry — that compose panels into a publication layout and emit `.pdf` + `.png` | `figures/c1_pivot_picture.py`, …, `figures/e10_cross_method_boxplots.py` |
| **render** | CLI; reads `configs/figures/manifest.yaml`, dispatches to the right figure builder, writes outputs to `figures/<id>.{pdf,png}`, regenerates `figures/README.md` | `render.py` |

**Style conventions.**

- `cdsbi.mplstyle` enforces serif math (mathtext), sans-serif text,
  colorblind palette, vector-friendly defaults (no rasterized
  elements unless explicitly opted in per panel).
- **Method colour / marker convention** (codified once in
  `style.METHOD_STYLE`):
  - CDSBI — strong dark colour (navy), filled circle marker. Always
    emphasised; the protagonist in every cross-method figure.
  - LF2I-BFF — orange, filled square.
  - NPE — light blue, filled triangle.
  - NLE — green, filled diamond.
  - NRE — grey, filled cross.
- Figure-size presets in `style.SIZES`: `single_column`,
  `double_column`, `square`, `wide`. All sized to render at 300 DPI
  PNG and crisp at the manuscript's column width.

**Manifest schema** (`configs/figures/manifest.yaml`):

```yaml
<figure_id>:
  description: "<one-line description for the gallery README>"
  source_runs:                # list of run-dirs; empty for synthetic-driver figures
    - outputs/<sweep>/<timestamp>
    - ...
  builder: cdsbi.analysis.figures.figures.<module>:render
  output:
    pdf: figures/<figure_id>.pdf
    png: figures/<figure_id>.png
```

Synthetic-driver figures (no sweep dependency) declare
`source_runs: []` and embed their toy data inside the builder module.

**Build flow.**

1. Author runs `python -m cdsbi.analysis.figures.render --all` to
   populate `figures/`.
2. Manuscript references `\includegraphics{figures/<id>.pdf}`.
3. `figures/README.md` is auto-generated from the manifest with PNG
   previews and source-run links.

## Catalogue

**Conceptual figures (Parts I–IV) — 6 figures.**

| ID | Title | Manuscript hook | Source |
|---|---|---|---|
| C1 | Pivot picture | §2.2 the pivot and the implied CD | `outputs/8_1_baseline_sweep/2026-05-27_00-57-31/` (one CDSBI run) |
| C2 | (R1) failure schematic | §2.3 the role of (R1) | Synthetic driver (1D analytic non-monotone-in-θ r) |
| C3 | (R2) folding mechanism | §2.4 + §3.5 the role of (R2) | `outputs/8_4_ablation/2026-05-27_23-55-59/` (R1-only run) |
| C4 | NF-MLE entropy floor | §3.2 strict propriety via KL non-negativity | Synthetic driver (Gibbs's inequality demo) |
| C5 | Triangular / KR structure | §6.1 the architectural class | Synthetic driver (2D toy r showing the autoregressive triangle) |
| C6 | Position in SBI literature | §10 position vs other SBI methods | Hand-curated table → matplotlib axes-as-table |

C1: two panels showing the histogram of X | θ₀ at three θ₀ values, and
r(θ₀; X) | θ₀ ~ N(0, 1) overlaid on the standard normal pdf. The
"the pivot does what it says on the tin" panel.

C2: a non-monotone-in-θ r curve with its multi-valued inverse drawn
explicitly. Pure illustration; the synthetic driver is a few-line
analytic example.

C3: the trained r(θ, T) surface from the R1-only ablation, side by
side with Z(θ) numerically. The geometric "smoking gun" for
architectural-(R2).

C4: loss vs density family, conditional entropy as the floor, strict
propriety as "loss = floor ⟹ density correct".

C5: 2D toy showing the autoregressive triangular structure (r₁
depends on θ₁ + X; r₂ depends on θ₁, θ₂ + X). Conceptual diagram.

C6: methods on rows, columns are (target — posterior / likelihood /
ratio / CD), (single-stage?), (coverage by construction?),
(finite-d guarantee?). Rendered as a matplotlib axes-as-table so
the style is consistent with the other figures.

**Empirical figures (Part V — §8.1–§8.4) — 10 figures.**

| ID | Title | Manuscript hook | Source |
|---|---|---|---|
| E1 | §8.1 CDSBI calibration | §8.1 results | `outputs/8_1_baseline_sweep/2026-05-27_00-57-31/` |
| E2 | §8.1 cross-method coverage | §8.1 results | same |
| E3 | §8.2 joint diagnostics | §8.2 results | `outputs/8_2_baseline_sweep/2026-05-27_04-36-58/` + `2026-05-27_05-32-50/` (NPE re-run) |
| E4 | §8.3 Jacobian recovery | §8.3 results | `outputs/8_3_baseline_sweep/2026-05-27_11-18-28/` |
| E5 | §8.4 doubly-monotone calibration | §8.4 results | `outputs/8_4_baseline_sweep/2026-05-28_12-05-31/` |
| E6 | §8.4 (R2) ablation: standard recipe | §8.4 ablation paragraph | `outputs/8_4_ablation/2026-05-27_23-55-59/` |
| E7 | §8.4 catastrophic folding (longer recipe) | §8.4 ablation paragraph | `tests/ablation/test_trained_folding.py` fixture / regen on demand |
| E8 | Headline cross-method summary | §8.5 synthesis | all four `8_x_baseline_sweep/` dirs |
| E9 | Budget saturation | §8.5 synthesis | same |
| E10 | Per-experiment boxplots | §8.5 synthesis | same |

E1: marginal-PIT histogram + nominal-vs-empirical coverage curve at
medium budget, one seed. "It works at d = 1."

E2: all 5 methods' coverage curves at medium budget on one axis,
CDSBI on the diagonal.

E3: joint Mahalanobis PIT histogram (left), 2D coverage error tile
across (θ₀ grid × α) (right). The inferentially-primary diagnostic
for d > 1.

E4: trained E[∂r/∂θ] heatmap vs L⁻¹ over 5 seeds, with max-element
residual annotated. Theorem A-d verified empirically.

E5: marginal-PIT + nominal-vs-empirical coverage + final-loss-vs-
entropy-floor bar at the medium budget. CDSBI at the entropy floor.

E6: training-loss trajectories for R1+R2 and R1-only flows under
the standard recipe, with the entropy floor (~0.99) drawn. R1-only
under-converges to ~1.40.

E7: a longer-recipe training-loss trajectory dipping below the
entropy floor (~0.5 vs floor 0.99). The catastrophic-folding case
captured as a regression. Data from the `test_trained_folding.py`
fixture; the script can regenerate the trajectory on demand if the
fixture is stale.

E8: 5 methods × 4 experiments coverage_error_max on one log-y axis,
with the binomial-noise floor as a horizontal band. The CLAUDE.md
headline table rendered as a figure.

E9: pivot RMSE vs total parameter budget, per method, per experiment.
CDSBI saturates earliest.

E10: 4-panel coverage_error_max boxplots, one per §8.x, 5 methods
grouped, all seeds × budgets pooled.

## Reproducibility

**Manifest is the only place run-dirs are referenced.** Updating
which sweep a figure uses is a one-YAML-edit operation; figure
builder code stays sweep-agnostic.

**Outputs are git-tracked.** `figures/<id>.pdf` and `.png` are checked
in so the manuscript builds anywhere without needing the run-dirs
themselves (run-dirs stay gitignored).

**Gallery is auto-generated.** `figures/README.md` lists each figure
with its description, embedded PNG preview, and source run-dir,
regenerated from the manifest by the `--gallery` flag.

**Manuscript integration is the last step of each empirical-figure
land.** Each figure lands as: (1) builder module + tests, (2)
manifest entry + render, (3) `cd_sbi_v7.tex` `\includegraphics`
insertion with caption. Conceptual figures group into one final
"theory-figures" commit since they don't tie 1:1 to existing
results sections.

## Testing

**Panel-level unit tests** (`tests/unit/test_panels_<name>.py`).
- Feed each panel a synthetic mini-dataframe (10–20 rows).
- Assert the returned Axes has the right number of Lines / Patches /
  Texts for the data shape.
- Assert the active style matches `cdsbi.mplstyle` (e.g.,
  `Axes.spines['top'].get_visible()` matches the style setting).
- Fast (< 1s per test).

**Figure-level integration tests** (`tests/integration/test_figures_smoke.py`).
- One test per catalogue entry.
- Call the builder with a synthetic mini-run-dir fixture (10 rows of
  parquet) or its synthetic driver.
- Assert it writes a PDF and a PNG of nonzero size to a tmp_path.
- Doesn't validate visual correctness — that is the author's
  eyeballs' job.

**Manual.** Author inspects rendered PDFs / PNGs and approves.

## Decomposition — F0–F3 milestones

The implementation plan should ladder up, not try to render all 16
figures in one shot.

**F0 — Infrastructure (no figures yet).**

- `style.py`, `cdsbi.mplstyle`, `data_io.py`, manifest schema +
  loader, render CLI, gallery generator.
- A "hello world" placeholder figure to exercise the full pipeline
  end-to-end.
- Smoke-test fixtures (synthetic mini-run-dir parquet files).
- Wire `pyproject.toml` + any imports needed.

**F1 — Empirical primitives (panels).**

- Implement reusable Axes-level panels: `pit_histogram`,
  `coverage_curve`, `boxplot_per_method`, `jacobian_recovery_scatter`,
  `loss_trajectory_with_floor`, `cross_method_summary_log_y`,
  `position_table_as_axes`, plus any small helpers (e.g.,
  `noise_floor_band`).
- Each driven by synthetic data in its unit test; no sweep
  dependency yet.

**F2 — Empirical figures E1–E10.**

- Compose panels into the 10 empirical figures, one section at a
  time:
  - E1, E2 land with a §8.1 manuscript-integration commit.
  - E3 lands with §8.2.
  - E4 lands with §8.3.
  - E5, E6, E7 land with §8.4.
  - E8, E9, E10 land with §8.5 synthesis.

**F3 — Conceptual figures C1–C6.**

- Synthetic-driver figures first (C2, C4, C5; quickest to author).
- Data-bound figures next (C1, C3; pull from existing run-dirs).
- Table-as-figure last (C6).
- Land as a single "theory figures" commit, with manuscript
  insertions distributed across Parts I, II, III, IV, and Part VI's
  §10.

Each milestone is sized to fit a bite-sized TDD plan in
`docs/superpowers/plans/` (1–3 hours' work).

## Risks and open questions

- **Figure-budget creep.** 16 figures is at the top of the
  brainstorm-agreed 12–18 range. If F2 lands before F3 and the
  manuscript already feels busy, conceptual figures can be pruned
  (drop C5 first, then C2 and C4) without losing empirical content.
- **C3 (folding mechanism) authoring complexity.** Visualising a
  trained R1-only flow's r(θ, T) surface is harder than the other
  figures because it needs to load the actual model checkpoint, not
  just the parquet aggregates. If checkpoint loading turns out to be
  fiddly, fall back to plotting an analytic 1D folded r — still makes
  the geometric point.
- **E7 source-data fragility.** The `test_trained_folding.py` fixture
  is itself seed-sensitive (commit f1d17a7 documents this). The
  figure should be regeneratable on demand from the test's training
  recipe, not pinned to one cached run, so a future seed sweep can
  refresh it.
- **Manuscript figure placement.** Inserting 16 figures into a
  tables-only paper will shift pagination significantly. Plan for one
  pagination-review pass after F3 lands.

## Acceptance criteria

- `figures/` contains 16 figure PDFs + 16 PNGs, all auto-generated
  from `configs/figures/manifest.yaml`.
- `figures/README.md` is the auto-generated gallery.
- `cd_sbi_v7.tex` references all 16 via `\includegraphics`, with
  captions written, and rebuilds cleanly under the documented LaTeX
  cycle.
- One unit test per panel module + 16 figure smoke tests pass under
  `pytest` (added to the fast suite, not gated behind the `intensive`
  or `ablation` markers).
- Style is uniform across all figures (verified by spot-check of
  three figures: one conceptual, one cross-method, one ablation).
