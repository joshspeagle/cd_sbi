# CD-SBI Visualizations — Design Spec

**Date:** 2026-05-28
**Status:** Approved by user 2026-05-28 (post dual-review revision).
Ready for implementation-plan handoff (F0 first).
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
- Unit tests at the panel (Axes-level) layer with exact-count
  assertions, and smoke tests at the figure (file-output) layer with
  Axes-count assertions.
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
| **style** | Shared `cdsbi.mplstyle` rcParams (packaged inside the subpackage; loaded via `plt.style.use(importlib.resources.files(...))`); method colour/marker dict; figure-size presets; backend pinned to `Agg` for headless rendering; mathtext fallback to `cm` if serif font unavailable | `style.py`, `cdsbi.mplstyle` |
| **data_io** | Two data paths: (a) `aggregates.py` loads `index_row.parquet` from one or more run-dirs into tidy pandas frames (the existing `loaders.load_runs` pattern); (b) `checkpoints.py` loads `model.pt` and returns `{arch_metadata, final_loss}` — **note: current `model.pt` saves NO `state_dict`/weights** (verified in F0), so it serves `loss_history_tail` (E6, E7) but NOT figures that need trained weights (C1, C3). The C1/C3 weights gap must be resolved before F3 (re-run those configs saving weights, or use the analytic/in-process fallback). Both paths resolve sweep paths via the manifest. | `data_io/aggregates.py`, `data_io/checkpoints.py` |
| **panels** | Reusable Axes-level chart primitives: `pit_histogram`, `coverage_curve`, `boxplot_per_method`, `jacobian_recovery_scatter`, `loss_bar_with_floor` (tail-mean, not trajectory — see catalog), `loss_trajectory_with_floor` (only the tail-100 window; used by E7's regen path), `cross_method_summary_log_y`, `position_table_as_axes`, and a few more | `panels/*.py` |
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
  section: "<manuscript section label, e.g. '8.4', '2.4', 'theory'>"   # for --section= filter
  source_runs:                # list of run-dirs (parquet aggregates); empty for synthetic-driver figures
    - outputs/<sweep>/<timestamp>
    - ...
  checkpoint_runs:             # list of run-dirs whose model.pt the figure needs; usually [] or one entry
    - outputs/<sweep>/<timestamp>/method=...,seed=N
  builder: cdsbi.analysis.figures.figures.<module>:render
  output:
    pdf: figures/<figure_id>.pdf
    png: figures/<figure_id>.png
```

Synthetic-driver figures (no sweep dependency) declare both
`source_runs: []` and `checkpoint_runs: []` and embed their toy data
inside the builder module. The `section` field powers
`render --section=8.4` to rebuild just one experiment's figures.

**Build flow.**

1. Author runs `python -m cdsbi.analysis.figures.render --all` to
   populate `figures/`.
2. Manuscript references `\includegraphics{figures/<id>.pdf}`.
3. `figures/README.md` is auto-generated from the manifest with PNG
   previews and source-run links.

## Catalogue

**Conceptual figures (Parts I–IV) — 6 figures.**

| ID | Title | Manuscript hook | Source | Needs checkpoint? |
|---|---|---|---|---|
| C1 | Pivot picture | §2.2 the pivot and the implied CD | `outputs/8_1_baseline_sweep/2026-05-27_00-57-31/` (one CDSBI run) | Yes — needs trained CDSBI flow to compute r(θ₀; X) over a sample |
| C2 | (R1) failure schematic | §2.3 the role of (R1) | Synthetic driver (1D analytic non-monotone-in-θ r) | No |
| C3 | (R2) folding mechanism | §2.4 + §3.5 the role of (R2) | `outputs/8_4_ablation/2026-05-27_23-55-59/` (R1-only run) | Yes — needs trained R1-only flow to surface r(θ, T); see C3 fallback below |
| C4 | NF-MLE entropy floor | §3.2 strict propriety via KL non-negativity | Synthetic driver (Gibbs's inequality demo) | No |
| C5 | Triangular / KR structure | §6.1 the architectural class | Synthetic driver (2D toy r showing the autoregressive triangle) | No |
| C6 | Position in SBI literature | §10 position vs other SBI methods | Hand-curated table → matplotlib axes-as-table | No |

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

| ID | Title | Manuscript hook | Source | Needs checkpoint? |
|---|---|---|---|---|
| E1 | §8.1 CDSBI calibration | §8.1 results | `outputs/8_1_baseline_sweep/2026-05-27_00-57-31/` | No |
| E2 | §8.1 cross-method coverage | §8.1 results | same | No |
| E3 | §8.2 joint diagnostics | §8.2 results | `outputs/8_2_baseline_sweep/2026-05-27_04-36-58/` + `2026-05-27_05-32-50/` (NPE re-run) | No |
| E4 | §8.3 Jacobian recovery | §8.3 results | `outputs/8_3_baseline_sweep/2026-05-27_11-18-28/` | No |
| E5 | §8.4 doubly-monotone calibration | §8.4 results | `outputs/8_4_baseline_sweep/2026-05-28_12-05-31/` | No |
| E6 | §8.4 (R2) ablation: tail-mean loss | §8.4 ablation paragraph | `outputs/8_4_ablation/2026-05-27_23-55-59/` | Yes — needs `arch_metadata.loss_history_tail` from R1+R2 and R1-only runs |
| E7 | §8.4 catastrophic folding (longer recipe) | §8.4 ablation paragraph | regenerated on demand from `tests/ablation/test_trained_folding.py` recipe | Implicit — fixture captures the full trajectory in-memory; no model.pt needed for the figure |
| E8 | Headline cross-method summary | §8.5 synthesis | all four `8_x_baseline_sweep/` dirs | No |
| E9 | Budget saturation | §8.5 synthesis | same | No |
| E10 | Per-experiment boxplots | §8.5 synthesis | same | No |

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

E6: **tail-mean loss bar chart** comparing R1+R2 and R1-only flows
across the four budgets under the standard recipe, with the entropy
floor (~0.99) drawn as a horizontal band. The runners persist only
the last 100 training steps (`arch_metadata.loss_history_tail`), so
this is a converged-regime view, not a full trajectory. The story
the figure carries is "R1-only sits ~0.4 above the floor at all
budgets; R1+R2 sits at the floor" — which is exactly the
under-convergence claim in the manuscript.

E7: full-trajectory plot of the longer-recipe R1-only training run
dipping below the entropy floor (~0.5 vs floor 0.99). Because the
test fixture (`tests/ablation/test_trained_folding.py`) runs the
training in-process and keeps the full loss array, this figure is
regenerated on demand from the fixture (not loaded from a saved
trajectory file). The CLAUDE.md note about seed-fragility applies —
the figure builder should use a fixed seed that the fixture is
known to fold under.

E8: 5 methods × 4 experiments coverage_error_max on one log-y axis,
with the binomial-noise floor as a horizontal band. The CLAUDE.md
headline table rendered as a figure. **Scalar-only summary, no
seed-spread, no per-α detail.**

E9: **coverage_error_max** vs total parameter budget, per method,
per experiment. Uses coverage_error_max (not pivot RMSE) because
pivot_rmse is conditional on pivot-based procedures and would be
`None` for NLE/NRE/LF2I-BFF, silently dropping three methods.
CDSBI saturates earliest.

E10: 4-panel coverage_error_max boxplots, one per §8.x, 5 methods
grouped, all seeds × budgets pooled. **Shows seed × budget
distribution; this is the figure that exposes variance, which E8
elides.**

**On the apparent overlap between E2, E8, E10:** each does
something the others do not. E2 is per-α (the coverage *curve* at
d=1, with α on the x-axis) — only here can the reader see how
miscalibration depends on the nominal level. E8 is the scalar
headline (a single coverage_error_max per (method, experiment), on
log-y) — the "at a glance" figure for the synthesis section. E10
shows the *distribution* across seeds and budgets per experiment —
the variance check that justifies the mean ± std in the tables.
The spec deliberately keeps all three because their semantic
content does not overlap, only their data source.

## Reproducibility

**Manifest is the only place run-dirs are referenced.** Updating
which sweep a figure uses is a one-YAML-edit operation; figure
builder code stays sweep-agnostic.

**Tracked artifacts re-render byte-identically.** The render CLI
suppresses the timestamp metadata matplotlib embeds (PDF
`CreationDate`, PNG `Software` chunk), so re-running a figure whose
inputs and builder are unchanged produces identical bytes and no
spurious git diff on the committed `figures/` outputs.

**Outputs are git-tracked, with a size budget.** `figures/<id>.pdf` and
`.png` are checked in so the manuscript builds anywhere without
needing the run-dirs themselves (run-dirs stay gitignored). Target
**≤ 400 KB per PDF, ≤ 150 KB per PNG, ≤ 9 MB total under `figures/`**.
If a figure exceeds the per-file budget, first try reducing PNG DPI
(300 → 200) and decimating heavy scatters; if total `figures/` size
crosses the budget, the PNG gallery moves to a `figures/gallery/`
that is gitignored and the README links to GitHub-rendered PDFs
instead (no git-lfs — the manuscript only needs the PDFs, which
stay small).

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

**Backend pinning.** All tests set `matplotlib.use("Agg")` at module
import so they run headless under CI. The `cdsbi.mplstyle` file
declares `mathtext.fallback: cm` so the serif math choice falls back
gracefully when the configured font isn't installed.

**Panel-level unit tests** (`tests/unit/test_panels_<name>.py`).
- Feed each panel a synthetic mini-dataframe (10–20 rows).
- Assert the returned Axes has the **exact expected counts** of
  Lines / Patches / Texts for the data shape (the reviewer's
  stronger form — "nonzero file" is too weak).
- Assert the active style matches `cdsbi.mplstyle` (e.g., the
  configured palette is in use; spines obey the style).
- Assert the method-colour convention: when a panel plots N methods,
  the rendered Lines / Patches use the colours from
  `style.METHOD_STYLE` in the canonical order.
- Fast (< 1s per test).

**Figure-level integration tests** (`tests/integration/test_figures_smoke.py`).
- One test per catalogue entry.
- Call the builder with a synthetic mini-run-dir fixture (10 rows of
  parquet, or a tiny pickled checkpoint for the four
  checkpoint-needing figures) or its synthetic driver.
- Assert it writes a PDF and a PNG of nonzero size **and** that the
  resulting PDF has the expected number of figure-level Axes objects
  (e.g., E1's two-panel figure has 2 Axes). Doesn't validate visual
  correctness — that is the author's eyeballs' job — but does catch
  panel-composition regressions.

**Visual acceptance (mandatory, not "eyeballs optional").** Automated
tests verify structure, never whether a figure *reads well and lands
its message*. So every figure gets an explicit visual-acceptance step
before its commit is final, and the F0–F3 plans bake it in as a task
step:

1. Render the figure to PNG (`render --fig <id>`).
2. **Actually view it** — the executing agent is multimodal and
   `Read`s `figures/<id>.png` so the pixels enter context; a human
   executor opens it in an image viewer.
3. Judge against a checklist: (a) renders at all — not blank, no
   clipped/overlapping labels; (b) style applied — top/right spines
   absent, grid faint, fonts legible at column width; (c) math
   renders — `$\theta$`-style labels show real glyphs, not tofu
   boxes; (d) **message lands** — a naïve reader could state the
   figure's one-sentence takeaway from the picture alone, and in
   cross-method figures CDSBI is the visual protagonist with the
   colour convention honoured.
4. If it fails any check, iterate on the builder and re-render before
   committing; record the verdict in one line in the commit message.

This replaces the weaker "author inspects and approves" — the
inspection is a defined, repeatable gate, run for the F0 hello-world
and every F1–F3 figure.

## Decomposition — F0–F3 milestones

The implementation plan should ladder up, not try to render all 16
figures in one shot.

**F0 — Infrastructure (no figures yet).**

- `style.py`, `cdsbi.mplstyle`, `data_io/aggregates.py`,
  `data_io/checkpoints.py`, manifest schema + loader, render CLI,
  gallery generator.
- A "hello world" placeholder figure to exercise the full pipeline
  end-to-end, including its visual-acceptance step.
- Smoke-test fixtures (synthetic mini-run-dir parquet + model.pt).
- `.gitignore` negation so `figures/` artifacts are tracked despite
  the global `*.pdf` rule.

**F1 — Empirical primitives (panels).**

- Implement reusable Axes-level panels: `pit_histogram`,
  `coverage_curve`, `boxplot_per_method`, `jacobian_recovery_scatter`,
  `loss_trajectory_with_floor`, `cross_method_summary_log_y`,
  `position_table_as_axes`, plus any small helpers (e.g.,
  `noise_floor_band`).
- Each driven by synthetic data in its unit test; no sweep
  dependency yet.

**F2a — Empirical figures E1–E5 (per-section).**

- Compose panels into the per-experiment figures, one manuscript
  section at a time:
  - E1, E2 land with a §8.1 manuscript-integration commit.
  - E3 lands with §8.2.
  - E4 lands with §8.3.
  - E5 lands with §8.4.

**F2b — Empirical figures E6–E10 (ablation + synthesis).**

- E6 (tail-mean ablation bar chart) lands with the §8.4 ablation
  paragraph.
- E7 (longer-recipe folding trajectory) lands with the same commit
  as E6 if regen is straightforward, otherwise immediately after.
- E8, E9, E10 land together with the §8.5 synthesis section update
  (they share the same four-sweep data source and the same
  cross-method colour convention).

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
  trained R1-only flow's r(θ, T) surface needs the model.pt — the
  `data_io.checkpoints` path exists for this. If checkpoint loading
  is fiddly *or* if the trained surface doesn't visualise cleanly,
  the fallback is **an analytic 1D folded r constructed by hand**
  (e.g., r(θ, X) = θ − X + 0.4 sin(2X)). This is materially a
  different figure than the original — it shows the geometric
  mechanism, not a trained example — but it is an **acceptable
  outcome**: the conceptual point (Z(θ) > 1 → strict propriety
  violated) survives the substitution, and the manuscript caption
  can note the figure is illustrative.
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
- `figures/` total under the 9 MB budget; per-file budgets met.
- All four checkpoint-needing figures (C1, C3, E6, E7) render
  successfully from the `data_io.checkpoints` path; C3's fallback
  is documented in its module docstring if invoked.
