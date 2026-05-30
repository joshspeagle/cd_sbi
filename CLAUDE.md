# CLAUDE.md

Notes for Claude Code agents working in this repository.

## Project at a glance

**CD-SBI** is a research project on an SBI framework that targets a
**calibrated confidence distribution** (CD) — a frequentist alternative to
the Bayesian posterior / likelihood / ratio targets of mainstream SBI
methods (NPE, NLE, NRE). The current source of truth is the working draft
`cd_sbi_v7.tex`.

Core idea, in brief:

- Learn a pivot `r(θ; X)` such that `r(θ₀; X) | θ₀ ~ N(0, I_d)` for every
  true `θ₀` — pointwise frequentist calibration, with no prior.
- Train via NF-MLE (identical to the SNL loss).
- Enforce monotonicity in `θ` **and** in `X` *architecturally* (UMNN
  integrand in 1D; autoregressive triangular flow in `d > 1`). Autograd-
  only Jacobians are documented as a failure mode (§8.4).
- The induced CD is `H_r(θ; X) = Φ_d(r(θ; X))`, and confidence sets are
  `C_α(X_obs) = {θ : ‖r(θ; X_obs)‖² ≤ χ²_{d,α}}`.

The eventual goal of the codebase is infrastructure to train SBI models
that produce UMP(U)-style confidence distributions in this framework,
scaling beyond the toy validation experiments currently in the draft.

## Commands

**Build the manuscript** (clean BibTeX cycle):

```bash
/usr/bin/pdflatex cd_sbi_v7 && /usr/bin/bibtex cd_sbi_v7 \
  && /usr/bin/pdflatex cd_sbi_v7 && /usr/bin/pdflatex cd_sbi_v7
```

Use the absolute path: the system TeX Live install at `/usr/bin/`
is the working one. The conda `pdflatex` at
`/home/joshspeagle/miniconda3/bin/pdflatex` (first on PATH) is broken
on this machine — its perl-based `mktexfmt` can't find
`mktexlsr.pl` and bails before opening `pdflatex.fmt`.

**Python codebase** (v0 + v1 + v2 + v3 + unknown-(μ,σ²) Stage A + Stage B bake-off landed; ~318 fast tests + intensive replication/verdict tests + opt-in (R2) ablation suite):

```bash
pip install -e ".[dev]"               # install cdsbi package + dev deps
pytest                                # fast tests (unit + integration + diagnostics)
pytest -m intensive                   # opt-in full-budget replication (minutes)
pytest -m ablation                    # opt-in (R2) ablation tests (mechanism + trained folding)
python -m cdsbi.experiments.run experiment=8_1_replication seed=0   # single run
python -m cdsbi.experiments.run -m experiment=8_3_baseline_sweep \
    method=cd_sbi,npe,nle,nre,lf2i_bff \
    budget=small,medium,large,xlarge \
    seed=0,1,2,3,4 \
    training.fresh_batch=false        # 100-run cross-method sweep (≈ 5 hr)
```

**Render the figure suite** (16 figures: 10 empirical E1–E10 + 6 conceptual C1–C6):

```bash
python -m cdsbi.analysis.figures.render --all              # render all figures → figures/<id>.{pdf,png}
python -m cdsbi.analysis.figures.render --all --section 8.4  # filter (NOTE: --section requires --all)
python -m cdsbi.analysis.figures.render --gallery          # regenerate figures/README.md gallery
python tools/regen_figure_data.py                          # re-run CDSBI figure-data sources (GPU; needs PYTHONPATH=.)
```

## Current state of the repo

- `cd_sbi_v7.tex` — the manuscript (50 pages; §8.1 / §8.2 / §8.3 carry
  v0 / v1 / v2 sweep-averaged results). Source of truth.
- `cd_sbi.bib` — 45 BibTeX entries; manuscript uses natbib.
- `references/<bibkey>.md` — paper note per cited entry, built during round 2.
- `reviews/round{1,2,3}/` — per-round critic reports and audit trail.
- `docs/superpowers/specs/` — design specs.
- `docs/superpowers/plans/` — implementation plans (v0, v1, v2, v3 plans live here; v4 plan next).
- `src/cdsbi/` — Python package (v0 + v1 + v2 + v3 + (μ,σ²) Stage A landed). Six core layers + `confidence_set/`, `experiments/`, `analysis/`, `reproducibility/`. v3 added `DoublyMonotoneUMNN`, `JointUMNNFlow`, `JointUMNN1DFlow`, `MLPConditioner`, `ReducedSimulator`. (μ,σ²) Stage A added `NormalUnknownMeanVar`, `SingleIndexMonotoneFlow`, `SufficientStatConditioner`, `MarginalCDRecovery`. Stage B added `DeepSetsConditioner`, `EnergyCalibrationLoss`+`EnergyCDSBIRunner`, `AffineCouplingBijection`+`InvertibleSummaryConditioner`+`ExactDensityCDSBIRunner`, `SufficiencyRecovery`+`FloorIntegrity`.
- `configs/` — Hydra config groups (target / flow / conditioner / method / training / budget / experiment).
- `tests/` — `unit/`, `integration/`, `diagnostics/`, opt-in `intensive/` (4 replication tests: §8.1, §8.2, §8.3, §8.4) and opt-in `ablation/` (3 tests: safety-check, trained-folding, 1D mechanism).
- `src/cdsbi/analysis/figures/` — visualization suite (F0 infra + F1 panels + F2 empirical + F3 conceptual). Manifest-driven (`configs/figures/manifest.yaml`); builders are pure `render(spec) -> Figure` composing F1 panels; the render CLI owns all disk IO. matplotlib/Agg via `style.apply_style()`.
- `figures/` — 16 tracked figure PDFs/PNGs + auto-generated README gallery (the manuscript's `\includegraphics`). `.gitignore` negates the global `*.pdf` rule for this dir.
- `LICENSE`, `README.md`, `.gitignore` — repo setup.

## Specs and plans

The repo uses a brainstorm → spec → plan → implementation workflow
(superpowers skill family). Two artifacts live under `docs/superpowers/`:

- **Specs** (`docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`) — the
  validated design output of a brainstorming session. Reviewed by both
  the author and an independent reviewer agent before being handed off.
- **Plans** (`docs/superpowers/plans/YYYY-MM-DD-<feature>.md`) — bite-sized
  TDD-style implementation plans, derived from an approved spec. Each step
  contains the actual code to write and the test that drives it.

Active artifacts:
- `docs/superpowers/specs/2026-05-25-cd-sbi-experiment-infrastructure-design.md` (covers v0 → v8 roadmap)
- `docs/superpowers/plans/2026-05-25-cd-sbi-v0-experiment-infrastructure.md`
- `docs/superpowers/plans/2026-05-26-cd-sbi-v1-multivariate-iid.md`
- `docs/superpowers/plans/2026-05-27-cd-sbi-v2-correlated-sigma.md`
- `docs/superpowers/plans/2026-05-27-cd-sbi-v3-r2-ablation.md`
- `docs/superpowers/specs/2026-05-28-cd-sbi-visualizations-design.md` + `docs/superpowers/plans/2026-05-28-cd-sbi-viz-{f0-infrastructure,f1-panels,f2-figures,f3-conceptual}.md` (the 16-figure visualization suite)

## Manuscript status

The three-round vetting workflow on the manuscript is **complete** as of
May 2026. Round 1 was factual accuracy (8 critic agents, 88 items
integrated); round 2 was citation hygiene + literature review (natbib
migration, 45 BibTeX entries, 7 per-Part lit-review agents); round 3 was
pedagogy + accessibility for a mixed astronomer + statistician audience
(7 discovery agents + 2 fresh-reader agents). See `reviews/round{1,2,3}/`
for the full audit trail; commit history reflects each round's
integration.

The manuscript is in a shape suitable for sharing with collaborators
per the round-3 fresh-reader verdicts.

## Conceptual map of the draft

- **Parts I–II — Framework + loss.** Pivot `r`, calibration manifold `M`,
  NF-MLE objective (§3); regularity (R1) monotone-in-θ, (R2) monotone-in-X.
  Strict-propriety argument (§3.2) + a five-class taxonomy of alternative
  objectives and why NF-MLE is the one used (§3.7).
- **Part III — 1D theory.** Theorem A (C¹ uniqueness, location-normal →
  Schweder–Hjort UMPU CD); Theorem A\* (Lipschitz, under extra condition
  (R3)); Theorem C (lift to regular 1-parameter exponential families);
  Theorem C\* (discrete via Lancaster's randomized PIT, requires (R4)).
- **Part IV — Multivariate.** Triangular autoregressive flows; Theorem A-d
  gives uniqueness within that class as the Knothe–Rosenblatt
  rearrangement. Corollary: for Gaussian `X ~ N(θ, Σ)`,
  `r^KR(θ; X) = L⁻¹(θ − X)` with `LL^T = Σ`.
- **Part V — Experiments.** §8.1 1D loc-normal; §8.2 2D loc, `Σ = I`; §8.3
  2D loc, `Σ` correlated; §8.4 exponential rate + autograd-Jacobian
  ablation (the cautionary tale for (R2)).
- **Parts VI–VII — Practice + open problems.** Implementation recipe
  (§9); position vs other SBI methods (§10); open problems (§11):
  uniqueness beyond the autoregressive class, KR ordering selection,
  whether (R3) follows from (R1)+Lipschitz, misspecification, higher-d
  scaling, sequential variants, discrete without scalar sufficient
  statistic, efficient `C_α` computation.

## Working principles for this repo

- **Don't trust the draft uncritically.** Verifying it is a workstream.
- **`ρ` is not a prior.** Coverage statements are frequentist and pointwise
  in `θ₀`; the proposal is a sampling device, not a Bayesian object.
- **Monotonicity is architectural, not penalty-based.** Any implementation
  work should enforce (R1), (R2) by construction (UMNN / triangular
  flow) — §8.4 is the empirical reason. The v0 design encodes this as
  `Flow.monotonicity_guarantees`, checked against `Loss.required_guarantees`
  before training (see `docs/superpowers/specs/...`).
- **Tone.** The draft uses astronomer-friendly bridges (e.g. §1.2). Keep
  that voice for new writing aimed at the same audience.
- **Brainstorm → spec → plan workflow.** Non-trivial code work goes
  through the `superpowers:brainstorming` → spec → `superpowers:writing-plans`
  → execution pipeline. Specs and plans land in `docs/superpowers/`.

## Notation cheat sheet

- `Φ`, `Φ_d`: standard normal CDF / product CDF on `ℝ^d`.
- `r*`: canonical / true pivot (closed-form when it exists).
- `H_r = Φ_d ∘ r`: induced CD.
- `M`: calibration manifold (the set of `r` with the right pushforward).
- `r^KR`: Knothe–Rosenblatt rearrangement.
- `T(X)`: sufficient statistic (in exponential-family discussion).
- `C_α`: level-α confidence set.

## Repo conventions

- Default branch: `main`.
- LaTeX build artifacts (`*.aux`, `*.log`, `*.bbl`, `*.synctex.gz`, …),
  common Python noise (`__pycache__`, `*.egg-info`, `.pytest_cache`),
  and `outputs/` (run-dir root) are gitignored; the PDF is not committed.
- Commit messages: concise subject line, optional body explaining *why*
  (not *what*). Claude-assisted commits include a
  `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
  trailer. No `--no-verify`, no `--amend` on existing commits unless
  explicitly requested.

## Codebase status (v0 + v1 + v2 + v3 done)

All four phases are **implemented, test-passing, and reflected in the
manuscript**. ~201 fast tests pass in ~90s; 4 intensive replication tests
clear their tolerance bands (§8.1 ~5 min, §8.2 ~16 min, §8.3 ~5 min,
§8.4 ~5 min wall on GPU). The cross-method sweep at each section runs
5 methods × 4 budgets × 5 seeds = 100 runs. v3 also adds a separate
(R2) ablation sweep (R1-only `JointUMNNFlow` vs R1+R2 `DoublyMonotoneUMNN`).

**Headline empirical result across v0 → v1 → v2 → v3 (coverage_error_max,
mean ± std; lower = better; noise floor ~0.02):**

| Method | §8.1 (1D) | §8.2 (2D iid) | §8.3 (2D corr) | §8.4 (exp rate, on T) |
|---|---|---|---|---|
| **CDSBI** | **0.025** | **0.025** | **0.025** | **0.028–0.034** ← at noise floor across all four |
| LF2I-BFF | 0.06–0.07 | 0.11–0.12 | 0.11–0.14 | 0.06–0.09 (3–4× floor) |
| NRE | 0.08–0.12 | 0.14–0.18 | 0.14–0.19 | 0.12–0.20 (6–10× floor) |
| NLE | 0.025 (tied) | 0.07–0.09 | 0.10–0.12 | 0.21–0.25 (10–13× floor) |
| NPE | 0.06–0.07 | 0.05–0.10 | 0.05–0.09 | 0.24–0.25 (12–13× floor) |

The CDSBI → NLE gap widens monotonically as the data distribution gets
more non-trivial — theory predicted this; v0 → v3 confirmed it empirically.
§8.4 also flips the ordering of the non-CDSBI methods: LF2I-BFF moves
ahead of NLE because its calibration head fits the χ²-shaped test
statistic, whereas the Bayesian methods over-concentrate on a small
region of the prior.

**Per-phase summary:**

- **v0 (§8.1).** UMNN, AdditiveFlow1D, MAFAdapter, 5 method runners
  returning ConfidenceProcedure subtypes, 1D root-finder, 4 diagnostics,
  Hydra/run-dir/seeding/device plumbing. Final v0 close-out moved the
  default training recipe to `adamw_cosine_warmup` with
  `fresh_batch=false` (finite-sample regime as the research-realistic
  default).
- **v1 (§8.2).** `TriangularAdditiveFlow`, multivariate `confidence_set`
  via ray-bisection from a found center, `JointMahalanobis` diagnostic
  (the inferentially-primary 2D check), per-coordinate PIT, `paper_table_8_2`.
  In-execution fix-pack F1–F7 added cross-procedure caching (`_BatchCache`
  on `id(x_obs_batch)`) and dispatcher-level X|θ_0 sharing; LF2I-BFF
  quantile-head schedule retuned after the §8.1 sweep exposed a
  U-shape budget-extreme artifact.
- **v2 (§8.3).** `LocationGaussian2D_corr` simulator with closed-form
  `r_star_jacobian()`; new `JacobianRecovery` diagnostic empirically
  validates the Knothe–Rosenblatt uniqueness claim (Theorem A-d) —
  trained `E[∂r/∂θ]` matches `L⁻¹` to ~3% max-element residual across
  5 seeds. No new flow code; v1's TriangularAdditiveFlow covers the
  correlated case unchanged. `paper_table_8_3` adds the
  `jacobian_max_residual` aggregate column. Mid-execution perf fix
  dropped `PosteriorBasedProcedure` default `n_samples` 10000 → 2000
  (5× NPE wall speedup; coverage answer unchanged).
- **v3 (§8.4).** Non-additive exponential-rate target with closed-form
  truth pivot `r*(θ, T) = Φ⁻¹(F_{χ²_{10}}(2θT))` (sufficient statistic
  `T = Σ X_i`, `n = 5`). New: `ExponentialRate` simulator,
  `MLPConditioner` (frozen-sum X→T with `½ log 5` Jacobian — the
  framework's first non-trivial `log_det_jac_input_contribution`),
  `ReducedSimulator` (wraps non-CDSBI methods so they receive `(θ, T)`
  for apples-to-apples comparison), `DoublyMonotoneUMNN` (§6.1 form 2 —
  main flow, `{R1, R2}` architectural guarantees), `JointUMNNFlow`
  (R1-only ablation flow), `JointUMNN1DFlow` (1D mechanism analog).
  New `tests/ablation/` test category with `ablation` pytest marker:
  safety-check (mismatched guarantees raise `MonotonicityMismatchError`
  unless `allow_ablation=True`), trained-folding (R1-only longer-recipe
  final loss falls below the entropy floor — the §3.5 mechanism failure
  captured as a regression), 1D direct-construction mechanism. The
  doubly-monotone arm reaches the entropy floor (final NF-MLE loss
  0.985 ± 0.027 vs floor 0.99) across all budgets; the R1-only arm
  under-converges to ~1.40 at the standard recipe.

**Known caveats (still open; revisit at the relevant milestone):**

- **NPE d > 1 credible region uses empirical Mahalanobis**, not a
  true KDE-based HPD. Analytic HPD for Gaussian posteriors (exact under
  uniform priors, which is what §8.2/§8.3 have), approximation otherwise.
  Footnoted in §8.2 and §8.3 manuscript tables. Revisit when a
  non-Gaussian-posterior target lands (v4+).
- **SetSize d > 1 uses set diameter as a width proxy**, not true
  ellipsoid volume. Adequate for paper-table sanity at d = 2; would
  need proper volume estimation at higher d or for inferential use.
- **Coverage diagnostic uses a fixed 5-point θ_0 grid.** Works at
  d ≤ 2; v6 high-d will need a θ_0-grid generation strategy.
- **(R2) ablation landed in v3** — `tests/ablation/test_trained_folding.py`
  demonstrates the §3.5 mechanism failure (final loss drops below the
  entropy floor under a longer training recipe; under-converges at the
  standard recipe). Seed-fragile by design — see commit f1d17a7 notes.
- **CDSBI v3 flow dispatch trap.** `configs/method/cd_sbi.yaml` hardcodes
  `flow: additive_umnn`, so an experiment's `override /flow: doubly_monotone`
  does NOT take for `cd_sbi` unless you also pass `method.flow=doubly_monotone`
  (the `_build_flow` guard, commit `cb1e08e`). A single §8.4 replication needs
  `flow=doubly_monotone method.flow=doubly_monotone` and NO redundant top-level
  `method=cd_sbi`. Verify `model.pt`'s `arch_metadata.flow_class` after a run.
- **`model.pt` has no `state_dict`** — only `{arch_metadata (incl.
  `loss_history_tail` = last 100 steps), final_loss}`. A trained flow cannot be
  reloaded; recompute requires re-training. (Why F3 conceptual figures use
  closed-form/analytic drivers, not trained checkpoints.)
- **§8.x sweep data is split across timestamp dirs** (method-specific re-runs).
  §8.4 CD-SBI/`doubly_monotone` lives in
  `outputs/8_4_baseline_sweep/2026-05-27_20-20-16` (5 methods);
  `2026-05-28_12-05-31` is the 4-method on-T re-run. Aggregate figures use
  `analysis.figures.data_io.figure_data.load_sweep()` to dedup
  `(method, budget, seed)` across all timestamps of a sweep.

**Manuscript-to-code mapping:**
- §8.1 numbers: `outputs/8_1_baseline_sweep/2026-05-27_00-57-31/` +
  `outputs/8_1_baseline_sweep/2026-05-27_07-50-18/` (LF2I-BFF retune).
- §8.2 numbers: `outputs/8_2_baseline_sweep/2026-05-27_04-36-58/`
  (non-NPE non-LF2I) + `2026-05-27_05-32-50/` (NPE re-run) +
  `2026-05-27_08-00-09/` (LF2I-BFF retune).
- §8.3 numbers: `outputs/8_3_baseline_sweep/2026-05-27_11-18-28/`
  (non-NPE) + `2026-05-27_16-08-27/` (NPE re-run with `n_samples=2000`).
- §8.4 numbers: `outputs/8_4_baseline_sweep/2026-05-28_12-05-31/`
  (apples-to-apples 5-method × 4-budget × 5-seed sweep on `T`) +
  `outputs/8_4_ablation/2026-05-27_23-55-59/` (R1+R2 vs R1-only sweep).

## Unknown-(μ, σ²) Gaussian — Stage A + Stage B bake-off landed (M0–M3.2′)

A separate research track (the first target with a **scale/nuisance**
parameter): `NormalUnknownMeanVar` (θ = (log σ, μ), `n_iid=10` replicates per
observation X ∈ ℝ¹⁰, `d_theta=2`). Specs:
`docs/superpowers/specs/2026-05-29-cd-sbi-unknown-mean-variance-design.md`
(Stage A) + `…-stage-b-device-bakeoff-design.md` (Stage B, co-adaptation
reframing). Plans: `docs/superpowers/plans/2026-05-29-cd-sbi-{mu-sigma-m0,
mu-sigma-m1,m3-0,m3-1p,m3-2p}-*.md` (the I-B plan `…-m3-1-arm-ib-…` is a
documented contrast, NOT executed as a calibrating arm).

- **M0 (Stage-A core).** Key finding: a **doubly-monotone** flow (∂r/∂θ>0 AND
  ∂r/∂feat>0) **cannot** represent a scale parameter — `r_σ` must increase in
  θ(log σ) but decrease in the data feature (both enter via `s²/σ²`). Fixed by
  the new **`SingleIndexMonotoneFlow`** (`single_index_monotone.py`): per-coord
  `r_k = G_k(s_θk·softplus(p_θk(ctx))·θ_k + s_fk·softplus(p_fk(ctx))·feat_k +
  off_k(ctx); ctx)`, `G_k` a monotone-increasing UMNN of the index → independent,
  globally-guaranteed R1/R2 signs (no `θ_ref` restriction), non-additive. Signs
  injected from the simulator at wire-time (`theta_signs`/`feat_signs`). Oracle
  `SufficientStatConditioner` → `(log s², X̄)`; increasing-θ pivot convention
  `r*_σ=Φ⁻¹(1−F_{χ²}((n−1)s²/σ²))`, `r*_μ=√n(μ−X̄)/σ`. Recovery smoke RMSE 0.10.
- **M1 (Stage-A diagnostics).** New **`MarginalCDRecovery`** diagnostic: σ²-CD
  reads off `Φ(r_σ)` (direct χ²); μ-CD requires **marginalizing the σ nuisance**
  out of the joint confidence density (finite-difference integration over log σ)
  → recovers the **Student-`t_{n−1}`** CD. Primary metric = KS of each PIT vs U;
  secondary = 95th-pct per-X residual vs the analytic CD. 2-D 3×3 product
  coverage θ₀-grid; `paper_table_mu_sigma`. Intensive 5-seed replication
  (`experiment=mu_sigma_replication`): pivot_rmse 0.042, coverage 0.026, σ²/μ KS
  0.020, μ-resid p95 0.017, final_loss 0.904 at the entropy floor (H=0.921),
  JointMahalanobis 5/5.
- **Flow dispatch:** `mu_sigma_replication.yaml` sets `method.flow=
  single_index_monotone` itself (the cb1e08e guard) so the experiment is
  self-contained.
- **Stage B — learned summary device bake-off (M2→M3.2′).** Question: can a
  *learned* `s_φ(X):ℝ¹⁰→ℝ²` calibrate in the pivot machinery? **σ (scale) is the
  hard direction; the objective decides if it survives** (bake-off, judged on the
  M3.0 harness — `SufficiencyRecovery` Spearman + coverage/marginal-CD KS +
  `FloorIntegrity`):
  - **M2 naive end-to-end NF-MLE → CHEATS** (`DeepSetsConditioner`): collapses
    σ-info, loss −5.3 below floor 0.92. Captured as `tests/integration/
    test_mu_sigma_stage_b_smoke.py` (regression).
  - **Arm I-B (two-stage freeze, predict-θ) → WARPS** (`pretrain_summary` +
    `TwoStageCDSBIRunner`, plan only): σ² recovered (0.97) but shrinkage → fixed
    affine pivot miscalibrates σ. Documented contrast, not executed.
  - **Arm II-A (end-to-end `EnergyCalibrationLoss`, grouped-by-θ₀;
    `EnergyCDSBIRunner`) → COLLAPSES σ:** energy score hits its floor (1.83≈1.77)
    with σ²-Spearman **0.014**. KEY FINDING: **calibration ≠ informativeness** —
    per-θ₀ marginal calibration is satisfiable by a μ-only summary.
  - **Arm I-A (invertible exact density; `AffineCouplingBijection` +
    `InvertibleSummaryConditioner` + `ExactDensityCDSBIRunner`) → RECOVERS σ:**
    σ²-Spearman **0.980**, coverage KS ≤ 0.077, loss 14.47 > H(X|θ)=13.66 (no
    cheat). **Verdict: information-preservation (exact change-of-variables) is
    required; calibration-only collapses scale.** New harness:
    `SufficiencyRecovery`, arm-aware `FloorIntegrity` (loss→floor map:
    NFMLE→entropy_lower_bound, ExactDensity→data_entropy_lower_bound),
    `procedure.encode_fn` + `simulator.{oracle_summary,data_entropy_lower_bound}`.
  - **`DeepSetsConditioner.standardize`** flag (default True; off for the energy
    arm). `cd_sbi` method configs select the loss via `method.loss ∈
    {nfmle,energy,exact_density}` (run.py `_build_method`).

## Bivariate-normal unknown-(μ, Σ) — d=5 Stage-A landed (N1 + N2)

The next generalization: `NormalBivariateUnknownCov` (θ = (ℓ₁₁, ℓ₂₂, L₂₁, μ₁,
μ₂) in **log-Cholesky** coords, `Σ=LLᵀ`, `n_iid=10`, X ∈ ℝ²⁰, `d_theta=5`,
`p=2`). Branch `feat/musigma-m3-verdict`. Spec
`docs/superpowers/specs/2026-05-30-cd-sbi-bivariate-normal-mu-cov-design.md`;
plans `…/plans/2026-05-30-cd-sbi-mu-cov-{n1-stage-a-core,n2-diagnostics-
replication}.md`; **verdict `…/specs/2026-05-30-cd-sbi-mu-cov-n2-verdict.md`**.

- **The enabler — Wishart Bartlett decomposition** (multivariate analog of the
  1-D χ²/Student-t): scatter `A=Σ(Xᵢ−X̄)(Xᵢ−X̄)ᵀ ~ Wishart₂(n−1,Σ)`, ⫫ X̄ (Basu);
  with `A=DDᵀ`, `Σ=CCᵀ` (Cholesky), the factor `T=C⁻¹D` has independent entries
  `T₁₁²~χ²_{n−1}`, `T₂₂²~χ²_{n−2}`, `T₂₁~N(0,1)`. Inherently triangular → drops
  into the existing **`SingleIndexMonotoneFlow` with NO architecture change**.
  Closed-form `r*` (Bartlett): `r₁=Φ⁻¹(1−F_{χ²_{n−1}}(T₁₁²))`,
  `r₂=Φ⁻¹(1−F_{χ²_{n−2}}(T₂₂²))`, `r₃=T₂₁`, `r₄=√n(μ₁−X̄₁)/C₁₁`,
  `r₅=√n(−(L₂₁/(C₁₁C₂₂))(μ₁−X̄₁)+(μ₂−X̄₂)/C₂₂)` — validated N(0,I₅) at truth
  (KS≤0.011). KR ordering diagonals→off-diag→mean; `theta_signs=(+,+,−,+,+)`,
  `feat_signs=(−,−,+,−,−)`; features `(log D₁₁, log D₂₂, D₂₁, X̄₁, X̄₂)`.
- **N1 (Stage-A core).** `normal_bivariate_unknown_cov.py` (simulator + `r_star`
  + `oracle_summary` + `entropy_lower_bound`≈2.03 + `data_entropy_lower_bound` +
  `analytic_marginal_cd_pit`), `BartlettSummaryConditioner`,
  `flows/invert.py::autoregressive_invert` (per-coord bisection, fwd∘inv 1e-6).
  Recovery smoke RMSE 0.10.
- **N2 (Stage-A diagnostics + replication).** `MultivariateMarginalCDRecovery`
  (3 covariance direct PITs + **joint Hotelling-T² μ-marginal** = multivariate
  analog of the Student-t check, recovered via `autoregressive_invert`),
  `PivotBasedProcedure.flow` (set by `CDSBIRunner.fit`), 16-pt LHS coverage grid,
  `paper_table_mu_cov`. **Verdict: the framework generalizes to d=5** — 4/5 pivot
  coords calibrate to the noise floor; aggregate covariance χ² / joint Hotelling /
  joint Mahalanobis χ²₅ / entropy-floor all hold.
- **Documented limit (μ₂).** The doubly-cross-coupled mean coord is **mildly**
  miscalibrated at extreme θ₀ (per-coord PIT KS up to ~0.10; std ~0.9–1.3;
  central coverage fine) — a **ctx-MLP expressivity limit on the affine index
  z₅**, NOT finite-sample / convergence / capacity / G-curvature / w-asymmetry
  (all ruled out; identity-G trades scale-for-shape, no clean win). Accepted and
  documented; `test_replicate_mu_cov.py` pins it as a regression (KS<0.16),
  precedent `test_trained_folding.py`. RMSE-vs-`r*` is only a loose recovery
  sanity (`r*` is one specific calibrated pivot; calibration only needs M).

- **N3 (Stage-B learned summary) — FINDINGS milestone, not a build.** Verdict
  `…/specs/2026-05-30-cd-sbi-mu-cov-n3-stage-b-verdict.md`; evidence prototypes
  `…/evidence/2026-05-30-n3-stage-b-summary/`. The information-preserving I-A arm
  (the 1-D winner) does **NOT** scale to the 2-D covariance: generic *learned*
  invertible summaries route the linear means but **not the quadratic (co)variance**
  sufficient stats (7 variants; best = 4/5, comp-2 variance A₂₂ never routes;
  canonical corr `[.999 .999 .958 .919 .16]`). **Adding flexibility makes it worse**
  (Glow 1×1 mixing, permutation-equivariance, asinh all → lazier 2/5). Calibration
  passes throughout (χ²₅ ≈0.03) → *valid but inefficient* (calibration≠sufficiency,
  sharpest yet). A bespoke **Helmert+polar** structure-informed summary recovers all
  5 + calibrates but only by hard-coding the Bartlett decomposition — **existence
  proof, not a method** (not productionised). **LF2I-BFF baseline at d=5 scored
  coverage_err 0.20–0.39 from raw X — confirmed an artifact of OUR implementation, NOT
  LF2I** (recipe research pass + code audit, 2026-05-30): `lf2i_bff.py:71–87,114`
  estimates the BFF marginal with a **fixed N=64 sample set, frozen + blind to d** (d=1
  exact linspace; d=2 ≈8/axis degraded-but-real; d=5 ≈2.3/axis = noise). The recipe
  (Eq. 10) needs **no grid** — the marginal is `E_{θ~π}` (MC over proposal draws),
  calibration is grid-free quantile regression. **§8.1–8.4 numbers NOT invalidated**
  (d=1 exact; d=2 genuine N=64 measurement). **Fix LANDED** (`lf2i_bff.py`): BFF
  marginal now MC over true-prior draws (`simulator.sample`), d-aware `max(marginal_n,
  128·d)`, config `marginal_grid_n→marginal_n` (alias kept), +3 TDD regression tests.
  **Fair d=5 LF2I-BFF (marginal_n=2048):** coverage_err 0.091 (3-pt) / 0.189 (LHS) vs
  0.393/0.201 unfixed — central coverage repaired; residual ~0.19 at extreme θ₀ is a
  classifier/quantile-head budget matter, ~7× the oracle. Stage-A (oracle Bartlett)
  stands; Stage-B learned-summary is an **open problem**.

**Strategic reframe (user, 2026-05-30 — [[scalable-neural-copula-strategy]]):** bespoke
per-problem summaries are dead ends; the target must **scale to high-d & arbitrary
distributions**, framed as a **neural copula** (the pivot `r(θ;X)~N(0,I)` IS a
normal-scores/Gaussian-copula transform). Likely the **fixed-dim summary bottleneck is
itself the unscalable step**. A dedicated broad strategy brainstorm is **deferred until
the current experiments are done** (user's call).

**Next milestone:** the scalable neural-copula strategy discussion (deferred), then the
v-track roadmap: v4 (SBI benchmark — Two Moons, SLCP, Gaussian Mixture), v5 (§3.7
alt-loss — the II-A finding feeds this), v6 (synthetic high-d), v7 (real-data
astronomy), v8 (image/sequence). Open threads: whether a power/sharpness term rescues
II-A; a richer ctx-conditioned index for cross-coupled coords (the μ₂ refinement).
