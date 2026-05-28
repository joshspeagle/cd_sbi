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

**Python codebase** (v0 + v1 + v2 + v3 landed; ~201 fast tests + 4 intensive replication tests + opt-in (R2) ablation suite):

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

## Current state of the repo

- `cd_sbi_v7.tex` — the manuscript (50 pages; §8.1 / §8.2 / §8.3 carry
  v0 / v1 / v2 sweep-averaged results). Source of truth.
- `cd_sbi.bib` — 45 BibTeX entries; manuscript uses natbib.
- `references/<bibkey>.md` — paper note per cited entry, built during round 2.
- `reviews/round{1,2,3}/` — per-round critic reports and audit trail.
- `docs/superpowers/specs/` — design specs.
- `docs/superpowers/plans/` — implementation plans (v0, v1, v2, v3 plans live here; v4 plan next).
- `src/cdsbi/` — Python package (v0 + v1 + v2 + v3 landed). Six core layers + `confidence_set/`, `experiments/`, `analysis/`, `reproducibility/`. v3 added `DoublyMonotoneUMNN`, `JointUMNNFlow`, `JointUMNN1DFlow`, `MLPConditioner`, `ReducedSimulator`.
- `configs/` — Hydra config groups (target / flow / conditioner / method / training / budget / experiment).
- `tests/` — `unit/`, `integration/`, `diagnostics/`, opt-in `intensive/` (4 replication tests: §8.1, §8.2, §8.3, §8.4) and opt-in `ablation/` (3 tests: safety-check, trained-folding, 1D mechanism).
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

**Next milestone:** v4 (SBI benchmark — Two Moons, SLCP, Gaussian
Mixture, etc.). After v4: v5 (§3.7 alt-loss), v6 (synthetic high-d),
v7 (real-data astronomy), v8 (image/sequence). See spec §12 for the
full roadmap.
