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
pdflatex cd_sbi_v7 && bibtex cd_sbi_v7 && pdflatex cd_sbi_v7 && pdflatex cd_sbi_v7
```

**Python codebase** (available once Task 1 of the v0 plan lands):

```bash
pip install -e ".[dev]"               # install cdsbi package + dev deps
pytest                                # fast tests (unit + integration + diagnostics)
pytest -m intensive                   # opt-in full-budget replication (minutes)
python -m cdsbi.experiments.run experiment=8_1_replication seed=0   # single run
python -m cdsbi.experiments.run -m experiment=8_1_baseline_sweep    # full sweep
```

## Current state of the repo

- `cd_sbi_v7.tex` — the manuscript (47 pages, post-round-3). Source of truth.
- `cd_sbi.bib` — 45 BibTeX entries; manuscript uses natbib.
- `references/<bibkey>.md` — paper note per cited entry, built during round 2.
- `reviews/round{1,2,3}/` — per-round critic reports and audit trail.
- `docs/superpowers/specs/` — design specs (v0 infrastructure spec lives here).
- `docs/superpowers/plans/` — implementation plans (v0 implementation plan lives here).
- `src/cdsbi/` — Python package (planned; v0 implementation in progress).
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

The current active artifacts:
- `docs/superpowers/specs/2026-05-25-cd-sbi-experiment-infrastructure-design.md`
- `docs/superpowers/plans/2026-05-25-cd-sbi-v0-experiment-infrastructure.md`

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

## v0 codebase status

The v0 experiment infrastructure (replicating manuscript §8.1 + matched-
budget baselines against NPE / NLE / NRE / LF2I) is **fully specced and
planned**, awaiting implementation:

- Spec: `docs/superpowers/specs/2026-05-25-cd-sbi-experiment-infrastructure-design.md`
  (revised twice via dual self-review)
- Plan: `docs/superpowers/plans/2026-05-25-cd-sbi-v0-experiment-infrastructure.md`
  (25 TDD-style tasks, every step has full code)

Once v0 lands, the natural next milestones (per the spec's roadmap) are
v1 (§8.2 multivariate Σ=I), v2 (§8.3 correlated Σ), v3 (§8.4 exponential
rate + (R2) ablation), then real-data targets (SBI benchmark suite,
astronomy inference, image observations).
