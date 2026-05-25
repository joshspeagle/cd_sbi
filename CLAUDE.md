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

## Current state of the repo

- `cd_sbi_v7.tex` — the manuscript, renamed from `cd_sbi_v6.tex` before
  round 1 of the multi-round vetting workflow. Round 1 (factual
  accuracy) is complete; the file has been edited in place. See
  `reviews/round1/` for the audit trail.
- `reviews/round1/` — round-1 critic reports (one per Part + cross-cutting)
  and the claim & evidence inventory with an integration log mapping each
  commit to its Part-level scope.
- `LICENSE`, `README.md`, `.gitignore` — repo setup.
- No code yet.

## Round-1 status (May 2026)

Round 1 of the three-round vetting workflow is **complete**. Eight
critic agents (one per Part + cross-cutting) flagged 88 items across
the manuscript (~39 ✓, ~43 ⚠, ~6 ✗); all were integrated into
`cd_sbi_v7.tex`. The one initially-escalated item — the §5.7.2 Bin(1, θ)
counterexample (C-5.7-counterex) — was resolved by a follow-up
math-skeptic agent that found a working Bin(3, θ) construction (the
intermediate Bin(2, θ) attempt also fails; see
`reviews/round1/r4_independence_check.md`). The agent's finding also
caught a sign-convention issue in the manuscript's (R4), which has been
corrected.

## Round-2 status (May 2026)

Round 2 of the vetting workflow is **complete**.

- `cd_sbi.bib` at the repo root holds 45 BibTeX entries (the 30 from the
  v6 References section, two papers cited inline but missing from v6
  (Lueckmann 2021, MAF 2017), and four added in round 2: Knothe 1957,
  Villani 2003, Lehmann-Romano 2005, and Stevens 1950).
- The manuscript is migrated to natbib (`\cite`/`\citet`/`\citep`).
  Build cycle is `pdflatex → bibtex → pdflatex → pdflatex`.
- `references/<bibkey>.md` contains a paper note for each cited entry,
  built by seven parallel lit-review agents (one per Part).
- `reviews/round2/` holds the citation inventory and the seven
  per-Part lit-review reports.
- Integrated attribution corrections: §1.1 Hermans-2022 scope (three
  families, not four); §7.1 UMNN (W&L use ELU+1 + Clenshaw-Curtis,
  not softplus + Gauss-Legendre — softplus is our substitution);
  §6.3 add Knothe 1957 to KR citation; §11.2 drop "entropic/" from
  Carlier-Galichon-Santambrogio characterization; §11.4 Wehenkel 2025
  mechanism is RoPE (OT-on-real-data), not generalized Bayes; §11.6
  Lueckmann 2021 doesn't measure coverage so removed from the
  sequential-miscalibration citation; PatelEtAl2023 venue upgraded
  to ICML 2024.
- Substantive citation gaps closed: Villani 2003 added at §4.2
  (1D monotone rearrangement); Lehmann-Romano 2005 at §5.1 (exp-family
  setup) and §7.3 (KS Kolmogorov distribution); Stevens 1950 alongside
  Lancaster 1961 at §5.7.1 (randomized PIT priority); CranmerEtAl2020
  / Fraser2011 / SinghEtAl2007 / XieSingh2013 promoted from \nocite
  to inline at §1.1–§1.2; GneitingRaftery2007 at §3.7 Class-4 CRPS
  dual form.

Round 3 (holistic review and cleanup) is the next phase; per the plan,
the user re-plans round 3's scope after round 2 lands.

## Round-3 status (May 2026)

Round 3 of the vetting workflow is **complete**. Focused on pedagogy
and accessibility for an audience that includes astronomers as well as
statisticians, with two failure modes targeted: undefined / under-
explained terms (especially around normalizing-flow and architectural
content), and "oracle-style" writing where conclusions are asserted in
friendly language but never motivated.

Workflow: seven per-Part discovery agents + one global agent did a
phase A+B survey (`reviews/round3/part{0..7}_discovery.md`); main
thread synthesized findings into a detailed plan
(`/home/joshspeagle/.claude/plans/round3-pedagogy.md`); per-Part edits
were applied in seven separate commits, each with a clean pdflatex
build; two fresh-reader agents (astronomer + statistician) did a
final cold-read (`reviews/round3/{astronomer,statistician}_review.md`)
and surfaced a final round of targeted fixes that landed in one
last commit.

What changed at the manuscript level:
- §1.5 Roadmap added; opening paragraphs added at each Part (II–VII).
- §2.2 "Regularity conditions at a glance" reference table covers the
  full R-zoo (R1–R4, R3_U, R1^auto, R2^auto) with first-defined
  locations.
- In-line "what is X" introductions for the load-bearing concepts
  (CD, pivot, calibration manifold, PIT, normalizing flow, change-of-
  variables Jacobian, UMNN, sufficient statistic, MLR, KR
  rearrangement).
- Each major theorem now has a "what we are about to prove" preamble
  (T-A, T-A*, T-C, T-C*, T-A-d, and the new labeled Theorem 3.2 on
  strict propriety of NF-MLE).
- §3.5 -log Z(θ) mechanism, §8.4 ablation, and §3.7 Class 1–5
  taxonomy rewritten as multi-step stories.
- §6 multivariate intro expanded with a 1D → multivariate bridge;
  worked d=2 example added in §6.1; Cholesky corollary reframed as
  example-then-theorem.
- §9 architecture-choice table now has in-cell glosses (not just
  section pointers); §10 reorganized around two axes with explicit
  "what CD-SBI buys you" comparisons.
- §11 open problems grouped by theme (theoretical extensions /
  robustness / scaling & practice / engineering); Status of claims
  table now category-tagged.
- §6.4 closes the load-bearing ρ-a.e.→every-θ_0 calibration lift with
  a short continuity-plus-dense-support argument.

End-of-round 3 PDF: 47 pages (from 35 at end of round 2). Build clean
via pdflatex + bibtex + pdflatex + pdflatex. The manuscript is in a
shape suitable for sharing with both astronomer and statistician
collaborators per the dual-reader fresh-reviewer agents' verdicts.

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
  flow) — §8.4 is the empirical reason.
- **Tone.** The draft uses astronomer-friendly bridges (e.g. §1.2). Keep
  that voice for new writing aimed at the same audience.

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
- LaTeX build artifacts (`*.aux`, `*.log`, `*.bbl`, `*.synctex.gz`, …)
  and common Python noise are gitignored; the PDF is not committed.
- Commit / PR style is not yet fixed — check recent history or ask before
  assuming.

## Possible future directions (not yet committed)

Flagged in conversation but not promised:

- Reference implementation of the §8 experiments.
- Higher-`d` scaling studies (open problem §11.5).
- Sequential / amortized variants (open problem §11.6).
- Misspecification (open problem §11.4).
