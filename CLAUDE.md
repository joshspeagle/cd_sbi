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

Round 1 of the three-round vetting workflow is complete pending user
sign-off. Eight critic agents (one per Part + cross-cutting) flagged 88
items across the manuscript (~39 ✓, ~43 ⚠, ~6 ✗); all were integrated
into `cd_sbi_v7.tex` except one **escalation** flagged with an inline
`% TODO(round1-escalation)` marker:

- **Bin(1, θ) counterexample in §5.7.2 (C-5.7-counterex)** does not
  actually separate (R4) from (R3_U). A Bin(2, θ) replacement is the
  recommended fix; the v6 construction is preserved pending user input.

Round 2 (literature & citations + BibTeX migration) is the next phase;
see `/home/joshspeagle/.claude/plans/our-first-task-will-mellow-harbor.md`
for the workflow plan.

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
