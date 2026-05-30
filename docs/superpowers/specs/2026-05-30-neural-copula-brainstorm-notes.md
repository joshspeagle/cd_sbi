# Neural-copula strategy — brainstorm notes + rough-prototype findings (2026-05-30)

**Status:** PRE-SPEC exploration (the user asked for "deep brainstorming and rough
prototyping before committing"). Not a design spec yet. Evidence prototypes in
`docs/superpowers/evidence/2026-05-30-neural-copula-probes/`.

## Motivation

N3 showed bespoke summaries don't scale ([[mucov-bartlett-d5-verdict]]) and the PI
set the north star: a strategy that **scales to high-d and arbitrary distributions**,
framed as a **neural copula** ([[scalable-neural-copula-strategy]]).

## The design-space map (the key reframe)

The CD-SBI NF-MLE loss is the **SNL likelihood** `−log p(feat|θ) = ½‖r‖² −
log|∂r/∂feat|`: `r` maps **feat → N(0,I) conditioned on θ**, so `feat` MUST be
`d_θ`-dimensional (square Jacobian). **The summary bottleneck is structural, not an
implementation choice.** This yields a trilemma, all confirmed empirically:

1. **SNL-likelihood on a `d_θ` feat** → bottleneck; learning a sufficient `d_θ`
   summary for quadratic/complex stats fails (N3).
2. **Exact-density invertible** → no cheat, but the pivot still reads a `d_θ` subset →
   won't route quadratics (N3).
3. **Calibration-only** → no bottleneck, but collapses (II-A energy arm).

## The escape that worked: NLE + score pivot (a NEW direction)

Model `p(X|θ)` with a scalable conditional flow (NLE; **no bottleneck, info-complete,
cannot cheat** — proper likelihood on raw X). Read a `d_θ`-efficient frequentist CD
from the **score** `U(θ;X)=∇_θ log q(X|θ)`: at the true θ, `U|θ ~ N(0, I(θ))`
(mean-zero exactly; ≈Gaussian by CLT over the n_iid replicates). Rao stat
`U^T I(θ)^{-1} U ~ χ²_{d_θ}` → confidence set. The conditional-flow latent IS the
copula transform of X|θ; the score reads the θ-dependence. **No summary to engineer.**

## Rough-prototype results (coverage_error_max; noise floor ~0.02–0.03)

| target | exact/summary route | NLE+score (asymptotic) | + grid-free calibration | full-latent χ²₍dₓ₎ |
|---|---|---|---|---|
| (μ,Σ) d=5 | oracle 0.026 (Bartlett) | 0.060 | **0.038** | 0.108 |
| Cauchy loc-scale d=2 (**no suff. stat**) | **impossible** | **0.022** | 0.046 | — |

**Findings:**
1. **Escapes the bottleneck & scales.** Recovers the full (μ,Σ) covariance CD where
   every learned summary (7 N3 variants) failed — with no summary engineering.
2. **Generality validated.** On Cauchy location-scale — **regular but with NO finite
   sufficient statistic**, so the summary/oracle/Bartlett route is impossible — the
   score CD hits the noise floor (0.022). This is the scalability thesis demonstrated.
3. **Grid-free calibration sharpens finite-sample looseness** where asymptotics are
   loose ((μ,Σ): 0.060→0.038, near oracle) but adds mild noise where they're already
   tight (Cauchy: 0.022→0.046). So calibration is an optional sharpening, not mandatory.
4. **Full-latent (`d_x` pivot) is worst** (0.108): exact-in-theory but inefficient +
   flow-approximation error. Confirms the score's `d_θ`-efficiency matters.
5. **Likelihood-model robustness gates generality.** A vanilla MAF NaN'd on raw
   heavy-tailed Cauchy; an **asinh data pre-transform** (a θ-independent bijection, so
   the score is unchanged) fixed it. Data conditioning is part of the recipe (the PI's
   asinh intuition, resurfaced as tail-stabilization for the likelihood model).

## Emergent recipe (candidate scalable neural copula)

**conditional flow `q(X|θ)` (with data conditioning for robustness) + score-based
`d_θ` CD (+ optional grid-free calibration).** Scales to arbitrary dimension and
distribution; no summary bottleneck; cannot cheat (proper likelihood); near-floor
coverage incl. on no-sufficient-statistic targets.

## Comparison across all targets (probing round 2; rough, single-seed)

coverage_error_max (floor ~0.02–0.03). NC = neural-copula NLE+score.

| target | d_θ | exact CD-SBI / oracle | NC asymptotic | NC calibrated |
|---|---|---|---|---|
| §8.1 LocationNormal1D | 1 | 0.025 | **0.009** | ~floor |
| §8.2 LocationGaussian2D_iid | 2 | 0.025 | 0.032 | 0.040 |
| §8.3 LocationGaussian2D_corr | 2 | 0.025 | 0.034 | 0.042 |
| §8.4 ExponentialRate | 1 | 0.028–0.034 | 0.063 | 0.061 |
| (μ,σ²) NormalUnknownMeanVar | 2 | 0.026 | 0.043 | **0.030** |
| (μ,Σ) bivariate cov | 5 | 0.026 | 0.060 | **0.038** |
| Cauchy loc-scale (**no suff. stat**) | 2 | **impossible** | **0.022** | 0.046 |
| higher-d 5×Gaussian(μ,logσ) | **10** | — | **0.058** | 0.071 |
| **SLCP** (multimodal benchmark) | 5 | — | **0.115** | **0.426** ✗ |

**Reading:**
- **Regular targets:** NC score CD sits at 0.01–0.06 — within ~1–2.5× the *exact*
  CD-SBI floor. Gaussian-location (§8.1–8.3) is near-exact (linear score → exact χ²).
  Nonlinear-score cases (§8.4, (μ,σ²), (μ,Σ)) are looser (asymptotic) but **grid-free
  calibration sharpens them toward the floor** ((μ,σ²) 0.043→0.030; (μ,Σ) 0.060→0.038).
- **Scales in θ-dimension:** d_θ=10 → 0.058 (no blow-up).
- **Generality:** Cauchy (no sufficient statistic; CD-SBI machinery can't be built) →
  **0.022**, the floor.
- **THE BOUNDARY — non-regular/multimodal breaks it.** SLCP (θ₃,θ₄ enter squared →
  sign-symmetry multimodality + Fisher degeneracy near 0) → 0.115 asymptotic, 0.426
  calibrated (calibration *can't* rescue a non-regular statistic). The score/Rao CD
  assumes a regular, non-degenerate-Fisher, locally-quadratic (unimodal) log-likelihood;
  where that fails, so does the method. (Note: the earlier §8.1 0.227 was a
  d_x=1 zero-padding artifact, not a failure — clean run is 0.009.)

**Verdict so far:** the neural-copula score CD is a **scalable, general workhorse for
REGULAR models** — matching/approaching the exact CD-SBI floor, extending to
no-sufficient-statistic targets and higher d — but it is **NOT universal**: multimodal /
non-identifiable / Fisher-degenerate problems (a large part of the SBI benchmark suite)
need a different CD construction.

## The strategic fork (for the broad discussion)

The score CD is **asymptotic** (finite-sample loose, sharpenable by calibration);
the manuscript's monotone KR pivot is **exact (UMPU)** but doesn't scale to arbitrary
distributions. They are **complementary**: KR pivot for exact/tractable cases; the
neural-copula score as the scalable general workhorse. Open questions: behavior at
higher d and on standard SBI benchmarks (two-moons/SLCP); whether calibration can be
made to reliably reach the floor without hurting tight cases; how this repositions the
manuscript (a second, scalable estimator within the same CD framework).

## Caveats

Rough prototypes (single seed, modest budgets, throwaway code). Numbers are
directional, not validated replications. Next step (if pursued): a proper
brainstorm→spec on the neural-copula method, then TDD implementation + replication.
