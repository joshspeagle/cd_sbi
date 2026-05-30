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
