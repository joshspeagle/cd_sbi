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

## Naming + NSF density model (2026-05-30)

**Method name: Score-CD** (honest: NLE density model + score-test confidence-distribution
readout). Considered CD-NLE / Pivot-SBI; Score-CD names the mechanism.

**MAF → NSF density model (zuko), repeat SLCP + (μ,Σ).** Does a more expressive
likelihood model fix SLCP, or is the failure intrinsic?

| target | NLE -logq (MAF → NSF) | asymptotic (MAF → NSF) | calibrated (MAF → NSF) |
|---|---|---|---|
| SLCP | 10.99 → **3.06** (much better fit) | 0.115 → **0.111** (unchanged) | 0.426 → **0.648** (worse) |
| (μ,Σ) d=5 | — → 15.07 | 0.060 → 0.066 | 0.038 → 0.055 |

**Decisive finding:** NSF fit the SLCP likelihood *dramatically* better (−logq 10.99→3.06)
but Score-CD coverage **did not improve** (0.111≈0.115) and calibration got **worse**
(0.648). So the SLCP breakage is **NOT a density-model-quality problem** — it is the
**intrinsic regularity failure** of the score test: `θ₃,θ₄` enter squared → Fisher
**degeneracy** at `θ₃,θ₄→0` (which the U(−3,3) prior hits) + **sign-symmetry
multimodality**. A better likelihood model makes the density sharper but the score
statistic *more* pathological at the degeneracies, so calibration fails harder. **The
regularity boundary is architecture-independent.** On the regular (μ,Σ), NSF≈MAF
(0.066 vs 0.060) — the density model is *not* the coverage lever there either; the
gap to the exact oracle is finite-sample asymptotics, not flow quality.

**Implication:** MAF vs NSF does not move Score-CD coverage on these targets. NSF is
still the sensible **default density model** (far better fit on complex likelihoods;
needed for high-d / structured data), but the coverage levers are (a) problem
regularity and (b) finite-sample calibration — not flow expressivity. Evidence:
`…/evidence/2026-05-30-neural-copula-probes/proto_score_cd_zuko.py`.

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

## SLCP "fix" probe: LF2I-BFF benchmark (2026-05-30)

Hypothesis tested: swap the score for a Fisher-free CALIBRATED statistic (LF2I-BFF) →
dodge the degeneracy. Result (fixed LF2I-BFF, MC marginal_n=2048, asinh, true U(-3,3)^5 box):

| | SLCP benign grid (θ₃,θ₄ away from 0) | SLCP full-box LHS (incl. θ₃,θ₄≈0) |
|---|---|---|
| Score-CD | 0.111 asymptotic / 0.648 calibrated | — |
| LF2I-BFF | 0.148 (conservative @50%, fine @90/95) | **0.500** |

**LF2I-BFF does NOT rescue SLCP** — comparable on the benign region, blows up over the
full box. **Partial refutation of the statistic-swap fix:** the failure is not *which
statistic*, it is that near θ₃,θ₄→0 the parameter is **nearly non-identifiable** — the
likelihood is near-deterministic, the statistic's distribution is pathological/wildly
θ-dependent, and the learned critical-value surface c_α(θ) can't be fit there, for ANY
statistic in the family. Both Score-CD and LF2I-BFF handle regular SLCP and break in
the degenerate region.

**Sharper "what resolves it":** (a) reparameterize away the degeneracy (problem-specific,
non-scalable — the trap); (b) degeneracy-aware / local critical-value calibration (open,
hard); (c) accept that non-identifiable regions cannot have tight valid CDs — the honest
output is an uninformative (huge/unbounded) set; the goal becomes validity + honesty
about non-identifiability, not tightness. Evidence: `proto_lf2i_slcp.py`.

## Score-CD productionized + matched-budget cross-method sweep (2026-05-31)

`ScoreCDRunner` (src/cdsbi/methods/score_cd.py): NLE(MAF, identical to the NLE
baseline) + score CD readout, two variants as CriticalValueProcedures —
`rao` (UᵀÎ(θ)⁻¹U ~ χ²_{d_θ}, analytic threshold) and `cal` (‖U‖² + learned c_α(θ)).
Wired into run.py (configs score_cd_{rao,cal}); §8.4's on-T ReducedSimulator applies
(method≠cd_sbi). Added to the §8.1–8.4 sweeps (2 variants × 4 budgets × 5 seeds × 4
sections = 160 runs, fresh_batch=false).

**coverage_error_max (mean/5 seeds), by budget — floor ~0.025:**

| sec | method | small | med | large | xlarge |
|---|---|---|---|---|---|
| 8.1 | cd_sbi / score_rao / nle | 0.025 / 0.029 / 0.025 | 0.025/0.030/0.025 | 0.025/0.029/0.030 | 0.025/0.030/0.028 |
| 8.2 | cd_sbi / score_rao | 0.022/**0.043** | 0.020/0.043 | 0.026/0.075 | 0.027/**0.192** |
| 8.3 | cd_sbi / score_rao | 0.022/**0.048** | 0.025/0.044 | 0.027/0.089 | 0.026/**0.182** |
| 8.4 | cd_sbi / score_rao / lf2i | 0.028/**0.363**/0.090 | .../.../0.078 | .../.../0.062 | 0.034/0.363/0.057 |

(score_cd_cal: §8.1 0.08, §8.2 0.08–0.12, §8.3 0.07–0.13, §8.4 0.12–0.21. Other
baselines: NPE 0.05–0.10, NLE 0.025–0.25, NRE 0.08–0.19, LF2I 0.06–0.14.)

**Findings:**
- **Score-CD-rao beats every non-CD-SBI baseline (NPE/NLE/NRE/LF2I) at small/medium
  budget on §8.2/§8.3** (0.043–0.048) and ties NLE/CD-SBI on §8.1 — a real win for the
  score readout.
- **Budget-degradation (d≥2):** rao 0.043→0.192 (§8.2), 0.048→0.182 (§8.3) small→xlarge.
  §8.1 (d=1) flat. Hypothesis: NLE overfits the FIXED finite set (fresh_batch=false);
  the score amplifies density overfitting. **CONFIRMED:** fresh_batch=true (infinite
  data, no overfit) → degradation VANISHES: §8.2 xlarge 0.192→0.033, large 0.075→0.029;
  §8.3 xlarge 0.182→0.032, large 0.089→0.034 — back to the floor, alongside CD-SBI.
  So it is NLE overfitting (score-amplified), fixable by fresh data / early-stopping /
  regularization; CD-SBI's monotone pivot is regularized-by-construction so it needs
  no such care (its key practical robustness advantage).
- **§8.4 failure (0.363, flat):** Score-CD gets the on-T reduction (d_x=1) + no asinh,
  so its score on the skewed 1-D Gamma T is badly miscalibrated. Contradicts the raw-X
  probe (0.063, battery4) — confirms data-conditioning gates Score-CD. (Follow-up:
  raw-X / asinh for §8.4.)
- **CD-SBI stays at the floor (0.02–0.03) across all sections/budgets** — the comparison
  reinforces its robustness; Score-CD is a strong low/medium-budget method but brittle
  (overfits with budget, sensitive to data conditioning / reduction).

## Scalable eval engine + (μ,Σ) comparison (2026-05-31)

**The μΣ full sweep exposed that the *evaluation harness* doesn't scale** (≈93 min/run;
21 GiB OOM at d=5) — the per-θ₀/per-diagnostic loops recompute the statistic (a
backward pass for Score-CD) in one un-chunked forward. Redesigned per Option-1:
`cdsbi/diagnostics/engine.py::evaluate_coverage` — **statistic-once** (compute the
pivot r / test-stat T a single time per θ₀, derive all metrics), **chunked**
(memory O(chunk)), **simulate-once**. Results:
- **Verified bit-identical to the legacy Coverage diagnostic** on trained models
  (CD-SBI pivot + Score-CD critical-value: max|Δ|=0.0000). Pure speed/memory win.
- **21 GiB OOM → 75 MiB; ~80 min eval → <1 s** (~5000×). Correctness + chunk-invariance
  unit-tested (`tests/unit/test_eval_engine.py`).

**(μ,Σ) d=5 comparison** (engine-evaluated, 16-pt LHS grid, 12k steps, fresh_batch=False,
3 seeds), coverage_error_max by budget:

| method | small | medium | large | xlarge |
|---|---|---|---|---|
| cd_sbi | 0.054 | 0.094 | 0.091 | 0.152 |
| score_cd_rao | 0.110 | 0.105 | 0.105 | 0.135 |
| score_cd_cal | 0.124 | 0.170 | 0.125 | 0.199 |

- Both above the §8.1–8.3 floor: d=5 is harder (CD-SBI's documented μ₂ extreme-θ₀ limit
  inflates the worst-over-grid metric; finite-data overfitting hits both). On a
  non-extreme 3-pt grid CD-SBI is 0.023 (floor) — the 16-pt LHS worst-case is
  extreme-θ₀-driven.
- **CD-SBI degrades with budget (0.054→0.152)** at d=5 (bigger single_index overfits
  the fixed set); **Score-CD-rao stays flat ~0.11–0.14 and matches/beats CD-SBI at
  large/xlarge** — a reversal from lower d where CD-SBI's by-construction regularization
  kept it at floor.

**Follow-up:** wire the engine into run.py to make the whole harness scalable (the
engine is a standalone primitive now; the §8.x diagnostic loop is still the legacy path).

## DECISION (2026-05-31): rename Score-CD → LF2I-Score; integrate as a §10 pointer; close out

After a deep read of the manuscript's aims/spine (§1.3 top-line goal = *exact*
pointwise CD; §3.7 = NF-MLE is the SNL/Class-5 loss and the contribution is the
*monotone architectural recipe* that makes it a CD *by construction*; §4–6 =
uniqueness/UMPU theorems; §10 already identifies the exact pivot with the
standardized **score / signed-root LR**, "asymptotically N(0,1)"):

**Verdict — the method is NOT a CD-SBI variant; it is an LF2I-family method.** Its
calibrated form IS LF2I (learned test statistic + Neyman-inverted quantile critical
values) with the **score** as the statistic — the mode §10 defines CD-SBI *against*.
So it is renamed **LF2I-Score** and kept a **separate contribution**, not a section of
the exact-CD paper (adding it would dilute the crisp exact/by-construction/UMPU thesis
and blur the §10 distinction).

**Integrated into the manuscript** as a single measured paragraph in §10 ("A
test-statistic variant: LF2I-Score") appended to the existing score/signed-root-LR
discussion: notes the score of a learned neural likelihood, calibrated by quantile
regression, gives a scalable confidence set in the LF2I/Neyman mode — *asymptotic, not
exact/by-construction; two-stage* — a complement to CD-SBI, developed separately. Cites
existing refs (LF2I, WALDO, Schweder–Hjort); manuscript rebuilt cleanly.

**Code naming:** the implementation is still `cdsbi/methods/score_cd.py` +
`score_cd_{rao,cal}` configs (calibrated variant = LF2I-Score; asymptotic-χ² variant =
"Rao test on a neural likelihood"). A code rename is deferred to the separate LF2I-Score
contribution — and **gated on a novelty literature check** (is neural-likelihood-score
as the LF2I statistic new? ACORE=odds, Waldo=posterior moments, BFF=Bayes factor —
the score slot may be open, but score-based frequentist SBI is active).

**Exploration closed.** Reusable outputs landed on main: the Score-CD/LF2I-Score method,
the scalable eval engine (wired into run.py — kills the d=5 OOM), and the §10 manuscript
pointer. Open follow-ups: the separate LF2I-Score paper (+ lit check + code rename), the
deferred neural-copula strategy brainstorm, the multimodal/SLCP frontier.
