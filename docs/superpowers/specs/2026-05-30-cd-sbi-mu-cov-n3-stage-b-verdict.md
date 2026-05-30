# CD-SBI bivariate-normal (μ, Σ) — N3 Stage-B verdict (learned summary)

**Milestone:** N3 — Stage-B learned summary `s_φ: ℝ²⁰ → ℝ⁵` for the d=5 unknown-(μ,Σ)
target, via the information-preserving **Arm I-A** (invertible summary + exact-density
loss) verified in the 1-D (μ,σ²) bake-off. **Branch:** `feat/musigma-m3-verdict`.
**Outcome: a FINDINGS milestone, not a build** — the generic learned summary does not
scale to the covariance, bespoke fixes don't count, and the result reframes the whole
track toward a scalable **neural-copula** strategy (see §6).

This is a negative + existence result, deliberately kept as evidence rather than
productionised. Prototypes preserved under
`docs/superpowers/evidence/2026-05-30-n3-stage-b-summary/`.

---

## 1. The question

(μ,σ²) Stage-B verdict ([[musigma-stage-b-bakeoff-verdict]]): only the
**information-preserving** Arm I-A (invertible `AffineCouplingBijection` summary +
exact-density loss, bounded below by `H(X|θ)` so no collapse-cheat) recovered the
single scale parameter. N3 asks: does the same I-A recipe, unchanged, recover the
**full 2-D covariance** sufficient statistic (the 3 directions `A₁₁, A₂₂, A₁₂` ≈ the
Cholesky `D₁₁, D₂₂, D₂₁`) plus the 2 means, when the summary is *learned* from raw
X ∈ ℝ²⁰? The 1-D arm had only one scale to find; here the sufficient stats are
**quadratic** (Σx², Σx₁x₂) and the cross term is **signed**.

## 2. Headline finding: generic learned summaries don't route the quadratic stats

The I-A stack recovers the **means trivially** (linear) but **fails to route the
quadratic covariance information** into the d_theta-dimensional summary S. Seven
architecture variants, scored by how many of the 5 sufficient dimensions S captures
(canonical correlation S ↔ sufficient-stat, cross-checked with raw-moment R² and
per-coordinate PIT), all fresh-batch (no overfit), loss above floor:

| variant | sufficient dims recovered | note |
|---|---|---|
| plain affine coupling (the spec's I-A) | **4/5** (μ₁,μ₂,A₁₁,A₁₂) | A₂₂ missing |
| plain affine, **bigger, fresh** (best) | **4/5** | cross-cov A₁₂ → 0.92 |
| + Glow invertible 1×1 (LU) mixing | 2/5 (means only) | flexibility → *lazier* |
| naive permutation-equivariant + affine | 2/5 (means only) | scrambles; bad readout |
| asinh-affine coupling (`sinh(a·asinh b+d)`) | 2/5 + underfit | sinh tails unstable |
| asinh conditioner-features + affine | 2/5 | |
| equivariant-asinh + asinh router | 3/5 (means+A₁₁) | |

**Mechanism (unambiguous across the matrix):**
- **Linear means always route** into S; **quadratic (co)variances resist** — the
  exact-density objective is content to leave them in the ancillary block, paying a
  modest (~2-nat) penalty, because the ancillary stays *approximately* θ-free.
- **More architectural flexibility makes it WORSE.** Glow channel-mixing,
  permutation-equivariance, and asinh transforms all let optimisation find lazier
  2/5 solutions. The plain fixed-mask affine coupling's *rigidity* is what
  accidentally forces partial (4/5) routing.
- Even at best (big plain affine, fresh-batch), **one variance scale never routes**:
  canonical correlations S↔sufficient-stat = `[0.999, 0.999, 0.958, 0.919, 0.16]` —
  four directions captured, the fifth (comp-2 variance `A₂₂`, raw-moment R² 0.04)
  genuinely absent. Not a parameterisation artifact.
- **Calibration passes throughout** (joint ‖r‖²~χ²₅ KS ≈ 0.03–0.04; per-coord PIT
  mostly fine) because a flow emits N(0,1) marginals regardless of what S contains.
  The CD is therefore **valid but inefficient** on the un-routed direction — the same
  *calibration ≠ sufficiency/efficiency* theme as the (μ,σ²) bake-off and the N2 μ₂
  limit, here at its sharpest.

Neither of the user's two architectural suggestions rescued it: **asinh** (sign-aware,
log+linear — the right intuition for the signed cross-covariance) was unstable in
output form and inert as a feature; **permutation-equivariance** (correct symmetry for
exchangeable replicates) scrambled the readout. Both *reduced* recovery. The
obstruction is deeper than the elementwise transform: nothing **pressures** the
quadratic information into a low-dimensional summary.

## 3. Existence proof (not a method): the structure-informed summary

A bespoke **Helmert + polar/QR** invertible summary recovers all 5 and calibrates:
- Fixed Helmert over the 10 exchangeable replicates → 2 channel means + 18 contrasts
  (orthogonal, log-det 0) — converts "pool across replicates" into a fixed op.
- Fixed polar/QR on the contrasts → exposes the Bartlett Cholesky `(D₁₁, D₂₁, D₂₂)`
  as radial/projection coords; the sphere directions are the 15 ancillary dims
  (provably θ-free by Basu). Tractable log-det `8·logD₁₁ + 7·logD₂₂ + const`.
- Validated: forward = exact oracle Bartlett features (diff ~1e-5), invertible
  (recon err 1.9e-6), pivot per-coord PIT KS ≤ 0.069 (the 0.069 is μ₂ at an extreme —
  the *same* mild N2 limit), χ²₅ KS 0.027.

**But this is an existence proof, not a scalable method.** It works only by hard-coding
the bivariate-normal Bartlett decomposition; it would need re-deriving for every new
distribution and does not survive "arbitrary complex distributions." We deliberately
do **not** productionise it.

## 4. Baseline: LF2I-BFF also breaks at d=5

A natural question (does an explicitly-calibrated method sidestep sufficiency?): run
the LF2I-BFF 2-step baseline on raw X∈ℝ²⁰. Its quantile-calibrated critical values
*should* guarantee frequentist coverage regardless of sufficiency. Result (reasonable
but untuned settings; BFF grid = 128 pts over a (−3,3)⁵ bounding box):

| grid | coverage_error_max |
|---|---|
| 3-pt (center + 2 extremes) | **0.393** (50% set covers 11% at center) |
| 16-pt LHS over the true prior | **0.201** |

Our LF2I-BFF scored coverage_error 0.20–0.39 at d=5 — **but this is a defect in OUR
implementation, not LF2I** (confirmed by a recipe research pass + a code audit,
2026-05-30):

- **The recipe needs no grid.** The BFF averaged term (Dalmasso et al. 2024, EJS,
  Eq. 10) is `E_{θ~π}[∏ᵢ O(Xᵢ;θ)]` — a Monte-Carlo expectation over **proposal draws**;
  critical-value calibration is grid-free quantile regression of `c_α(θ)` on θ. The
  only "grid" in LF2I is the candidate-θ list for confidence-set membership (pointwise,
  replaceable). The PI's assertion is correct.
- **Our defect** (`lf2i_bff.py:71–87,114`): the BFF marginal uses a **fixed N=64 sample
  set, frozen at fit time and blind to d**. At d=1 (§8.1, §8.4-on-T) it is an exact
  64-pt linspace quadrature; at d=2 (§8.2, §8.3) ≈ 8 pts/axis (degraded but real —
  consistent with the published 0.11–0.14); at d=5 ≈ 2.3 pts/axis → the marginal is
  noise. Compounded by a box-uniform proposal that extrapolates the classifier outside
  its training prior.
- **No correctness bugs** in sign convention, pinball loss, or containment; the
  calibration stage is faithful and grid-free.
- **§8.1–8.4 published numbers are NOT invalidated** (d=1 exact; d=2 a genuine
  measurement of the N=64 estimator); no re-run required.
- **Faithful/scalable fix landed** (`lf2i_bff.py`): the BFF marginal is now an MC
  average over draws from the **true prior** (`simulator.sample`, the proposal the
  classifier trained on — removes the box-uniform extrapolation too), with a d-aware
  count `max(marginal_n, 128·d)`; config `marginal_grid_n → marginal_n` (alias kept).
  TDD: 3 new regression tests in `tests/integration/test_lf2i_bff_smoke.py`.

**Fair d=5 LF2I-BFF baseline (fixed impl, marginal_n=2048):**

| coverage_error_max | unfixed (N=64 box) | **fixed (N=2048 MC-prior)** | CD-SBI oracle |
|---|---|---|---|
| 3-pt grid (center+extremes) | 0.393 | **0.091** | 0.026 |
| 16-pt LHS (true prior) | 0.201 | **0.189** | 0.026 |

The fix repairs the catastrophic central-region under-coverage (50% set: 11% → 46%
at θ₀=0). The residual ~0.19 worst-case is at *extreme* θ₀ — consistent with
classifier/quantile-head capacity at the prior edges (a training-budget matter), not
the marginal estimator. **Fair verdict:** LF2I-BFF is a legitimate d=5 baseline with
decent central coverage but still ~7× the oracle's worst-case error; it does not match
the oracle CD-SBI procedure. (Evidence: `…/evidence/.../proto_lf2i_fair.py`.) For
contrast, CD-SBI Stage-A with the **oracle** Bartlett summary gets coverage ~0.026.

## 5. Verdict

**No scalable *learned* story exists yet for the d=5 covariance CD**, and bespoke
fixes don't count. The Stage-A oracle and the structure-informed summary both calibrate
because they *hand-supply* the sufficient statistic; every attempt to *learn* the
quadratic covariance generically (7 variants) recovers at most 4/5 and is not improved
— often harmed — by added flexibility; the headline calibration baseline (LF2I-BFF)
fails outright at d=5. The d=5 (μ,Σ) **Stage-A** result stands (oracle Bartlett pivot,
[[mucov-bartlett-d5-verdict]]); **Stage-B learned-summary is an open problem**, and the
right framing of that problem is the scalability concern in §6.

## 6. Reframe → scalable "neural copula" (the real next direction)

The user flagged (2026-05-30) that specialised, per-problem constructions are a dead
end; the strategy must **scale to many dimensions and arbitrary distributions**, framed
as a kind of **neural copula** ([[scalable-neural-copula-strategy]]). This is apt for
CD-SBI: the pivot target `r(θ;X) ~ N(0,I_d)` at truth *is* a normal-scores /
Gaussian-copula transform. The scalable version: a conditional, information-preserving
(invertible) neural map over `(θ, X)` that scales by construction (autoregressive /
coupling / transformer flows), with information-preservation as the *generic* guard
against the sufficiency-collapse we hit — no per-problem summary engineering. Key
implication from N3: **the fixed-dimension summary bottleneck may itself be the
unscalable step** — a neural copula needn't compress to `d_theta` features; the pivot
can read the full invertible representation.

**Deferred (user's call):** a dedicated broad strategy discussion + brainstorm once the
current experiments are done. N3 closes here as the evidence base that motivates it.
