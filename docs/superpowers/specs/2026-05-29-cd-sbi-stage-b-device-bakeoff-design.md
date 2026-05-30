# CD-SBI Stage B — learned-summary device bake-off (design)

**Status:** validated design (brainstorm 2026-05-29), ready for plans.
**Supersedes:** the Stage-B portion (§B.3 device list) of
`2026-05-29-cd-sbi-unknown-mean-variance-design.md` — that spec proposed device-1
(drop summary log-det + normalize) as the primary; M2's smoke proved device-1
*structurally* cannot work (audit below). This spec replaces it with a three-arm
bake-off against the Stage-A oracle control.
**Target:** unknown-(μ, σ²) Gaussian, θ=(log σ, μ), `n_iid=10`, `d_theta=2`
(`NormalUnknownMeanVar`). Stage A (oracle summary + `SingleIndexMonotoneFlow`)
is landed on `main` and is the control.

> **Revision (2026-05-29) — co-adaptation reframing.** M3.0 (shared harness)
> landed; the M3.1 dry-run of **Arm I-B (two-stage freeze)** then surfaced a
> deeper finding (§1.5) that reorders the bake-off. **The arms are now framed by
> co-adaptation:** the disease is an *exploitable objective*, not co-adaptation
> itself; under a *cheat-free* objective, end-to-end co-adaptation is the cure
> (it forces the summary to emit calibratable coordinates). **Arm II-A is now the
> primary; Arm I-A is the benchmark; Arm I-B is demoted to a documented
> contrast** (it is structurally fragile — see §1.5). §3 and §4 reflect this.

---

## 1. The audit — why a learned non-square summary cheats NF-MLE

With the learned summary `feat = s_φ(X): ℝ¹⁰→ℝ²` and `log_det_contrib = 0`, the
per-example NF-MLE loss is
```
ℓ = ½‖r‖² + (d/2)·log 2π − log|∂r/∂feat|,   r = T(θ; feat).
```
For a **fixed** feature map, minimizing over the flow `T` gives, at the optimum,
the conditional differential entropy `E[ℓ] = H(feat | θ)`. In Stage A `feat` is
the oracle sufficient statistic, so `H(feat|θ)` is a **fixed constant** (the floor
≈ 0.92); the loss is bounded and the minimizer is calibrated.

When `feat` is **learned**, `H(feat|θ)` is no longer fixed. Minimizing the loss
*over φ* drives it down: the conditioner squeezes `feat|θ` toward a near-point-mass,
the flow stretches that narrow distribution back to `N(0,I)` (large `∂r/∂feat`),
so `log|∂r/∂feat| → +∞` and `ℓ → −∞`. **Observed in M2:** final loss −5.29 vs
floor 0.92.

The collapse **trades away sufficiency** — it discards the information most
expensive to encode. For mean-pooled DeepSets, μ rides on the *linear* statistic
`X̄` (trivial), while σ² needs a *second moment* (`X̄² → s²`, nonlinear). So the
net keeps `X̄` (M2: Spearman 0.996) and sacrifices σ² (0.47). The σ-confidence
sets become invalid even as the aggregate loss looks excellent — the harmful core
of the cheat.

**Root cause (Fisher–Neyman).** `−log p(X|θ) = −log p(feat|θ) − log p(X|feat)`;
for sufficient `feat` the second term is θ-free. NF-MLE on a *reducing* summary
silently **drops `−log p(X|feat)`**, and a learned reducer then inflates the
remaining `−log p(feat|θ)` by collapsing `feat`. The objective optimizes
`−log p(feat|θ)` (collapsible) instead of `−log p(X|θ)` (fixed, bounded by
`H(X|θ)`).

**Why device-1 was structurally doomed.** Its two parts miss the cheat channel:
`log_det_contrib=0` zeroes the *conditioner's* Jacobian credit, but the cheat is
the *flow's* `log|∂r/∂feat|`; and feature normalization fixes feature *scale*
(unit variance) while the collapse is about feature *information* (a unit-variance
feature can still be near-deterministic given θ). Scale ≠ information.

**Consequence.** Every principled fix must restore dependence on `−log p(X|θ)`,
sidestep it, or replace the objective that needs it. Three admissible families
(soft-penalty / decoder-MI variants are excluded by the project's
architectural-not-penalty ethos).

---

## 1.5 The second finding — affine-single-index bias vs warped learned features

The M3.1 dry-run of Arm I-B exposed a second, distinct issue (verified by running
the pipeline). `SingleIndexMonotoneFlow` bakes in an inductive bias: the pivot is a
monotone function of an **affine** index `z_k = a·θ_k + b·feat_k + off`, with `a,b`
**constant** at coordinate 0 (which has no autoregressive context). This is exactly
right for the **oracle** feature, because the closed-form pivot genuinely is
single-index affine in the natural sufficient statistic (`r*_σ = g(log s² − 2 log σ)`).
A **learned** summary has no reason to emit that natural coordinate. Predict-θ
(Arm I-B's stage 1) emits the **posterior mean** `E[log σ|X]`, which *shrinks* — a
monotone but **nonlinear warp** of `log s²`. The constant-coefficient coord-0
combiner cannot un-warp it, so even though **sufficiency is retained** (Spearman
0.97), the **σ-coordinate miscalibrates** (joint KS 0.10–0.11 vs the 0.06 band;
σ-marginal KS 0.17–0.31). Sufficiency is necessary but not sufficient — the
*coordinate* must also be in the flow's affine basis.

**The unifying axis: co-adaptation under a cheat-free objective.**

| Device | Co-adapts summary↔pivot? | Objective | Outcome |
|---|---|---|---|
| M2 naive | yes | NF-MLE (Jacobian cheat channel) | **cheats** — co-adapts to inflate density; σ collapses |
| Arm I-B freeze | **no** (frozen after predict-θ) | predict-θ, then NF-MLE | **warps** — sufficient but locks a non-affine coordinate the flow can't calibrate |
| Arm II-A | yes | calibration-invariant (no Jacobian) | *hypothesis:* co-adapts to calibration; no cheat channel; learns flow-friendly coords |
| Arm I-A | yes | NF-MLE but exact (invertible) | *hypothesis:* co-adapts; exact change-of-variables, can't destroy info |

Co-adaptation is **not** the disease — an exploitable objective is. Under a
*cheat-free* objective, end-to-end co-adaptation is the **cure**: gradient pressure
forces the summary to emit exactly the coordinates the pivot can calibrate, which
**solves the warp and the cheat at once**. Arm I-B forfeits co-adaptation (it
freezes a summary trained for a *different* objective), so it cannot un-warp — it is
**structurally fragile**, not fixable by tuning. Hence the reframing: lead with the
end-to-end cheat-free arms; keep I-B only as the contrast that motivates them.

---

## 2. Framing — co-adaptation bake-off (II-A primary, I-A benchmark, I-B contrast)

Deliverable = an **empirical verdict** on the §1.5 question: *does end-to-end
co-adaptation under a cheat-free objective make the learned summary emit un-warped,
calibratable coordinates* — calibrating as well as the Stage-A oracle? The
motivating use case is targets where the sufficient statistic is **unknown**;
(μ,σ²) is the controlled stress test where we know the truth and can measure both
recovery and the warp. All arms are judged on the **same** harness (M3.0, landed).

### 2.1 Shared evaluation harness (build once, reuse)

- **Calibration (existing diagnostics):** `PivotRMSE`, `MarginalPIT`, `Coverage`
  (2-D 3×3 grid), `MarginalCDRecovery` (σ²→χ², μ→Student-t), `JointMahalanobis`.
- **`SufficiencyRecovery` (new diagnostic):** Spearman (rank, monotone-invariant)
  of each learned feature vs the oracle `(log s², X̄)`; reports per-coordinate
  best-|Spearman| and Pearson (diagnostic). Gated like `MarginalCDRecovery`
  (no-ops unless the simulator exposes the oracle stat + a learned conditioner is
  present). Quantifies "did the summary keep the sufficient information."
- **`FloorIntegrity` (landed in M3.0; needs the empirical-floor fix):** **arm-aware.**
  - NF-MLE arms (I-A): final loss must satisfy `loss > H − tol` (does NOT sink
    below the conditional-entropy floor). **Fix required:** the M3.0 version
    compares to the *oracle*-feature floor `entropy_lower_bound()`; a learned
    summary's features have a *different* conditional entropy `H(feat|θ)`, so the
    oracle floor gives a **false cheat** for any non-oracle-scale summary (the
    M3.1 run hit exactly this). The check must use an **empirical `H(feat|θ)`**
    estimated from the trained `encode_fn` (MC over θ then the NF-MLE loss at the
    true conditional density of the actual features), not the oracle closed form.
  - Calibration-loss arm (II-A): no `H` to undercut — instead require the scoring
    rule → its floor (≈0) **with** `SufficiencyRecovery` intact (a degenerate
    summary cannot satisfy calibration across θ₀, so low loss + high sufficiency
    is the integrity signal).
- **Control numbers (Stage A, in hand):** coverage_error_max 0.026, σ²/μ KS 0.020,
  μ_t_resid p95 0.017, JM 5/5, final_loss 0.904 ≈ H 0.921. An arm "passes" if it
  matches the control within the same tolerance bands AND `SufficiencyRecovery`
  Spearman > 0.9 on both coordinates AND `FloorIntegrity` holds.

### 2.2 The M2 smoke becomes a cheat-capture regression test

Convert `tests/integration/test_mu_sigma_stage_b_smoke.py` from a (red) calibration
assertion into a **regression test that asserts the device-1 cheat is present**
(final loss ≪ floor AND σ²-Spearman ≪ 0.9), documenting the pathology — exact
precedent: `tests/ablation/test_trained_folding.py` captures the §3.5 folding
failure the same way. This keeps the suite green and pins the motivating finding.

---

## 3. The arms — order of attack: II-A (primary) → I-A (benchmark) → I-B (contrast)

Build and judge the **end-to-end co-adaptive** arms first (they are the candidate
cures per §1.5); I-B is implemented only enough to *document* its fragility.

### Arm I-B — two-stage freeze (DOCUMENTED CONTRAST — not a contender)

**Status:** the M3.1 plan (`dc0c08e`) + the review dry-run already produced I-B's
result: predict-θ pretraining recovers sufficiency (Spearman 0.97/0.9996) but the
frozen features are **warped** (posterior-mean shrinkage), and the fixed-sign
affine pivot **miscalibrates σ** (joint KS 0.10–0.11; §1.5). This is the structural
fragility of forfeiting co-adaptation — captured as a contrast/ablation that
*motivates* the co-adaptive arms, **not** pursued as a calibrating device. No
attempt to rescue it (rectifier, InfoNCE, etc.) — that effort goes into the arms
that co-adapt by construction.

- **Stage 1.** DeepSets `s_φ:ℝ¹⁰→ℝ²` trained alone to predict θ (MSE), then frozen.
- **Stage 2.** The unchanged pivot + NF-MLE on frozen feat. Calibrates only if the
  frozen coordinate happens to be in the flow's affine basis — which predict-θ
  (shrinkage) violates for σ. (Retained mainly as the contrast in M3.3′.)
- **New code:** `pretrain_summary(simulator, summary, head, config) -> frozen summary`
  utility; a `FrozenConditioner` wrapper (or freeze-in-place: zero the conditioner
  param group so `fit()`'s optimizer is a no-op on it). `fit()` already tolerates a
  conditioner with no trainable params.
- **Verdict signal:** preserves the d=2 monotone-pivot architecture exactly;
  generalizes (stage-1 finds sufficient features when the truth is unknown). Risk:
  not end-to-end; stage-1 features must be orientable by the fixed-sign pivot
  (surfaces as miscalibration if not — detectable).

### Arm II-A — calibration-invariant objective (PRIMARY; §3.7/v5 thread)

*Why primary (§1.5):* end-to-end co-adaptation under a loss with **no Jacobian
cheat channel**. The summary co-adapts to minimize distance-to-`N(0,I)`, which
*requires* it to emit calibratable (un-warped) coordinates — so it is the minimal
device that addresses **both** the M2 cheat and the I-B warp at once. The empirical
question the bake-off answers first: does it?

- Keep DeepSets end-to-end + the monotone pivot, **replace NF-MLE** with a
  distance of `{r(θ₀;X)}` to `N(0,I₂)`. Training uses **grouped batches**: for each
  θ₀ in a batch of `B` proposal draws, sample `m` datasets `X_j ~ p(·|θ₀)`, compute
  `{r(θ₀; s_φ(X_j))}_{j=1..m}`, score its distance to `N(0,I₂)`, average over the
  `B` groups.
- **`EnergyCalibrationLoss`:** the energy distance between the empirical sample
  `{r_j}` and `N(0,I₂)` (closed-form / Monte-Carlo energy score against standard-
  normal reference draws). Bounded below by 0; invariant to feature
  reparameterization (sees only `r`'s distribution, no Jacobian); → 0 iff
  `r(θ₀;·)|θ₀ = N(0,I₂)`. An insufficient feat cannot achieve this **across all
  θ₀** simultaneously, so sufficiency is enforced implicitly.
- **New code:** `EnergyCalibrationLoss`; a grouped-sampling training path
  (`simulator.sample_x_given_theta` already supports `m` per θ₀) — a sibling runner
  method or a `batching="grouped_by_theta"` mode in `fit()`. The monotone flow and
  its R1/R2 guarantees are unchanged (still invertible-in-θ for confidence sets);
  only the *loss* changes.
- **Verdict signal:** end-to-end, no collapse channel, no Jacobian games; connects
  to the §3.7 alt-loss taxonomy (effectively pulls v5 forward). Risks: higher-
  variance training (group size `m`, batch `B` are knobs); must confirm it targets
  the calibration manifold `M` properly (it directly minimizes distance to `N(0,I)`
  at each θ₀, so it targets `M` by construction — but verify no degenerate
  optimum, e.g. `r` constant: a constant `r` is not `N(0,I₂)` so the energy score
  penalizes it).

### Arm I-A — invertible summary (BENCHMARK; end-to-end, exact)

*Role (§1.5):* the end-to-end, exact-change-of-variables gold standard. Co-adapts
like II-A but via an exact bijective likelihood (can't destroy info). If II-A
matches I-A on the harness, II-A wins on simplicity; if II-A falls short, I-A shows
the achievable ceiling.

- Permutation-equivariant **bijection** `s_φ: X → (S∈ℝ², A∈ℝ⁸)` over the
  exchangeable iid axis. Model the **full** conditional density:
  ```
  −log p(X|θ) = ½‖r(θ;S)‖² − log|∂r/∂S| − log|∂s_φ/∂X| + ½‖A‖² + const,
  ```
  where `r(θ;S)` is the existing monotone pivot on the 2 inference features and
  `A` is the 8 ancillary coordinates trained to `N(0,I₈)`. The bijection log-det
  `−log|∂s_φ/∂X|` and the `½‖A‖²` block are the anti-collapse terms: degenerating
  `S` forces the Jacobian toward singular, blowing up `−log|∂s_φ/∂X|`. Exact,
  bounded by `H(X|θ)`, and the **d=2 monotone pivot is preserved** as one block of
  a full normalizing-flow model of `X|θ`.
- **New code:** a permutation-equivariant coupling/spline flow over the iid axis
  (e.g. DeepSets-conditioned coupling layers, or a sort-based equivariant flow) —
  the heaviest lift; its `encode(X) → (feat=S, log_det=log|∂s/∂X|)` plugs into the
  existing `Conditioner` protocol with a **genuine, non-zero** `log_det_contrib`,
  and an auxiliary `½‖A‖²` term added to the loss.
- **Verdict signal:** the unimpeachable control for "what calibration is
  achievable with a learned, information-preserving summary." If II-A matches
  I-A's calibration, II-A wins on simplicity (no equivariant bijection needed).

---

## 4. Decomposition & order (revised — co-adaptation first)

Each numbered item is its own implementation plan; each lands a working unit + its
result on the shared harness.

- **M3.0 — shared harness. ✅ LANDED** (`SufficiencyRecovery` + arm-aware
  `FloorIntegrity` + the M2 cheat-capture regression test, on `feat/musigma-m2`).
  *Carry-over:* the `FloorIntegrity` **empirical-`H(feat|θ)` fix** (§2.1) — folded
  into M3.1′ since I-A needs an honest floor.
- **M3.1′ — Arm II-A (PRIMARY).** `EnergyCalibrationLoss` + grouped-by-θ₀ training
  path + the `FloorIntegrity` empirical-floor fix; intensive replication on the
  harness. **Answers the core §1.5 question first:** does co-adaptation under a
  cheat-free loss emit calibratable coordinates (σ included)?
- **M3.2′ — Arm I-A (BENCHMARK).** Permutation-equivariant bijection summary +
  full-density loss (the `+½‖A‖²` ancillary block); intensive replication. The
  achievable-ceiling control for II-A.
- **M3.3′ — verdict + I-B contrast + manuscript.** Cross-arm comparison table
  (II-A and I-A vs the oracle control on every harness metric); the **I-B contrast**
  written up from its existing plan + dry-run result (sufficient-but-warped →
  σ-miscalibration); the §1 + §1.5 audit written for the draft.

**I-B is NOT a separate build step** — its plan `dc0c08e` and the review dry-run
are the source material for the M3.3′ contrast. (The plan stays in the repo as the
record; it is not executed as a calibrating arm.)

---

## 5. Success criteria

- **Per arm (II-A, I-A):** matches the Stage-A control within the same tolerance
  bands (coverage_error_max ≤ 0.05; σ²/μ marginal-CD KS ≤ 0.06 — **σ included**, the
  coordinate I-B fails; JM ≥ 4/5 seeds at 2× floor) **AND** `SufficiencyRecovery`
  Spearman > 0.9 on both coordinates **AND** `FloorIntegrity` holds (arm-aware,
  empirical floor).
- **Milestone:** at least one end-to-end co-adaptive device (II-A or I-A) makes a
  *learned* summary meet the per-arm bar — answering §1.5 in the affirmative; the
  cross-arm table + verdict are produced; the M2 cheat and the I-B warp are both
  captured (regression test + documented contrast). A negative result for an arm
  (e.g. II-A high-variance) is a *reportable finding*, not a milestone failure —
  but if neither co-adaptive arm calibrates σ, that reopens the flow-architecture
  question (§1.5 directions: feature-rectifier / richer combiner).

---

## 6. Scope

**In scope:** the shared harness; the three arms; the cross-arm verdict on the
(μ,σ²) target. New components: `SufficiencyRecovery`, `FloorIntegrity`,
`pretrain_summary`/freeze, `EnergyCalibrationLoss` + grouped training, an
equivariant bijection summary.

**Out of scope:** transformer/SSM summaries and variable `n_iid` (the next
milestone this unlocks); unknown-Σ multivariate (d>2); a full 5-method
cross-method sweep (optional, only if the bake-off lands cleanly); device-2c
soft-penalty/decoder-MI variants (excluded by the architectural-not-penalty
ethos, noted only for completeness).

---

## 7. Working principles carried in

- **`ρ` is not a prior;** coverage is frequentist and pointwise in θ₀.
- **Regularity is architectural, not penalty-based** — hence the soft-MI/decoder
  device is excluded; the admissible arms each enforce boundedness *by
  construction* (frozen feat / reparam-invariant loss / exact bijection).
- **Don't trust the draft uncritically** — device-1 was the draft's primary and
  was wrong; the bake-off is the corrective.
