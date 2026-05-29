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

## 2. Framing — a three-arm bake-off with Stage A as control

Deliverable = an **empirical verdict**: which device(s) let a *learned* summary
calibrate as well as the Stage-A oracle. The motivating use case is targets where
the sufficient statistic is **unknown**; (μ,σ²) is the controlled stress test
where we know the truth and can measure recovery. All arms are judged on the
**same** harness.

### 2.1 Shared evaluation harness (build once, reuse)

- **Calibration (existing diagnostics):** `PivotRMSE`, `MarginalPIT`, `Coverage`
  (2-D 3×3 grid), `MarginalCDRecovery` (σ²→χ², μ→Student-t), `JointMahalanobis`.
- **`SufficiencyRecovery` (new diagnostic):** Spearman (rank, monotone-invariant)
  of each learned feature vs the oracle `(log s², X̄)`; reports per-coordinate
  best-|Spearman| and Pearson (diagnostic). Gated like `MarginalCDRecovery`
  (no-ops unless the simulator exposes the oracle stat + a learned conditioner is
  present). Quantifies "did the summary keep the sufficient information."
- **`FloorIntegrity` (new check):** **arm-aware.**
  - NF-MLE arms (I-A, I-B): final loss must satisfy `loss > H − tol` (does NOT
    sink below the conditional-entropy floor). Reuses the §8.4 floor logic.
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

## 3. The three arms

### Arm I-B — two-stage freeze (pragmatic primary)

- **Stage 1.** Train DeepSets `s_φ:ℝ¹⁰→ℝ²` *alone* to predict θ: minimize
  `E‖g_ψ(s_φ(X)) − θ‖²` for a small read-out head `g_ψ`. The MSE is bounded below
  (≥0); the optimal θ-predictor is a function of the sufficient statistic, so for
  the Gaussian target this recovers `X̄`- and `s²`-equivalent features. Freeze
  `s_φ` (eval mode, `requires_grad=False`); discard `g_ψ`.
  - *Stage-1 objective choice:* **predict-θ MSE** (simplest bounded sufficiency
    proxy). InfoNCE/`I(feat;θ)` is a documented fallback if predict-θ under-recovers
    σ² (e.g. recovers a posterior-mean-of-θ that is insensitive to σ at the prior
    center). The plan instruments the stage-1 `SufficiencyRecovery` and escalates
    only if needed.
- **Stage 2.** The *unchanged* Stage-A pivot (`SingleIndexMonotoneFlow`) +
  NF-MLE on the **frozen** feat. Frozen feat ⇒ loss → fixed `H(feat|θ)`, no
  collapse channel; calibrates iff feat is sufficient.
- **New code:** `pretrain_summary(simulator, summary, head, config) -> frozen summary`
  utility; a `FrozenConditioner` wrapper (or freeze-in-place: zero the conditioner
  param group so `fit()`'s optimizer is a no-op on it). `fit()` already tolerates a
  conditioner with no trainable params.
- **Verdict signal:** preserves the d=2 monotone-pivot architecture exactly;
  generalizes (stage-1 finds sufficient features when the truth is unknown). Risk:
  not end-to-end; stage-1 features must be orientable by the fixed-sign pivot
  (surfaces as miscalibration if not — detectable).

### Arm II-A — calibration-invariant objective (principled; §3.7/v5 thread)

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

### Arm I-A — invertible summary (exactness benchmark)

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
  achievable with a learned, information-preserving summary." If I-B or II-A
  match I-A's calibration, they win on simplicity.

---

## 4. Decomposition & order (cheapest → heaviest)

Each numbered item is its own implementation plan; each lands a working unit + its
result on the shared harness. An arm may end the line if it cleanly wins, but the
default (per the bake-off goal) is to run all three and report.

1. **M3.0 — shared harness.** `SufficiencyRecovery` diagnostic + `FloorIntegrity`
   check (arm-aware) wired into the runner/index-row; convert the M2 smoke to a
   cheat-capture regression test. (No new arm yet — infrastructure for judging all
   arms uniformly.)
2. **M3.1 — Arm I-B (two-stage freeze).** `pretrain_summary` + freeze + Stage-2
   pivot; intensive replication on the harness.
3. **M3.2 — Arm II-A (calibration objective).** `EnergyCalibrationLoss` + grouped
   training path; intensive replication on the harness.
4. **M3.3 — Arm I-A (invertible benchmark).** Equivariant bijection summary +
   full-density loss; intensive replication on the harness.
5. **M3.4 — verdict + manuscript.** Cross-arm comparison table (each arm vs the
   oracle control on every harness metric), the verdict (which device(s)
   calibrate), and the audit written up for the draft.

---

## 5. Success criteria

- **Per arm:** matches the Stage-A control within the same tolerance bands
  (coverage_error_max ≤ 0.05; σ²/μ marginal-CD KS ≤ 0.06; JM ≥ 4/5 seeds at 2×
  floor) **AND** `SufficiencyRecovery` Spearman > 0.9 on both coordinates **AND**
  `FloorIntegrity` holds (arm-aware).
- **Milestone:** at least one device makes a *learned* summary meet the per-arm
  bar; the cross-arm table + verdict are produced; the cheat is captured as a
  regression test. A negative result for an arm (e.g. II-A high-variance, or I-B
  under-recovering σ² with predict-θ) is a *reportable finding*, not a failure of
  the milestone.

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
