# CD-SBI Stage B — learned-summary device bake-off: verdict

**Milestone:** M3.3′ (closes the Stage-B bake-off). **Target:** unknown-(μ, σ²)
Gaussian, θ = (log σ, μ), `n_iid = 10` replicates per observation X ∈ ℝ¹⁰,
`d_theta = 2` (`NormalUnknownMeanVar`). **Design of record:**
`2026-05-29-cd-sbi-stage-b-device-bakeoff-design.md` (the audit §1/§1.5 + the
co-adaptation reframing). **Branch:** the arms landed on `feat/musigma-m2`
(merged to `main`); this document is the verdict + manuscript-ready source.

---

## 1. The question

Stage A established that the CD-SBI pivot calibrates the unknown-(μ, σ²) target
when fed an **oracle** sufficient statistic `(log s², X̄)` (coverage error 0.026,
σ²/μ marginal-CD KS 0.020, loss at the entropy floor). Stage B asks the question
that matters for targets where the sufficient statistic is **unknown**:

> Can a **learned** summary `s_φ(X): ℝ¹⁰ → ℝ²` live inside the NF-MLE pivot
> machinery and calibrate as well as the oracle — and if not, what does it take?

Four devices were evaluated against the Stage-A oracle control on one shared
harness (M3.0): `SufficiencyRecovery` (Spearman of learned features vs the oracle
`(log s², X̄)`, monotone-invariant), per-coordinate calibration
(`MarginalCDRecovery`, `Coverage`, `JointMahalanobis`), and an arm-aware
`FloorIntegrity` (final loss vs the conditional-entropy floor).

---

## 2. Two failure modes, one axis

The audit isolated two distinct ways a learned summary fails, unified by a single
axis — **does the summary co-adapt with the pivot, and under what objective:**

- **The cheat (information collapse).** With `log_det_contrib = 0` the NF-MLE loss
  is `½‖r‖² − log|∂r/∂feat|`; minimizing it over `φ` drives the loss to `−∞` by
  collapsing `feat|θ` (low differential entropy) — the flow stretches the narrow
  feature back to `N(0,I)` (large `∂r/∂feat`). Root cause (Fisher–Neyman): NF-MLE
  on a *reducing* summary optimizes the collapsible `−log p(feat|θ)` instead of the
  fixed `−log p(X|θ) = −log p(feat|θ) − log p(X|feat)`; a learned reducer inflates
  the surviving term by discarding the most-expensive-to-encode information — here
  **σ²** (the second moment), keeping the linear **μ** (X̄).

- **The warp (frozen mismatch).** `SingleIndexMonotoneFlow` bakes in an inductive
  bias — the pivot is monotone in an **affine** index `z_k = a·θ_k + b·feat_k +
  off`, with `a,b` *constant* at coordinate 0 (no autoregressive context). This is
  exactly right for the oracle feature (`r*_σ = g(log s² − 2 log σ)` is affine in
  `log s²`). A learned feature need not be in that affine basis — predict-θ
  pretraining emits the posterior mean `E[log σ|X]`, a *shrunk, nonlinear* warp of
  `log s²` the constant-coefficient combiner cannot un-warp.

**The unifying claim:** co-adaptation is *not* the disease — an *exploitable
objective* is. Under a **cheat-free** objective, end-to-end co-adaptation is the
cure (gradient pressure forces the summary to emit coordinates the pivot can
calibrate). Freezing forfeits co-adaptation, so it cannot un-warp.

---

## 3. The four arms (cross-arm table)

All numbers are from the harness diagnostics on the (μ,σ²) target, seed 0. The
Stage-A oracle is the control. (`SufficiencyRecovery` reports best |Spearman| of
each oracle coordinate vs the learned features; coverage KS is the joint
Mahalanobis PIT vs χ²₂ at the central θ₀ unless noted.)

| Device | co-adapt? | objective | σ²-Spearman | μ-Spearman | calibration (σ) | floor | outcome |
|---|---|---|---|---|---|---|---|
| **Stage-A oracle** (control) | n/a (fixed `(log s², X̄)`) | NF-MLE | 1.00 (by constr.) | 1.00 | KS 0.020 ✓ | loss 0.904 ≈ H 0.921 | calibrates |
| **M2 naive** end-to-end | yes | NF-MLE (Jacobian cheat channel) | 0.47 | 0.996 | — | **loss −5.3 ≪ H** | **CHEATS** (σ collapse) |
| **Arm I-B** two-stage freeze | no (frozen after predict-θ) | predict-θ → NF-MLE | 0.97 | 0.9996 | **KS 0.10–0.11 ✗** | n/a | **WARPS** (σ miscalibrates) |
| **Arm II-A** energy | yes | calibration-invariant energy score (no Jacobian) | **0.014** | 0.999 | not reached | loss **1.83 ≈ floor 1.77** | **COLLAPSES σ** |
| **Arm I-A** invertible | yes | exact `−log p(X|θ)` (bijection) | **0.980** | 0.999 | **KS ≤ 0.077 ✓** | loss 14.47 > H(X\|θ) 13.66 | **RECOVERS σ** ✓ |

Notes per arm:

- **M2 naive** (`DeepSetsConditioner` + NF-MLE, `log_det_contrib=0` + feature
  normalization = the original "device-1"). The cheat the milestone was built to
  diagnose. Captured as a regression test
  (`tests/integration/test_mu_sigma_stage_b_smoke.py`, precedent:
  `tests/ablation/test_trained_folding.py`).

- **Arm I-B** (`pretrain_summary` predict-θ + `TwoStageCDSBIRunner`; plan
  `…-m3-1-arm-ib-two-stage-freeze.md`). σ² **is** recovered (the MSE objective
  retains it), but the frozen feature is the shrunk posterior mean — a nonlinear
  warp of `log s²` — so the fixed-sign affine coord-0 pivot cannot calibrate σ.
  **Documented contrast, not executed as a calibrating arm** (it cannot be rescued
  by tuning; the fix would be a feature-rectifier / richer combiner — see §5).

- **Arm II-A** (`EnergyCalibrationLoss` + `EnergyCDSBIRunner`, grouped-by-θ₀;
  `experiment=mu_sigma_stage_b_energy`). The headline finding: the energy score
  reaches its calibration floor (1.83 vs the analytic 1.77 for `N(0,I₂)`) **with σ
  entirely collapsed** (Spearman 0.014). **Per-θ₀ marginal calibration to `N(0,I)`
  is satisfiable by a μ-only summary** — the residual σ-miscalibration is a weak
  (~0.06) energy-score penalty, too weak a gradient to force the second moment.
  **Calibration ≠ informativeness.**

- **Arm I-A** (`AffineCouplingBijection` + `InvertibleSummaryConditioner` +
  `ExactDensityCDSBIRunner`, exact `−log p(X|θ)`; `experiment=
  mu_sigma_stage_b_exact`). The bijection preserves information (cannot discard σ)
  and the ancillary `A` is modelled θ-free `N(0,I₈)`, so the objective routes all
  θ-information into the pivot features `S` **and** the pivot-NF-MLE term
  co-adapts `S` to a calibratable coordinate. σ² recovered (0.980), calibrated
  (KS ≤ 0.077), and the loss sits 0.8 nats **above** the data-entropy floor H(X|θ)
  — a proper NLL cannot beat its entropy, ruling out a cheat by construction.

---

## 4. Verdict

**For a learned summary to calibrate a scale/nuisance parameter inside the CD-SBI
pivot machinery, the training objective must preserve information — an exact
change-of-variables bounded below by `H(X|θ)`.** The two cheaper objectives both
fail on the scale direction, for instructive and *different* reasons:

- A **calibration-only** objective (II-A) is satisfiable without sufficiency:
  matching `r(θ₀;X)|θ₀` to `N(0,I)` at each θ₀ does not force the summary to retain
  the scale information, because a μ-only summary already makes the marginal look
  Gaussian. **Calibration ≠ informativeness** — coverage can be (nearly) achieved
  while the confidence set for σ is uninformative.

- A **frozen** summary (I-B) optimized for a *different* objective (predict-θ)
  locks in a coordinate the fixed-architecture pivot cannot calibrate, even though
  it is sufficient.

The **exact-density invertible arm (I-A) is the keeper.** It is the heaviest, but
it is the only device that both retains σ-information (structurally) and lets the
pivot co-adapt to a calibratable coordinate.

This connects to the framework's deeper target: the bake-off shows calibration
alone is too weak a criterion for a learned summary — which is exactly why the
CD-SBI goal is **UMP(U)** confidence distributions (power/informativeness), not
mere coverage. The II-A finding is a concrete instance of that distinction and is
the natural seed for the §3.7 alternative-loss work (v5).

---

## 5. Open threads (handoff)

- **Permutation-equivariant bijection.** Arm I-A used a standard (non-equivariant)
  coupling flow — sufficient for the discriminating test at fixed `n_iid`, but the
  exchangeable-X-respecting, variable-`n_iid` generalization (the bridge to the
  sequence-model work) is deferred.
- **Can a power/sharpness term rescue II-A?** The energy/calibration objective
  could be augmented with an informativeness term (toward UMP(U)) or
  second-moment-capable summary inputs (`X_i²`) / a per-coordinate whitened energy
  term so the easy μ-axis cannot dominate the gradient. Untested.
- **Feature-rectifier for the pivot.** A per-coordinate monotone feature-rectifier
  `m_k(feat_k)` before the affine index would let the fixed-sign pivot consume
  warped coordinates — would rescue I-B and harden I-A/II-A against warp. Untested
  (a flow-architecture direction, not pursued once I-A cleanly recovered σ).
- **Manuscript integration.** This document is manuscript-ready source for a Stage-B
  section of `cd_sbi_v7.tex` (the audit §2 + the cross-arm table §3 + the verdict
  §4). Not yet integrated into the LaTeX.

---

## 6. Reproduction

- **Harness:** `SufficiencyRecovery` (`src/cdsbi/diagnostics/sufficiency_recovery.py`),
  `FloorIntegrity` (`…/floor_integrity.py`, arm-aware loss→floor map),
  `procedure.encode_fn` + `simulator.{oracle_summary, entropy_lower_bound,
  data_entropy_lower_bound}`.
- **Arm verdicts (intensive tests, seed 0):**
  - II-A: `tests/intensive/test_replicate_mu_sigma_stage_b_energy.py` (σ²-Spearman 0.014).
  - I-A: `tests/intensive/test_replicate_mu_sigma_stage_b_exact.py` (σ²-Spearman 0.980, KS ≤ 0.077).
  - M2 cheat regression: `tests/integration/test_mu_sigma_stage_b_smoke.py` (loss ≪ floor, σ-collapse).
- **Headline runs:** `python -m cdsbi.experiments.run experiment={mu_sigma_stage_b_energy,
  mu_sigma_stage_b_exact} seed=0` (recipes pinned in the experiment configs).
