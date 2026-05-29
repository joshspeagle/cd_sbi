# CD-SBI — Unknown-(μ, σ²) Gaussian + Learned Summary Network — Design Spec

**Date:** 2026-05-29
**Status:** Draft — pending user sign-off before implementation-plan handoff.
**Source brainstorm:** Live session 2026-05-29.

## Problem statement

Every experimental target in the repo so far (§8.1–§8.4) infers **location
parameters with known/fixed scale** (or a single rate). The framework has
never been tested on:

1. a **scale / nuisance parameter** (σ² alongside the location μ), with the
   non-Gaussian marginal CD it implies (Student-t for μ, χ² for σ²); or
2. a **learned summary network** — every run to date uses either the
   `Identity` conditioner (raw X) or the frozen closed-form `frozen_sum`
   reduction. There is no trainable `Conditioner`, and `CDSBIRunner.fit`
   optimizes only `self.flow.parameters()`.

The unknown-(μ, σ²) Gaussian is the minimal target that forces both, and it is
the natural bridge to harder unknown-Σ multivariate problems. Critically, it
has a **known closed-form joint pivot and known marginal CDs**, so the abstract
question "can a learned summary live inside the calibration machinery?" becomes
empirically measurable.

This milestone is a two-stage investigation (A → B in one spec):

- **Stage A** establishes the non-additive autoregressive **pivot + CD**
  machinery on (μ, σ²) using the *oracle* sufficient statistic `(X̄, s²)`.
- **Stage B** replaces the oracle with a *learned* summary `s_φ(X): ℝⁿ → ℝ²`,
  trained jointly, and uses Stage A as the control to detect/diagnose the
  summary-learning failure mode and the regularity device that fixes it.

## Target definition

- Data: `X = (X₁, …, X_{n_iid})` iid `N(μ, σ²)`, **`n_iid = 10`** (heavy-tailed
  `t₉`, not asymptotically normal). One observation `X ∈ ℝ¹⁰` per parameter draw.
- Parameter: **`θ = (log σ, μ)`** (ordering deliberate — see pivot). `log σ`
  is unconstrained (clean for UMNN integration + uniform prior). `d_θ = 2`.
- Priors: `μ ~ U[−5, 5]`; `log σ ~ U[log 0.3, log 3]` (~10× scale range).
- SBI budget: `n_train ≈ 10k` simulated `(θ, X)` pairs (the existing
  `training.n_train`); θ shape `(n_train, 2)`, X shape `(n_train, 10)` — exactly
  the §8.4 `(n_train, n_iid)` layout with `n_iid` 5 → 10.

### The exact closed-form joint pivot (validation anchor)

Sufficient statistic `(X̄, s²)`; under normality `X̄ ⊥ s²` (Basu), giving an
**exact** `𝒩(0, I₂)` pivot at the true θ₀:

- `r*_σ(θ; X) = Φ⁻¹( F_{χ²_{n_iid−1}}( (n_iid−1) s² / σ² ) )`  — uses `(σ², s²)`
- `r*_μ(θ; X) = √n_iid · (X̄ − μ) / σ`  — uses `(μ, σ; X̄)`

Properties that drive the architecture:

- **Autoregressive / triangular, forced order σ → μ.** `r*_σ` is self-contained;
  `r*_μ` depends on `log σ` (the *earlier* coordinate). The scale must come first.
- **Non-additive.** `r*_μ` couples `μ` and `σ` multiplicatively (`/σ`), so the
  additive `TriangularAdditiveFlow` (§8.2/8.3, `r = a(θ) − b(X)`) **cannot**
  represent it. Requires a doubly-monotone (UMNN-integrand) construction.
- **Monotone as required:** `r*_μ` ↓ in μ (R1), ↑ in X̄ (R2); `r*_σ` ↓ in log σ
  (R1), ↑ in s² (R2).
- **Marginals are textbook CDs:** μ → Student-`t_{n_iid−1}`, σ² → χ²-based —
  the Schweder–Hjort UMP(U) confidence distributions, recoverable and checkable.

Note `r*_μ` uses the *parameter* σ (a component of θ), not the data `s²` — the
pivot is a function of `(θ, X)`, so this is legitimate and clean.

---

## Stage A — pivot + CD with the oracle summary

### A.1 Oracle summary conditioner

`SufficientStatConditioner` — frozen, zero-parameter `Conditioner` mapping
`X ∈ ℝ^{n_iid} → features = (s², X̄) ∈ ℝ²` (order paired to θ's (log σ, μ)).
`encode(X) → (features, log_det_contrib)`. The `(X̄, s²)` reduction's volume
factor is **θ-independent** (the discarded `p(X | X̄, s²)` is θ-free by
sufficiency), so it does not affect the NF-MLE argmin; we set
`log_det_contrib` to a constant (chosen so the reported loss is comparable to
the conditional-entropy floor, mirroring §8.4's cosmetic `½ log n`). The exact
constant is derived in the plan; getting it "wrong" only shifts the loss scale,
not the optimum.

### A.2 The non-additive autoregressive flow

New flow `TriangularDoublyMonotoneFlow(d)` — the natural generalization that
subsumes both existing multivariate pieces:

- Autoregressive/triangular like `TriangularAdditiveFlow`: coordinate `k`'s
  pivot `r_k` is conditioned on `(θ_{<k}, features_{<k})`.
- Each coordinate is **doubly-monotone** like `DoublyMonotoneUMNN` (§6.1 form 2):
  `r_k = b_k(feat_k) + ∫_{θ_ref}^{θ_k} softplus(α_k(t) + β_k(feat_k)) dt`, where
  `b_k, β_k` are monotone-increasing UMNNs of `feat_k` and `α_k`, `β_k`, `b_k`
  are additionally conditioned on `(θ_{<k}, features_{<k})`. This gives R1
  (`∂r_k/∂θ_k = softplus(·) > 0`) and R2 (`∂r_k/∂feat_k > 0`) architecturally,
  while being **non-additive** (the multiplicative coupling the truth needs).
- `monotonicity_guarantees = {R1, R2}`.
- For (μ, σ²): coord 0 = `r_σ(log σ; s²)`; coord 1 = `r_μ(μ; X̄)` conditioned on
  `log σ` (and, allowed but unused-by-truth, `s²`). The flow class contains the
  closed-form truth, so Stage A should recover it.

This is the first genuinely **non-additive multivariate** flow in the repo;
`TriangularAdditiveFlow` is its additive special case and `DoublyMonotoneUMNN`
its `d=1` special case (note this in the flow docstring).

### A.3 Stage-A validation (all against the closed-form truth)

- **PivotRMSE** vs `r*` (simulator exposes `r_star(θ, X)`).
- **MarginalPIT** per coordinate + **JointMahalanobis** (`‖r‖² ~ χ²₂`).
- **Coverage** on a 2-D θ₀ grid over `(μ, log σ)` (product grid, e.g. 3×3 =
  9 points; replaces the fixed 1-D 5-point grid — the documented "d≤2 needs a
  grid strategy" caveat, handled here by a small product grid).
- **Marginal-CD recovery** (new diagnostic, `MarginalCDRecovery`) — note the
  two marginals recover *asymmetrically*:
  - **σ²-CD is direct.** `r_σ` involves only `(σ², s²)` (no μ), so the trained
    σ²-marginal CD reads off `r_σ` and should match the χ²-based CD via KS.
  - **μ-CD requires marginalizing the nuisance.** The *joint* pivot's μ-component
    `r_μ = √n(X̄−μ)/σ` uses the true σ; the practically-useful **marginal CD for
    μ is Student-`t_{n_iid−1}`**, obtained by integrating the joint confidence
    density over the σ direction (the standard CD-marginalization), then KS-
    comparing to `t_{n_iid−1}`. This is the more delicate check — treat it as a
    secondary validation and pin the exact marginalization in the plan; do not
    assume the t-CD falls trivially out of `r_μ`.
- Budget sweep + (optionally) cross-method baselines — see Scope.

`JacobianRecovery` does **not** apply (the truth Jacobian is non-constant —
`r*` is non-affine in θ); it no-ops as it already does when `r_star_jacobian`
is absent.

---

## Stage B — learned summary network

### B.1 The summary architecture: permutation-invariant set encoder

The `n_iid` observations are **exchangeable** (iid), so the summary must be
permutation-invariant over them. `DeepSetsConditioner` — a trainable
`Conditioner` (an `nn.Module`):

`s_φ(X) = ρ( (1/n) Σ_i φ(X_i) )`, `φ: ℝ → ℝ^h`, `ρ: ℝ^h → ℝ²`.

DeepSets with mean-pooling can represent `(X̄, X̄²) → (X̄, s²)` **exactly**, so a
sufficient summary is within its class — the question is whether training finds
it. (A self-attention / transformer encoder over the set is the richer variant
that generalizes to *variable* `n_iid` and is the bridge to the originally-
discussed higher-d sequence-model work; noted as a follow-on, **out of scope
here** — `n_iid` is fixed at 10, DeepSets suffices and is the minimal sufficient
choice.)

### B.2 `fit()` change

`CDSBIRunner.fit` currently optimizes only `self.flow.parameters()`. Extend it
to also include `self.conditioner.parameters()` when the conditioner is an
`nn.Module` with parameters (no-op for `Identity` / frozen conditioners). This
is the one change to existing training code.

### B.3 The open question + candidate regularity devices

When `s_φ: ℝ¹⁰ → ℝ²` is learned and non-square, `log|∂s/∂X|` is not a genuine
Jacobian, so the NF-MLE change-of-variables is ill-posed and the network can
**cheat** — collapse/rescale features to inflate apparent density and drive the
loss *below* the conditional-entropy floor (an information-destroying analog of
the §3.5 (R2) folding pathology). Resolving this is the milestone's
contribution. Candidate devices, to be evaluated with Stage A as control:

1. **Primary — drop the summary log-det + feature normalization.** Set the
   conditioner's `log_det_contrib = 0` (the summary is a feature map, not part
   of the change-of-variables) and normalize features (e.g. running
   standardization) to remove the scale/collapse cheat. Calibration is then
   enforced through the pivot's `∂r/∂feat` and the marginal-PIT/NF-MLE objective
   alone; an *insufficient* summary leaves residual θ-information that makes
   `r(θ; s_φ)` non-standard-normal across θ, which the calibration diagnostics
   detect. (Simplest; matches mainstream SBI summary-net practice.)
2. **Principled fallback — invertible (dimension-preserving) summary.** Make
   `s_φ: ℝ¹⁰ → ℝ¹⁰` a normalizing flow (genuine `log|∂s/∂X|`), then take the
   first 2 coordinates as features. Exact change-of-variables, no cheating,
   heavier net. Use if (1) cheats.
3. **Two-stage freeze.** Pre-train `s_φ` to sufficiency (predict θ / InfoMax /
   end-to-end then freeze), then train the pivot on frozen features. Sidesteps
   joint ill-posedness; a comparison point.

The plan implements (1) first, instruments for the cheat, and escalates to (2)
only if observed. The deliverable is the empirical verdict + whichever device
makes the learned summary calibrate.

### B.4 Stage-B validation (Stage A is the control)

- **Sufficiency recovery:** correlate/regress learned `s_φ(X)` against the
  oracle `(X̄, s²)` — does it recover a smooth reparameterization of the
  sufficient statistic?
- **Floor integrity:** does the final NF-MLE loss respect the conditional-
  entropy floor, or dip below it (the cheat)? (Reuse the §8.4 floor logic.)
- **Calibration:** the same PivotRMSE / MarginalPIT / Coverage / Marginal-CD
  diagnostics as Stage A — does the learned-summary pivot calibrate as well as
  the oracle?

---

## Scope

**In scope:**
- `NormalUnknownMeanVar` simulator (θ=(log σ, μ), n_iid=10, closed-form
  `r_star`, `log_prob`, MC `entropy_lower_bound`).
- `SufficientStatConditioner` (oracle, Stage A) + `DeepSetsConditioner`
  (learned, Stage B), both implementing the `Conditioner` protocol.
- `TriangularDoublyMonotoneFlow` (non-additive autoregressive, d=2).
- `CDSBIRunner.fit` extension to train conditioner params.
- `MarginalCDRecovery` diagnostic; 2-D θ₀-grid coverage; Stage-B sufficiency +
  floor-integrity checks.
- Hydra configs (target / flow / conditioner / experiment) + intensive
  replication tests for Stage A and Stage B.

**Out of scope:**
- Self-attention / transformer / SSM summary encoders and variable `n_iid` (the
  originally-discussed higher-d sequence-model work) — DeepSets at fixed n_iid
  here; that is the *next* milestone this unlocks.
- Unknown-Σ multivariate (d > 2) — this (μ, σ²) case is the d=2 stepping stone.
- A full 5-method cross-method sweep is **optional/secondary** (see below).

**Cross-method baselines (secondary, optional):** a light NPE/NLE/NRE/LF2I
comparison on the same target (CDSBI calibrated vs baselines miscalibrated),
mirroring §8.x. Included only if Stage A + Stage B land cleanly; the milestone's
primary value is the non-additive pivot + the learned-summary investigation,
not a benchmark.

## Decomposition (milestones for the plan)

- **M0 — target + oracle + flow (Stage A core):** `NormalUnknownMeanVar`,
  `SufficientStatConditioner`, `TriangularDoublyMonotoneFlow`, wire into `run`,
  unit tests + a single-run smoke. Validate PivotRMSE/PIT against `r*`.
- **M1 — Stage A diagnostics + replication:** `MarginalCDRecovery`, 2-D coverage
  grid, intensive replication test (Stage A reaches the floor + calibrates).
- **M2 — Stage B summary + fit change:** `DeepSetsConditioner`, `fit()` optimizer
  extension, device (1), sufficiency + floor-integrity instrumentation.
- **M3 — Stage B investigation + (if needed) device (2):** run joint training,
  diagnose cheat vs calibrate, escalate to invertible summary if needed; record
  the verdict.
- **M4 — (optional) cross-method baselines + manuscript section.**

## Risks / open questions

- **The Stage-B cheat is the central risk and the central question** — device
  (1) may not suffice; (2) is the principled fallback but heavier. The spec
  deliberately treats this as an experiment with a control, not a foregone
  conclusion.
- **Oracle conditioner log-det constant.** Must be θ-independent (it is, by
  sufficiency); a wrong constant only shifts the loss scale (affects the
  entropy-floor comparison, not calibration). Derive carefully in the plan.
- **KR ordering σ → μ is forced by the truth** in Stage A; whether a *learned*
  flow discovers it (or whether order-randomization matters) is a smaller open
  question — fixed order here, randomization deferred with the sequence-model work.
- **2-D coverage θ₀-grid** must cover the heterogeneous (location, scale) box
  adequately; a 3×3 product grid is the starting point, revisit if coverage
  estimates are noisy at the scale extremes.

## Acceptance criteria

- Stage A: trained `TriangularDoublyMonotoneFlow` + oracle summary recovers `r*`
  (PivotRMSE at noise floor), all calibration diagnostics at their floors, final
  NF-MLE loss at the conditional-entropy floor, and the marginal CDs match
  Student-t / χ² within KS noise.
- Stage B: a clear, instrumented verdict on the learned summary — either it
  recovers a sufficient statistic and calibrates as well as the oracle (with the
  working device), or the cheat is demonstrated and the principled device (2)
  fixes it. Either outcome is a valid, documented result.
- All new code unit-tested; Stage A + Stage B replication tests under the
  `intensive` marker; fast suite green.
