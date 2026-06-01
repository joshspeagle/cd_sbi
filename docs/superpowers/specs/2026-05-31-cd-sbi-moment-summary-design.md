# CD-SBI: non-oracle confidence distributions via posterior-moment summaries

> **⚖️ VERDICT (2026-06-01) — explored, P1 built, line concluded.** P1 implemented &
> reviewed; the (μ,σ²) replication showed **central/bulk calibration near-oracle but a
> uniform (worst-θ₀) gap ~0.18→0.10-with-warp vs oracle 0.02** — the single-index
> ceiling (§16) + the learned summary's μ-entanglement (theory note **§17**). Outcome:
> **division of labor — CD/pivot for the regular/oracle regime; LF2I for the general /
> non-oracle / multimodal frontier.** P2 partially built (sign-model sim + non-monotone
> flow); P3/P4 not built. Kept as the design-of-record + the documented finding.

**Status:** explored → see §17 verdict. (Originally: design, brainstorm 2026-05-31.) **Supersedes & retires**
`2026-05-31-cd-sbi-s4-score-capture-design.md` — the Fisher-term / joint-training
approach is dropped: sequential regression dissolves the collapse problem (no Fisher
term, no §13.3 score bias, no likelihood-free-scoring phase), per theory note §15–16.
**Theory spine:** `docs/theory/2026-05-31-conditional-coverage-decomposition.md`
(§1–16). Every choice traces to a numbered result there.

---

## 1. Goal and why

**Goal.** A *non-oracle* CD-SBI procedure — no hand-provided sufficient statistic —
that is **asymptotically consistent**: valid (exact conditional coverage) and
efficient (set volume → oracle) for regular models, and *valid always* +
efficient-when-the-summary-suffices for symmetry-induced multimodal models, with
**correct disconnected confidence sets**.

**Why this shape (the lessons that retired the old spec).**
- *Validity is non-oracle for free* — any `(A)`-regular `d_θ`-dim summary + an
  NF-MLE pivot reaching `q≈p` covers exactly (Thm 1, Thm 5).
- *Efficiency = a (near-)sufficient summary* (Cor 12). The asymptotically-sufficient
  non-oracle device is a **regression estimate of a posterior moment** of `θ`
  (Fearnhead–Prangle / Waldo lineage): the posterior is minimal sufficient given a
  known prior, and its moments are estimable by plain supervised regression (§16).
- *Train it sequentially.* Fit the summary by regression, **freeze**, then calibrate
  the pivot by NF-MLE. Freezing structurally blocks the **entropy-floor cheat** (the
  M2 collapse, Prop 18): Stage-2's NF-MLE loss cannot game a summary it cannot touch.
  This is *not* a sufficiency guarantee — `E[m(θ)|X]` is only *asymptotically*
  sufficient (BvM); the finite-`n` efficiency gap is **measured** (`FisherRecovery`),
  not structural. Net: freezing buys validity-safety; efficiency is earned and
  checked. This replaces the entire Fisher-term apparatus.
- *Multimodality is an R1 question, not a bottleneck question* (Thm 15): an R1
  (monotone-in-θ) pivot is a homeomorphism, forcing connected sets; the *theory* says
  dropping R1 permits the correct disconnected sets with validity untouched. **Note
  this is free in theory but NOT in code:** no R1-off pivot flow and no
  disconnected-set extractor exist yet — both are builds (§4, C-items).

## 2. The construction (one method, two knobs)

    Stage 1 (summary):   h_φ(X) ≈ E[ m(θ) | X ]      — L² regression, then FREEZE
    Stage 2 (pivot):     r_ψ(θ; h_φ(X))               — NF-MLE; R1 on or off
    Confidence set:      C_α(X) = { θ : ‖r_ψ(θ; h_φ(X))‖² ≤ χ²_{d_θ,α} }

**Two knobs:**
- **Regression target `m(θ)`** — a **`d_θ`-dimensional** map (so the
  dimension-preserving NF-MLE pivot stays clean): `m(θ)=θ` (posterior mean) for
  regular models; a `d_θ`-dim **invariant** when the mean is *uninformative* (the
  modality signal: low regression R² on `θ`) — e.g. **coordinatewise `θ²`** for a
  product sign-symmetry (the sign model: `d_θ=1`, `m(θ)=θ²`). *Caveat:* the **full**
  2nd-moment tensor `θ^⊗2` has dim `> d_θ` for `d_θ>1` and is **not** an allowed
  target — only `d_θ`-dim invariants are. Finding the right `d_θ`-dim invariant for a
  general (non-product) symmetry is part of the open "which target" rule (§5 P2/P4);
  symmetries with no `d_θ`-dim sufficient invariant are the LF2I boundary (§9).
  **Selected by regression informativeness.**
- **R1 on/off** — on (monotone flow `SingleIndexMonotoneFlow`) ⟹ connected sets, for
  regular/unimodal targets; off ⟹ disconnected sets, for symmetry-multimodal targets.
  *R1-off is a NEW flow*, not a reuse: `SingleIndexMonotoneFlow` is monotone in θ **by
  construction** (`softplus>0, G'>0`), so it cannot represent `r(θ)=r(−θ)`; and
  `MAFAdapter` is a density estimator (`forward()` raises — it has no `(r, log|∂_T r|)`
  pivot contract). The R1-off pivot keeps the **monotone feature channel** (C1: `∂r/∂feat`
  constant-sign, for the NF-MLE density) but replaces the monotone θ-channel with an
  **unconstrained MLP `a_k(θ, ctx)`** — non-monotone in θ. The sign-model analytic
  pivot `r=√n(X̄−θ²)/σ` (monotone in `X̄`, parabolic in θ) is the target instance. See
  C1 in §4.

**The regular corner** (`m(θ)=θ`, R1 on) is the asymptotic-consistency base case;
**the multimodal corner** (`m(θ)=`higher moment, R1 off) is the sign-model class.

## 3. Theory spine — the asymptotic-consistency theorem

**Theorem (assembled).** For regular models (DQM, identifiable, iid replicates
`n→∞`), with `h` a consistent regression estimate of `E[m(θ)|X]` for an
asymptotically-sufficient `m` (the posterior mean under BvM), and `r̂` an NF-MLE
pivot on `(θ, h(X))`, as `(n, N, capacity)→∞`:
- **validity:** `sup_θ |coverage−α| → 0` (Thm 1 + NLE consistency `q̂→p` + Thm 5
  bound `√((L−H̄)/2)→0`);
- **efficiency:** `½ log(det I_X/det I_h) → 0` (Cor 12 + BvM asymptotic sufficiency,
  `I_h→I_X`).
Pieces are in hand or classical (Thm 1/5, Cor 12 ours; NLE consistency, LAN/BvM
cited). Multimodal: validity holds for any summary (Thm 1); efficiency holds when a
`d_θ`-dim sufficient `m(θ)` exists (sign model: `m(θ)=θ²`).

**Scope boundary (be explicit).** The NF-MLE pivot is dimension-preserving, so the
summary is `d_θ`-dimensional. This covers regular models and **symmetry-induced
multimodality whose sufficient reduction is `d_θ`-dim** (sign model). Genuinely rich
multimodality (minimal sufficient dim `> d_θ`, e.g. many unrelated modes) **exceeds
the clean pivot** and requires the LF2I extension (richer summary + Neyman-calibrated
statistic) — out of scope here, §9.

## 4. Architecture & components

| Component | Choice | New? |
|---|---|---|
| Summary `h_φ` | `MomentRegressionConditioner` — DeepSets backbone → `d_θ` outputs, trained by L² to predict `m(θ)`; **frozen** before Stage 2 | **new** (DeepSets reused) |
| Target `m(θ)` | config: `theta` (mean) / coordinatewise `theta_sq` (`d_θ`-dim) / learned `d_θ`-dim invariant | **new** (config) |
| Pivot (R1 on) | `SingleIndexMonotoneFlow` | reuse |
| **Pivot (R1 off)** [C1] | **`NonMonotonePivotFlow`** — monotone feature channel (C1 intact) + unconstrained MLP `a_k(θ,ctx)` θ-channel (non-monotone). NOT `MAFAdapter` (no pivot `forward`), NOT de-monotonized `SingleIndexMonotoneFlow` (monotone by construction). Advertises `monotonicity_guarantees = {R2}` (feature-monotone, no R1); the `moment_sign` experiment sets `method.allow_ablation: true` so `NFMLELoss.check_guarantees` (requires `{R1,R2}`) passes — mirrors how `JointUMNNFlow` runs | **NEW (load-bearing for P2)** |
| Runner | `TwoStageCDSBIRunner` — regression → freeze → NF-MLE | **new** (extends the Arm-I-B *plan*, not landed code; generalizes target to `m(θ)` + adds the R1-off path) |
| **Disconnected-set extractor** [C2] | for d_θ=1: enumerate `{θ:‖r‖²≤χ²}` connected components on a θ-grid. Both existing paths return a single interval (`_ray_sample_set_boundary` assumes **radial convexity**; `_confidence_set_1d` does one center + one bisection per side), so the extractor is selected at the **procedure level** for the R1-off arm — a flag (`disconnected=True`) or a distinct procedure subtype, NOT just a standalone module | **NEW (load-bearing for P2)** |
| Diagnostics | `Coverage` (pointwise, set-free — works for disconnected ✓), `SufficiencyRecovery` (reuse); `FisherRecovery` (true-score `det I_h/I_X`) and `Disconnectedness` (connected-components of `‖r(θ;X)‖²−χ²` on a θ-grid, **pivot-direct, no set construction**) | **both NEW** |

**Why I-B's failure doesn't recur.** Arm I-B ("two-stage predict-θ → warps") froze a
summary *and* used a fixed affine pivot, which could not absorb the summary's
shrinkage. Validity (Thm 1) holds for any frozen summary **given a flexible pivot**;
Stage 2 here is a full NF-MLE flow, so the warp does not occur. (Note: this is
*two-stage with no efficiency term*, not "M2 with a term swapped" — M2 was joint
end-to-end NF-MLE. The P1 "beat M2" comparison must use M2's *own* batching, §5 P1.)

**Config-dispatch caution (cb1e08e guard).** `run.py::_build_method` builds the
cd_sbi flow via `method.flow`; the R1 on/off choice (`SingleIndexMonotoneFlow` vs
`NonMonotonePivotFlow`) must thread through `method.flow`, and the runner must be the
two-stage one. Verify `model.pt arch_metadata.flow_class` after a run.

## 5. Phases

### Phase 1 — Regular consistency on (μ,σ²)
`m(θ)=θ` (mean), R1 on, sequential. Establish: valid (coverage-err ≤ 0.05),
efficient (`det I_h/I_X ≥ 0.8`, **measured** — not structural), and **no collapse**
— the frozen regression summary retains σ (σ²-Spearman ≥ 0.9) with **no** Fisher
term (freezing blocks the entropy cheat; cf. §1). Head-to-head vs oracle
(`SufficientStatConditioner`) and vs a **freshly-run** joint-NF-MLE M2 collapse
baseline (M2 exists only as a smoke test, not a sweep arm — recreate it, matched
batching): sequential should match oracle and beat M2.

### Phase 2 — Symmetry-multimodal (the headline new capability)
Sign model `X̄|θ~N(θ²,σ²/n)` (`d_θ=1`; a 2-D product-symmetry analogue optional).
The **modality detector**: Stage-1 regression of `θ` returns low R² ⟹ switch target
to `θ²` and **R1 off** (the new `NonMonotonePivotFlow`). Establish: valid (KS,
coverage-err ≤ 0.05 via the set-free `Coverage`), **correctly disconnected** (the
new pivot-direct `Disconnectedness` diagnostic counts ≥2 components; the
disconnected-set extractor recovers the two intervals), efficient (recovers the
sufficient stat, §16 witness: 2nd-moment R²≈0.99). *Build dependency:* this phase
needs C1 (the R1-off flow) **and** C2 (the disconnected-set extractor) — both new.
This is the capability the closed-form-CD framework was wrongly thought to lack (§15).

### Phase 3 — High-d non-oracle on d=5 Bartlett (P2-independent)
`m(θ)=θ` (the 5 log-Cholesky+mean coords), R1 on (`SingleIndexMonotoneFlow`,
`theta_signs` from the KR order diag→offdiag→mean; R1-realizable here by MLR /
§13.5), vs oracle `BartlettSummaryConditioner`. Depends only on P1 machinery (not P2,
not P4's detector). **Decisive test (a hypothesis, not a foregone conclusion):** does
regressing `θ` route the **quadratic covariance** params that I-A's invertibility
failed to route (N3: `A₂₂` never routed)? Regression *directly targets* the Cholesky
entries — a genuinely different mechanism — **but the named failure mode is that the
DeepSets backbone's SGD may not discover the cross-moment (scatter-matrix) features**
needed to predict `E[ℓ₂₁|X], E[ℓ₂₂|X]` from raw `X∈ℝ²⁰` (the quadratic is in-class
for mean-pooled DeepSets but finding it is the same kind of routing question N3
raised, now at the regression stage). Success: coverage + per-direction
`eig(I_h I_X^{-1}) → 1` across all 5, approaching oracle. A negative result localizes
the failure to regression-feature-learning — itself a sharp finding.

### Phase 4 — Knob-selection rule + uniform coverage (refinements)
(i) A principled **target/R1 selector** from simulations (regression-R² per moment
order ⟹ detect modality, pick `m` and R1); (ii) **S2b uniform-coverage reweighting**
(Thms 8–9, §13.4) for the pointwise (μ₂-style) tail — independent of the summary,
applied as an add-on. Both are refinements; neither gates P1–P3.

## 6. File layout

```
src/cdsbi/
  conditioners/moment_regression.py   # MomentRegressionConditioner (+ target m(θ))
  methods/cd_sbi_two_stage.py         # TwoStageCDSBIRunner (regress→freeze→NF-MLE)
  flows/non_monotone_pivot.py         # [C1] NonMonotonePivotFlow (R1-off, monotone-in-feat)
  confidence_set/disconnected.py      # [C2] connected-component set extractor (R1-off path)
  diagnostics/disconnectedness.py     # Disconnectedness (pivot-direct, θ-grid components)
  diagnostics/fisher_recovery.py      # FisherRecovery (true-score det I_h/I_X) — NEW (was plan-only)
  simulators/sign_normal_1d.py        # P2 sign model X̄|θ~N(θ²,σ²/n)
configs/
  method/cd_sbi_moment.yaml           # target m(θ), R1 on/off (method.flow!), two-stage params
  experiment/{moment_mu_sigma,moment_sign,moment_mu_cov}.yaml
tests/
  unit/test_moment_regression.py      # recovers E[m(θ)|X]; sign-model moment-order
  unit/test_non_monotone_pivot.py     # [C1] Gaussianizes (KS) + non-monotone-in-θ + monotone-in-feat
  unit/test_disconnected_extractor.py # [C2] recovers two intervals on the sign model
  integration/test_two_stage_runner.py # freeze works (params unchanged); no-collapse; valid procedure
  intensive/test_replicate_moment_{mu_sigma,sign,mu_cov}.py
```
`run.py::_build_method` extended (mind the cb1e08e flow guard, §4). `FisherRecovery`
is **NEW** — the retired spec only planned it; no `fisher_recovery.py` exists yet.

## 7. Testing strategy (TDD)

- **Unit:** `MomentRegressionConditioner` recovers `E[m(θ)|X]` on a synthetic; the
  sign-model moment-order witness (1st-moment R²≈0, 2nd-moment R²≈0.99 — already in
  `tests/theory/`); `Disconnectedness` flags two-interval vs single-interval sets.
- **Integration:** Stage-1 freeze actually freezes (summary params unchanged in
  Stage 2); the two-stage runner yields a valid `PivotBasedProcedure`; the frozen
  regression summary does **not** collapse (σ²-Spearman ≥ 0.9), unlike joint M2.
- **Intensive:** P1 (μσ vs oracle/M2), P2 (sign model — valid+disconnected+efficient),
  P3 (d=5 vs oracle).

## 8. Open risks (surface, don't hide)

- **The "which moment" rule** is the real open piece — regression-R² as a modality
  detector is plausible but unproven; P2/P4 build and test it. A bad rule picks an
  uninformative target ⟹ valid-but-wide sets (never invalid — Thm 1).
- **Per-coordinate / mixed modality is UNSOLVED** — a single global `m(θ)` + one
  R1-on/off switch cannot express "coord 1 wants `θ`, coord 2 wants `θ²`/R1-off." In
  scope: *fully* symmetric targets (all coords same regime). Mixed-modality (some
  coords multimodal, others not) is **out of scope**; the detector must flag it.
- **R1-off pivot calibration** — the new `NonMonotonePivotFlow` (C1) must reach
  `q≈p`; verify it Gaussianizes (KS) AND is genuinely non-monotone-in-θ on the sign
  model, while staying monotone-in-feature (C1).
- **High-order moment-regression variance** — low orders solid, high orders noisy;
  graceful degradation, not a cliff (§16). Cap the target order.
- **Genuine high-modality exits scope** (sufficient dim > `d_θ`) → LF2I (§9); the
  detector should *flag* this case (no `d_θ`-dim moment suffices), not fail silently.
- **BvM-fails-under-symmetry** — for the sign model the posterior stays bimodal as
  `n→∞`, so the *mean* is never sufficient; this is by design why P2 switches target.

## 9. Scope boundary / out of scope

- **Genuine rich multimodality (sufficient dim > `d_θ`):** the LF2I extension —
  richer (NLE-level) representation + Neyman-calibrated LRT statistic (global, *not*
  the local score — why LF2I-Score broke on SLCP). Reuses the existing LF2I-Score
  code, not this pivot framework. A separate spec.
- R1-realizability theory (S5, §13.5 — shape, not coverage); image/sequence targets.

## 10. What this retires (from the S4 score-capture spec)

The Fisher-information efficiency term, score control (§13.3), the A-vs-B flow-score
bake-off, the `λ` frontier, and the likelihood-free-scoring phase — **all moot**: the
summary is supervised regression (no score, no likelihood), the pivot is plain NF-MLE,
and freezing gives anti-collapse structurally. The only carry-over is `FisherRecovery`
(now a pure *diagnostic*, not a training signal).
