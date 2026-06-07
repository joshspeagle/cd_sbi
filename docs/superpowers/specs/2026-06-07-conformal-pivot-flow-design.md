# Conformal Pivot Flow (CPF) — design spec

**Status:** Reviewed (Round 1: validity + implementation + novelty; Round 2: correctness +
buildability) and revised. **Plan-ready.** Working name "Conformal Pivot Flow / CPF" provisional.
**Date:** 2026-06-07
**Track:** Route 2 of the "most-general LF2I" line — an amortized, coherent, sampleable,
conditionally-valid frequentist confidence distribution, framed as recalibrating a base posterior.
**Home:** `cd_sbi` (fit verdict: workable host; see §7).

> Output of the multi-agent design review (4 first-principles designs × 2 red-team rounds), then two
> review rounds on this spec. Seed for a TDD plan. **No code/tests until a plan is approved.**
>
> **Honest framing (see §2, §11).** The *validity layer* is **adopted wholesale** from
> TRUST/TRUST++ + CP4SBI; the *coherence* result is textbook rearrangement. CPF's increment is a
> learned **flow-pivot** statistic, a coherent **sampleable CD** readout, a **firewall discipline**
> (frozen read-only sampler), and one **finding** (a conformal floor converts CD-SBI's
> PKD-insufficiency from a validity break into a width cost). On current evidence CPF is **most
> defensible as the worked example anchoring the three-axis "validity/sufficiency/power" map paper**,
> graduating to a standalone method only if the M1b make-or-break clears both bars (§11 GATE 2).

---

## 1. Goal

Given `x`, output a **confidence distribution** `H(θ|x)`: (i) **valid** — every `C_α(x)` has
*conditional* (per-θ₀) coverage to the attainable ceiling (§6); (ii) **coherent** — `{C_α}` nested in
α by construction; (iii) **amortized**; (iv) **sampleable**.

**Recalibration framing.** A base posterior `q(θ|x)` (NPE/NLE/NPSE) is samplable, its score
evaluable, but not frequentist-calibrated (Hermans et al. 2022). CPF recalibrates it. The base engine
is a power/warm-start input, **never a source of validity**; the core can train its operator directly
from the simulator (§4.1).

**Primitives.** (i) Simulator `θ₀~π`, `x~p(·|θ₀)`. (ii) Optional base engine emitting `{θ_i}~q(θ|x)`
and `∇_θ log q(θ|x)` (density may be unavailable). (iii) Known prior π.

**Standing assumption.** `T(θ;x)` is **continuous (atom-free)** in its sampling law at each θ (makes
the PIT yield `U[0,1]`). Discrete/mixed simulators are out of scope for the per-θ₀ exactness claim.

---

## 2. Position & honest novelty ledger

| Prior work | Owns | Relation |
|---|---|---|
| **TRUST/TRUST++** (arXiv:2411.19368, Cabezas et al. 2024) — **co-top threat; ADOPTED** | conditional-CDF `Ĥ(·|θ)` of a statistic by partition quantile regression; finite-sample **local** + asymptotic **conditional** coverage | **is** CPF's §4.3 minus the flow. Adopted wholesale; not claimed. |
| **CP4SBI** (arXiv:2508.17077, 2025) — **co-top threat; ADOPTED** | local/Mondrian (LoCart) split-conformal of SBI **credible sets** via conditional-CDF score | conformal floor adopted; CPF emits a *sampleable CD* with a *learned pivot* statistic. |
| **CD-SBI** (this repo) — **the method CPF improves** | by-construction pivot `r(θ;x)~N(0,I)`; UMPU CD in the regular regime | no finite-sample floor; PKD-fragile (repo's d=5 μ₂ miscalibration; "calibration≠sufficiency"). CPF's **finding**: conformal floor → insufficiency is a width cost, not a validity break (must be *demonstrated*, §8). |
| Cal-PIT/LADaR (Dey 2022) | reshapes a *predictive* density (Y\|X) | prediction/marginal; CPF = parameter CD, per-θ₀. |
| LF2I/WALDO/LF2I-Score | statistic + per-α critical values; per-α *sets* | CPF = one coherent sampleable CD; learned-pivot statistic. |
| CFV-Galichon rearrangement; MCQRNN; super-level-set non-crossing | monotone/non-crossing curves & coherence | CPF's "Coherence Proposition" is **inherited textbook**, cited not headlined. |

**Genuinely-CPF claims, ranked:** (1, lead, demonstrable) conformal floor over a CD-SBI pivot turns
PKD-insufficiency into a width cost; (2) coherent **sampleable** CD readout; (3, discipline not
capability) the flow is **frozen read-only** across a validity boundary so the sampler can't
contaminate coverage — a *cleanliness/auditability* claim, **not** "operator unification" (one model
that both scores and samples is the default in flow-SBI).

**Strategic honesty:** TRUST + CP4SBI bracket the validity core; CPF is an incremental combination +
one finding. Best framing = the three-axis map's worked example; standalone only if M1b clears both
bars. Decide at GATE 2, on evidence.

---

## 3. Hard constraints

1. **Three axes never conflated** (validity only from calibration). **Stated exception:** split/Mondrian
   conformal coverage is marginal over `(θ,x)~π×p` within a locale, so the *finite-sample* floor is
   **local-marginal under the proposal π** — the prior enters the validity axis through conformal
   localization. The one principled exception; acknowledged, not hidden.
2. **No-UMP off MLR** — "most powerful" needs a WAP weighting (prior → BFF; score = locally-most-powerful).
   Exposed as a power-axis **λ-dial**; no prior-free claim.
3. **Pitman–Koopman–Darmois** — fixed-dim sufficiency only for exponential families; compression costs
   **power, not validity** on the calibration side (what the floor buys).
4. **Non-identifiability (Dufour)** — unidentified θ → no *tight* valid set; honest output **valid-but-wide**
   (but see §6/§8: at finite n the certificate is π-marginal, which can under-cover at isolated θ₀).
5. **Freeze-before-calibrate** — flow frozen before calibration, on disjoint data; enforced **physically**
   by the checkpoint boundary + a disjointness assertion (§4.6).
6. **Conditional > marginal**; **finite-sample conditional coverage is impossible** (Foygel-Barber 2021).
   CPF claims **asymptotic-conditional ⊕ finite-sample-local-marginal-under-π**, **per-α**; never
   finite-sample-conditional, never all-α-simultaneous at finite sample.

---

## 4. Architecture

### 4.1 Operator — frozen conditional flow `r_φ(θ;x)`  [POWER + REPRESENTATION]

Conditional autoregressive/triangular monotone flow, conditioned on `c = E_ψ(x)`, trained by NF-MLE
toward `r(θ₀;x)|θ₀ ~ N(0,I)`:
`L_flow = E[ ½‖r_φ(θ₀;x)‖² − log|det ∂r_φ/∂θ| ]`. **A power-seeking loss, NOT a validity guarantee**
(validity created in §4.3). Frozen → safe to share across the two non-validity axes.

- Forward `θ→r`: reuse `flows/{triangular_additive,single_index_monotone}.py`.
- **No Fisher `Î` whitening.** Correct justification: **`F̂_{T|θ}` is fit per θ for whatever frozen
  scalar `T` we choose, so validity is statistic-agnostic; `Î` only affects power.** (Do *not* argue
  "monotone-reparam invariance" — at d>1 the vector reparam `r↦Î^{1/2}r` is not a per-θ monotone map
  of the scalar `‖r‖²`; the statistic-agnostic-calibration argument is the robust one. M1a tests
  whether `Î` was load-bearing *for power*.)
- Optional score warm-start (λ-dial, power ablation, OFF by default): regularize toward
  `u_λ=(1−λ)s_lik+λs_q`, `s_lik=s_q−∇log π`.

### 4.2 Statistic — `T(θ;x)=‖r_φ(θ;x)‖²`  [POWER, frozen]

Scalar; learned `r` gives anisotropic geometry without whitening. Frozen at §4.1.

### 4.3 Calibration — conditional CDF + conformal floor (ADOPTED from TRUST/CP4SBI)  [VALIDITY — sole source]

1. **Conditional CDF indexed by the HYPOTHESIZED θ.** Fit `F̂_{T|θ}(t)` = CDF of `T(θ;X)` under
   `X~p(·|θ)`, **as a function of the tested θ** (quantile regression of `T` *on θ*; conditioning
   variable = θ₀ from simulator pairs). **`x` enters only through `T(θ;x)`, never as the calibrator's
   conditioning variable** (conditioning `F̂` on x → a *credible-set*/marginal-over-π object — the
   TRUST-vs-Bayes distinction). Then `H(θ;x)=1−F̂_{T|θ}(T(θ;x))`; under exact `F̂` + atom-free `T`,
   `H(θ₀;x)|θ₀~U[0,1]` → per-θ₀ conditional coverage at all α, **asymptotically**.
   - **Negative control (mandatory; PER-θ₀ CONDITIONAL).** M0 must assert that conditioning `F̂` on x
     fails a **per-θ₀ conditional** KS (or a coverage-vs-θ₀ slope) test — **not** a pooled/marginal
     PIT, which the miswired calibrator *passes* (the x-conditioned PIT is uniform *marginally* — the
     Bayesian/credible property). Pre-register the detection threshold (mirroring §4.5d's δ-floor).
   - **Monotone-α head:** existing `MultiQuantileMLP` crosses (U-shape artifact). **Default = isotonic
     / CFV-Galichon rearrangement post-hoc** over the existing head; reg/MCQRNN = ablations.
2. **Coherence Proposition (INHERITED).** With `F̂_{·|θ}` monotone in its argument and **`H` a single,
   α-independent field**, `C_α(x)={θ:H(θ;x)>α}` and `α₁≤α₂ ⇒ C_{α₂}⊆C_{α₁}` for all x, all d.
   *Premise:* coherence holds for the *pre-conformal* `H`; the conformal layer must preserve it.
3. **Conformal floor — conformalize `H`, NOT `T`.** Use the nonconformity score `s=1−H=F̂_{T|θ}(T)` on
   calibration pairs, take its empirical quantile `q̂_α`; the set is `{θ:F̂_{T|θ}(T(θ;x))≤q̂_α}`.
   Equivalently, **"monotone recalibration of `H`" = α-relabeling via the single, α-independent global
   empirical score-quantile map** (so the plan tests *that* map). **Global `q̂_α`** is one threshold on
   the fixed field → coherence preserved + distribution-free finite-sample **local-marginal-under-π**
   floor (degrades to *wider* sets, never silent under-coverage). **Why not conformalize `T`:** a
   per-locale critical value on `T` makes the threshold θ-dependent → violates the §4.3.2 fixed-field
   premise → reintroduces crossing. **Scope:** M0/M1 use **global** conformal; the **local/Mondrian
   θ-partition is deferred to M2** (biggest build risk; documented d≥5 sparsity collapse). Under
   Mondrian the per-locale `q̂(locale(θ))` is piecewise-constant-in-θ → coherence becomes a *measured*
   property; global preserves it exactly.

Validity lives here and only here.

### 4.4 Sampler — the frozen flow, per-coordinate inversion  [REPRESENTATION — no coverage role]

Draw `θ~H(·|x)` by pushing `z~N(0,I)` through the **frozen** `r_φ^{-1}`. **Mechanism: per-coordinate
bisection** (`flows/invert.py::autoregressive_invert`, ~40 iters/coord), **single-index flow only at
first** (hardcodes `_s_theta`; triangular deferred). **Not a one-pass push.** Read-only; if wrong, the
*sets* (from `H`) stay valid, only the cloud is off. **Do not** equate the flow's `‖z‖²≤χ²` ball with
the valid set `C_α` (the latter comes from `H`/conformal). SIR-reweighting (ESS collapse) and Langevin
(mixing footgun) are rejected as defaults.

### 4.5 Audit  [independent verification]

Third disjoint split. (a) MC conditional coverage on a θ₀-grid — **NOTE: not a free reuse;**
`diagnostics/engine.py::evaluate_coverage` has a hardwired 3-way dispatch (`has_pivot`→χ²-on-`‖r‖²`;
`has_stat`→`T≤c_α`; else `contains_batch`), none of which fits an `H`-field set `{F̂(T)≤q̂}`. Action:
add a **4th branch (or an explicit adapter contract)** so `ConformalPivotProcedure` is evaluated as
`test_statistic=F̂(T(θ;x)), critical_value≡q̂_α` (and note the pivot χ²-KS/PIT diagnostics won't
auto-fire). (b) **firewall test** — corrupt the sampler, confirm `H`-sets unchanged. (c) **per-stage
isolated certification** (physical via §4.6) — calibrator with a known-true statistic; conformal with a
deliberately-wrong statistic (inflate widths, preserve coverage). (d) **leak audits, two arms + a
quantitative floor:** **A→B** (reuse flow data for calibration) **and** **B→C** (share
calibration/locale draws with the audit); clean run nominal, each contaminated arm must show **≥ a
pre-registered under-coverage δ** at `n_cal`; if indistinguishable from clean → underpowered, raise n.

### 4.6 Persistence (primary stage-decoupling mechanism)  [infrastructure]

**Framing correction:** `TrainedModel.state_dict` already exists in-memory (`methods/base.py:17`,
populated `methods/cd_sbi.py:189` via `self.flow.state_dict()`); only the **disk persist** is missing
(`run.py:635` saves `{arch_metadata, final_loss}` only). The add: **(i)** write the flow/conditioner
`state_dict` to disk (small); **(ii)** keep the metadata-only loader working under a versioned format
bump (`analysis/figures/data_io/checkpoints.py:30-39` hardcodes the metadata-only shape — must still
load); **(iii) new serialization** for the calibration head + conformal scores + (later) locale
partition (no existing analog); **(iv)** record `{seed,label}` provenance per stage (the `_derive(seed,
label)` substreams are deterministic) **and an explicit `assert_disjoint(provenance_A, draw_B)` gate
that FAILS LOUDLY on overlap**, with a unit test that an overlapping draw raises (provenance without
this check is decorative; the leak audit §4.5d must not be the *only* defense). This makes the
freeze-before-calibrate firewall **artifact-enforced**, makes §4.5c isolation trivial, and lets
calibration/conformal/Route-4 iterate against the same frozen flow without retraining. The in-memory
single-process runner (§5) remains a fallback (it works because one process never serializes).

---

## 5. Three-split firewall

```
 A (n_flow)  → train r_φ (NF-MLE)              → FREEZE → checkpoint
 B (n_cal)   → fit F̂_{T|θ} + conformal scores  → FREEZE → checkpoint
 C (n_audit) → MC coverage + leak/firewall tests
```

Disjoint by construction. **Streams `flow/cal/audit` do NOT exist** in `SeededRNGs` (only
`train/eval/init`); extend via `_derive(seed,label)` (~4 LOC) or derive in-runner. With persistence the
stages may be separate jobs; the in-memory `ConformalPivotFlowRunner` (TwoStageCDSBIRunner pattern) is
the fallback. #1 correctness risk = double-use leak; §4.5d + §4.6 `assert_disjoint` are the guards.

---

## 6. Coverage guarantees & ceiling

- **Asymptotic exact conditional, all α** — under exact `F̂` + atom-free `T` (idealization).
- **Finite-sample local-marginal *under π*** — split/Mondrian conformal, distribution-free, **per-α**
  (all-α simultaneity NOT finite-sample guaranteed).
- **Coherence** for all x, all d — for the **pre-conformal** `H`, preserved by conformalizing `H`
  (global map); measured under Mondrian.
- **NOT** finite-sample conditional (Foygel-Barber). Coverage **reported only from split C** MC.

---

## 7. cd_sbi implementation plan (reuse + gaps)

**WORKABLE HOST.** Corrected table:

| Need | Status | Action | Effort |
|---|---|---|---|
| Forward flow, `‖r‖²`, NF-MLE | ✅ `flows/*` | reuse | — |
| Flow inverse `z→θ` | ⚠️ `invert.py` per-coord bisection, **single-index only** (`_s_theta`); triangular `AttributeError`s | single-index for M0/M1; generalize later | Medium |
| RNG streams flow/cal/audit | ❌ only train/eval/init | extend via `_derive` | Small |
| Persist (disk) + provenance check | ⚠️ in-memory `state_dict` exists; disk-persist missing | §4.6: disk-persist + figures-loader compat + calibrator serialization + `assert_disjoint` | Medium |
| Conditioner on `x` | ✅ `conditioners/*` | reuse | Small |
| Monotone-α head | ⚠️ `MultiQuantileMLP` crosses | isotonic post-hoc (default) | Small |
| Conformal global split (recalibrate `H`) | ❌ absent | scalar global split-conformal | Small–Medium |
| Conformal local/Mondrian partition | ❌ absent, no scaffolding | **Large (≥500 LOC); biggest risk; defer to M2** | Large |
| `evaluate_coverage` for an `H`-field | ⚠️ hardwired 3-way dispatch, no `H`-field branch | **add 4th branch / adapter** | Small–Medium |
| Super-level-set procedure | ⚠️ NOT `PivotBasedProcedure` | `ConformalPivotProcedure` delegating to `CriticalValueProcedure` (`c_α≡q̂_α`; `contains_batch` handles scalar `c`) | Small |
| Sampler-PIT diagnostic (draw→PIT) | ⚠️ existing μ-CD uses finite-difference marginalization, **not** inversion | new draw-and-PIT path via `autoregressive_invert` (single-index) | Small–Medium |
| Simulators M0/M1 | ✅ `LocationNormal1D`, `NormalUnknownMeanVar` | reuse | — |
| **CP4SBI baseline** | ❌ none in repo | NPE + conditional-CDF + LoCart conformal — **a method build, needs the M2 partition** | Medium–Large |
| **TRUST++ baseline** | ❌ none in repo | second partition-QR method — **a method build** | Medium–Large |
| Config/dispatch | ✅ | method yaml pinning `method.flow` + `_build_method` branch | Small |

**Build order (M0-first):** 1. RNG streams. 2. `ConformalPivotProcedure` (delegate to
`CriticalValueProcedure`). 3. isotonic monotone-α. 4. global split-conformal on `H`. 5.
`evaluate_coverage` 4th branch. **(M0 needs 1–5.)** 6. `ConformalPivotFlowRunner` + persistence
(§4.6). 7. *(M1a)* sampler draw-and-PIT diagnostic. 8. *(M2)* local/Mondrian partition **(Large;
deferred)**. 9. *(M1b/M2)* CP4SBI + TRUST++ baseline builds. 10. configs + dispatch.

---

## 8. Milestones

**M0 — validity skeleton (no flow, global conformal).** `LocationNormal1D`, `T=‖r_star‖²`; recover
exact `N(x,σ²)`. Metrics: pivot KS vs N(0,1); `coverage_error_max` vs χ² floor; **non-crossing ≡ 0
*after both* the isotonic and conformal layers** (not just pre-conformal); **the per-θ₀ conditional
negative-control test** (§4.3.1). Certifies the validity machinery in isolation.

**M1a — design gate (kill-or-confirm `Î`); repo machinery only.** `NormalUnknownMeanVar`, no-`Î` flow
+ global conformal + audit + the sampler draw-and-PIT diagnostic + the two-arm leak audit. **GATE 1:**
`coverage_error_max ≲ 0.03` AND the flow-inversion sampler PIT passes KS ≤ 0.05 on **both** σ²-direct
(χ²) and μ-marginalized (**Student-t**) CDs. If the no-`Î` calibrator can't recover the Student-t → `Î`
load-bearing *for power*; reintroduce it.

**M1b — novelty bake-off (GATE 2).** Insufficient-summary CD-SBI + sharpness-at-matched-coverage.
**Near-term comparison (buildable without the partition):** CPF vs **global-conformal-on-NPE** — CPF
must (i) restore coverage that insufficient-summary CD-SBI loses **and** (ii) be sharper than
global-conformal-on-NPE. **True CP4SBI/TRUST++ head-to-head needs the local/Mondrian partition →
staged to after M2.** (Resolves the global-vs-Mondrian contradiction: don't compare against a
partition-method before the partition exists.)

**M2 — multivariate + partition + validity-only multimodal demo.** `LocationGaussian2D_corr` / d=5
`NormalBivariateUnknownCov` (anisotropic geometry + the local/Mondrian partition; watch d≥5 sparsity).
**Multimodal target (Two Moons / Gaussian mixture):** success criterion **down-scoped** to "π-marginal
coverage restored **and** per-θ₀ coverage no worse than baselines," **reporting the per-θ₀ coverage
*spread*, not just `coverage_error_max`** (a grid average would hide the conditional gap a π-marginal
floor can leave). Then the true CP4SBI/TRUST++ head-to-head.

**Baselines:** CP4SBI, TRUST++ (staged post-partition), CD-SBI, LF2I-BFF, raw posterior, global-conformal-on-NPE.
**Metrics:** conditional `coverage_error_max` **and per-θ₀ spread**; sharpness at matched coverage;
coherence; amortization speedup; recalibration-delta; sampler fidelity (not a validity metric).

---

## 9. Failure modes & guards

1. **Double-use leak (A→B / B→C)** → void floor. *Guard:* §4.5d two-arm δ-audit + §4.6 `assert_disjoint`.
2. **Partition sparsity at d≥5** → degrade to global conformal (marginal). *Guard:* learned locale
   embeddings; per-cell counts; if sparse, fall back to global + **down-claim to finite-sample-marginal**.
   Defer the partition to M2.
3. **θ-vs-x conditioning slip** (the fatal hole) → marginal-credible. *Guard:* the **per-θ₀ conditional**
   M0 negative control.
4. **Conformalize-`T` reintroducing crossing.** *Guard:* conformalize `H` only.
5. **Fisher degeneracy / non-identifiability** → `T` unstable; the π-marginal floor can still under-cover
   at isolated θ₀. *Guard:* report per-θ₀ spread; honest "π-marginal-valid, possibly conditionally
   under-covering at isolated θ₀, and wide" — not unqualified "valid-but-wide."
6. **Coverage read off the flow/sampler.** *Guard:* coverage API consumes only `H`/conformal; firewall CI gate.

---

## 10. Route 4 extension (after Route 2)

End-to-end flow training on a **differentiable conditional-coverage + WAP-sharpness** objective
(Gneiting; λ = WAP). **Reused:** flow, conformal layer, audit, firewall, **persistence**. **Changed:** α
sampled per-batch → coherence *enforced* (monotone penalty). **New risk:** co-optimized validity/power →
gameable surrogate. **Contract:** keep the **hard non-differentiable conformal projection** post-hoc to
re-establish validity regardless of training; keep firewall + leak audits as CI gates; the conformal
split stays frozen-disjoint from the end-to-end data. No Langevin.

---

## 11. Open questions & gates

- **GATE 1 (M1a):** does no-`Î` recover the Student-t? If no → reintroduce `Î`.
- **GATE 2 (M1b near-term, then post-partition true bake-off):** honest default = **map-paper worked
  example**; standalone **only if** (i) coverage restored over CD-SBI and (ii) sharper than
  global-conformal-on-NPE near-term *and* competitive with true CP4SBI/TRUST++ post-partition.
- **Partition localization at d≥5** — the scaling risk; may bound usable `d` or force the marginal down-claim.
- **Samples+scores conditioner** — deferred. **Triangular inversion** — generalize or stay single-index.
- **Naming** — finalize with GATE 2 positioning.

---

## 12. Summary

CPF = **a learned, frozen conditional-flow pivot whose norm is the statistic (power), fed into the
adopted TRUST/CP4SBI conditional-CDF + (global, later local) conformal calibration (validity), read out
as a coherent sampleable CD via super-level sets (inherited coherence), sampled by the same frozen flow
behind a checkpoint-enforced firewall (representation)** — no `Î` whitening, a three-split firewall made
physical by persistence + an `assert_disjoint` gate, kill-or-confirmed at (μ,σ²) by the no-`Î`
Student-t test (M1a), and judged near-term vs global-conformal-on-NPE then post-partition vs
CP4SBI/TRUST++ (M1b/M2). The validity layer is **adopted, not invented**; the increment is the
learned-pivot statistic + sampleable-CD readout + firewall discipline, plus the **finding** that a
conformal floor converts CD-SBI's PKD-insufficiency from a validity break into a width cost. Most
defensible as the three-axis map's worked example; standalone only if M1b clears both bars. `cd_sbi` is
a workable host. Route 4 swaps in an end-to-end differentiable-coverage objective behind the same
conformal projection and firewall.
