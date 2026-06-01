# Conditional coverage in CD-SBI: a decomposition, and the road to a non-oracle high-d construction

**Status:** working theory note. Each claim is tagged **[proven]**, **[needs
numerical check]**, or **[conjectural]**. Numerical validators live in
`tests/theory/test_coverage_decomposition.py`; each theorem names its test.

**Why this note exists.** The architectural-monotonicity (UMNN) program was
meant to *guarantee* conditional coverage everywhere. It did not, and we spent a
detour on an LF2I alternative as a result. The root cause is that the framework
has been bundling **three logically independent conditions** under one
"monotonicity" banner. This note separates them, proves what each one actually
buys, and shows that the property we care about — exact conditional coverage —
needs only two of them, neither of which is "monotone in θ." It then locates the
non-oracle high-dimensional frontier precisely: it is the *efficiency* axis, not
the *validity* axis.

---

## 0. Setup and the object of interest

A simulator defines a family of conditionals `{p(· | θ) : θ ∈ Θ}`, `Θ ⊆ ℝ^{d}`,
with data `X ∈ 𝒳`. Choose a **summary** `T = s(X) ∈ ℝ^{d}` (any measurable map;
`T = X` allowed when `dim 𝒳 = d`). A **pivot** is a map `r(θ; ·): ℝ^{d} → ℝ^{d}`.
The induced confidence set at level `α` is

    C_α(X) = { θ : ‖ r(θ; s(X)) ‖² ≤ χ²_{d,α} }.                         (0.1)

**Definition (conditional coverage).** `r` has *exact conditional coverage* if

    P_{X|θ₀}( θ₀ ∈ C_α(X) ) = α   for every θ₀ ∈ Θ and every α ∈ (0,1).   (0.2)

This is the pointwise frequentist guarantee — "everywhere" means the `∀ θ₀`. It
is equivalent to a family of **pushforward** conditions:

**Definition (calibration manifold).**
`M = { r : (r(θ₀; ·))_# p(· | θ₀) = N(0, I_d)  for every θ₀ }`.

`r ∈ M ⟺ (0.2)` holds, because then `‖r(θ₀; T)‖² ~ χ²_d` and (0.1) thresholds it
at the `α`-quantile. So the entire game is: **produce an `r ∈ M`, without an
oracle, in high dimensions.**

---

## 1. The master theorem: validity needs C1 + C2, and nothing else

**(C1) Injectivity-in-data.** For every `θ`, `T ↦ r(θ; T)` is a `C¹`
diffeomorphism of (the support of) the summary onto `ℝ^d`: injective, surjective,
with everywhere-nonvanishing Jacobian `∂_T r`.

**(C2) Density match.** `φ_d(r(θ; T)) · |det ∂_T r(θ; T)| = p(T | θ)` for all
`θ, T`, where `p(T|θ)` is the law of `T = s(X)` under `X ~ p(·|θ)`.

**(A) Regularity.** For every `θ`, the law `μ_θ` of `T = s(X)` is absolutely
continuous w.r.t. Lebesgue on `ℝ^d`, with a strictly positive `C¹` density
`p(T|θ)`. *(This is exactly the hypothesis that was missing from the first draft;
without it C1+C2 are not jointly satisfiable, and "any summary" below is false —
e.g. a rank-deficient learned `s_φ` concentrates `μ_θ` on a lower-dimensional set
and admits no diffeomorphism C1.)*

**Lemma 0 (existence). [proven]** *Under (A), the increasing-triangular
Knothe–Rosenblatt rearrangement `g_θ` of `μ_θ` to `N(0,I_d)` exists and is a `C¹`
diffeomorphism (Rosenblatt 1952; Bogachev–Kolesnikov–Medvedev). Setting
`r(θ;·) := g_θ` makes C1 and C2 hold simultaneously.* So a calibrated injective
pivot **exists for any summary satisfying (A)** — the hypotheses of Theorem 1 are
not vacuous and place a real, checkable condition on `s`.

**Theorem 1 (exact conditional coverage). [proven]**
*If C1 and C2 hold, then `r ∈ M`, i.e. (0.2) holds for every `θ₀` and every `α`.*

*Proof.* Fix `θ₀`. By C1 the map `g = r(θ₀; ·)` is a diffeomorphism, so for
`T ~ p(·|θ₀)` the pushforward `g_# p(·|θ₀)` has a density, given by the
change-of-variables formula as `p(g⁻¹(z)|θ₀) / |det ∂_T g (g⁻¹(z))|`. By C2 the
numerator equals `φ_d(z) · |det ∂_T g|`, so the ratio is `φ_d(z)`. Hence
`r(θ₀; T) ~ N(0, I_d)`, `‖r(θ₀;T)‖² ~ χ²_d`, and
`P(θ₀ ∈ C_α) = P(χ²_d ≤ χ²_{d,α}) = α`. ∎

**Proposition 1′ (radial NSC — coverage is weaker than Gaussianization). [proven]**
*Exact coverage at all levels at `θ₀` `⟺` `‖r(θ₀;T)‖² ~ χ²_d` under `T~p(·|θ₀)`.*

*Proof.* `coverage(θ₀,α) = G_{θ₀}(χ²_{d,α})`, `G_{θ₀}` the CDF of `‖r(θ₀;T)‖²`.
Since `α ↦ χ²_{d,α}` is the χ²_d quantile function, `G_{θ₀}(χ²_{d,α}) = α ∀α ⟺
G_{θ₀} = F_{χ²_d}`. ∎ This is **strictly weaker** than `r(θ₀;T)~N(0,I_d)` (any
law with a χ²_d radial part qualifies — rotations, non-uniform angular parts). So
Theorem 1 proves coverage via the *stronger* (Gaussianization) route; **Prop 1′ is
the actual necessary-and-sufficient condition.** Validity only ever needed the
radial law. *Consequence (carried into §6½/§9): this softening makes validity
easier to attain but makes the* collapse *problem (§9) strictly worse — a weaker
target constrains the summary less.*

**What Theorem 1 deliberately does *not* assume — the two non-requirements that
reorganize the whole project:**

- **No R1 (no monotonicity in θ).** Coverage at `θ₀` depends only on the law of
  the single slice `r(θ₀; ·)`. How `r(·; T)` behaves *across* `θ` is irrelevant
  to (0.2). Monotonicity in θ governs only the *shape* of `C_α`.
- **No sufficiency of `T`.** C2 matches whatever marginal `p(T|θ)` the chosen `T`
  induces. **Any summary satisfying (A)** yields valid coverage; sufficiency
  controls only the *size* of `C_α` (§3). (Not literally *any* map — (A) can fail
  for degenerate learned summaries.)

*Validator:* `test_theorem1_oracle_exact_coverage_everywhere` — with the
closed-form `r*` of `NormalUnknownMeanVar`, draws `X ~ p(·|θ₀)` at several `θ₀`
(including box-extreme) and checks per-coord normality (KS) and that empirical
coverage of `C_α` equals `α` at `α ∈ {.5,.8,.9,.95}`.

---

## 2. Three orthogonal axes (the decomposition)

Theorem 1 forces a clean separation that the manuscript currently entangles:

| Axis | Condition | Buys | Failure mode if absent |
|---|---|---|---|
| **Validity** | C1 + C2 | exact conditional coverage `∀θ₀` | under/over-coverage |
| **Efficiency** | `T` sufficient (§3) | smallest valid `C_α` | *valid but wide* |
| **Shape / optimality** | R1: `r` monotone-triangular in θ (§4) | nested, connected, UMP(U) CD; unique trained map | valid but oddly-shaped; non-identifiable training |

These are **independent**: you can have any one without the others. The property
we promised collaborators — coverage everywhere — is the **validity row alone**.
Every (μ,σ²) / μ₂ struggle was a failure to know *which row* was breaking.

---

## 3. The summary, sufficiency, and "valid but inefficient" as a theorem

Write `p(X|θ) = p(T|θ) · p(X | T, θ)` with `T = s(X)`. The pivot machinery (NF-MLE
on the summary) only ever sees and matches `p(T|θ)`; the residual `p(X|T,θ)` is
never modeled.

**Proposition 2 (validity ⟂ sufficiency — validity direction). [proven]**
*Let `T = s(X)` be any statistic satisfying (A) and let C1, C2 hold for the pivot
on `T`. Then `C_α` has exact conditional coverage (Thm 1), regardless of whether
`T` is sufficient.* ∎ (Nothing in Lemma 0 / Theorem 1 used sufficiency.)

**Efficiency direction — what is actually true, and what is not.** I previously
"proved" minimality of `E|C_α|` for sufficient `T` by gesturing at Neyman–Pearson.
That was **not a proof** and the general claim is false without optimality
hypotheses. The rigorous statement is only an identity:

**Proposition 2′ (efficiency = power, Fubini). [proven]**
*`E_{X|θ*} |C_α(X)| = ∫_Θ P_{X|θ*}( θ ∈ C_α(X) ) \, dθ`, with the integrand `= α`
at `θ = θ*` and, for `θ ≠ θ*`, equal to the probability the set wrongly includes
`θ`.* (Tonelli; the integrand is a nonnegative measurable inclusion probability.)

So expected volume = `α·|Θ|`-baseline minus the procedure's **power to exclude
false `θ`**. Sufficiency enters only through this power, and "sufficient ⟹ smaller
sets" holds *only for power-optimal (UMPU-type) pivots* — that is the R1/optimality
axis (§4), **not** a generic corollary. **[conjectural]** in general; **[proven]**
for a clean coarsening such as the sub-sample summary (where discarding data
strictly lowers power), which is what the numerical witness exercises.

This sharpens the recurring *calibration ≠ informativeness* finding (energy arm;
N3 summaries calibrate at `χ²₅ ≈ 0.03` yet recover only 4/5 directions): those
summaries were **valid** but low-power. The "failure" was on the efficiency axis.
The `E[log|det ∂r/∂θ|]` quantity in the test is a *local* log-volume proxy that
presupposes R1 and a Jensen step — a witness for the sub-sample case, not a proof.

**Pitman–Koopman–Darmois wall. [proven, classical]** A fixed finite-dimensional
sufficient statistic exists for all `n` **iff** the family is exponential. So
outside exponential families, *no* fixed `T ∈ ℝ^d` is exactly sufficient:
efficiency is fundamentally an *approximation/asymptotic* problem, while validity
is *exact*. This is the formal reason the non-oracle high-d frontier is the
efficiency axis (§7).

*Validator (scaffold):* `test_validity_without_sufficiency` (intensive, TODO) —
train a flow on a deliberately lossy summary (e.g. `X̄` only, dropping `s²`); assert
`χ²` coverage still holds while `E|C_α|` inflates vs the sufficient summary.

---

## 4. R1 selects a branch; it is not a coverage condition

**Lemma 3 (reflection symmetry of NF-MLE). [proven]**
*The NF-MLE objective `L(r) = E_{θ,T}[ ½‖r(θ;T)‖² + ½d·log2π − log|det ∂_T r| ]`
is invariant under the per-coordinate reflection `r_k ↦ −r_k`.*

*Proof.* `‖r‖²` is even in each `r_k`, and `|det ∂_T r|` is invariant under sign
flips of output coordinates. ∎

For a scalar conditional the two Gaussianizing transforms are
`Φ⁻¹(F(T|θ)) = −Φ⁻¹(1−F(T|θ))`, i.e. exactly an `r ↦ −r` pair — both push
`p(T|θ)` to `N(0,1)`, both have exact coverage. **Calibration is sign-blind.**

But `r_k ↦ −r_k` flips *both* `∂r_k/∂θ_k` and `∂r_k/∂T_k` simultaneously.
Therefore:

**Corollary 4 (what R1 is for). [proven]**
*Committing the diagonal sign `∂r_k/∂θ_k > 0` (R1) breaks the reflection
degeneracy: it selects a unique branch (the Knothe–Rosenblatt rearrangement) and
thereby **determines** the data-direction sign `s_k = sign(∂r_k/∂T_k)`, which is
fixed by the target's coupling — additive `θ−T ⟹ s_k=−`; multiplicative rate
`θT ⟹ s_k=+`; ratio/scale `s²/σ² ⟹ s_k=−`. R1 supplies identifiability and CD
nesting, not coverage.*

**Re-diagnosis of the UMNN failure (now exact). [proven]**
`DoublyMonotoneUMNN` hard-codes `(↑θ, ↑data)`. The two valid branches of the
scale conditional are `(↑θ, ↓data)` and `(↓θ, ↑data)`. `(↑θ, ↑data)` is **neither
branch**, so `p(s²|σ²) ∉ {φ(r)|∂r| : r ∈ class}`: the class is **non-realizable**
for ratio-type sufficiency, C2 is unreachable, and Thm 1's hypothesis simply
fails. This is a *class-design* failure (traceable entirely to Lemma 3), not an
optimization failure — categorically different from the μ₂ issue (§6), which is
realizable but under-optimized in the prior's tails.

*Validator:* `test_reflection_branch_also_calibrates` — the σ-flipped pivot
`[−r*_σ, r*_μ]` (the `(↓θ,↑data)` branch) is shown to have the **same** exact
coverage as `r*`, demonstrating sign-blind calibration and that `↑data` *is*
satisfiable — just not jointly with `↑θ`.

---

## 5. The finite-sample bridge: the loss gap is a coverage certificate

C2 is never exact; NF-MLE gives `q ≈ p`. Let the achieved loss be `L` and the
entropy floor `H̄ = E_{θ∼ρ}[ H(T|θ) ]` (`= simulator.entropy_lower_bound()`).

**Theorem 5 (coverage error ≤ loss gap). [proven]**
*Under C1 (so `q` is a genuine normalized density and the bijection `r(θ;·)`
preserves total variation),*

    E_{θ₀∼ρ}[ sup_α | P_{X|θ₀}(θ₀∈C_α) − α | ]
        ≤ E_{θ∼ρ}[ TV(p_θ, q_θ) ]
        ≤ sqrt( ½ · E_θ[ KL(p_θ ‖ q_θ) ] )
        =  sqrt( (L − H̄) / 2 ).                                          (5.1)

*Proof.* Pushforward by the bijection `r(θ;·)` preserves TV, so the coverage
error at `θ` (two measures of the same Borel set `B_α`) is `≤ TV(ν_θ, N(0,I)) =
TV(p_θ, q_θ)`. Pinsker, then Jensen, then `L − H̄ = E_θ KL(p_θ‖q_θ)` (the NF-MLE
identity, exact for this loss form). ∎

Two consequences that re-found the project's main diagnostic:

- **"Reach the entropy floor" *is* "bound the coverage error."** `L − H̄` is a
  certificate. This is why logs cite `0.985` vs floor `0.99`.
- **Folding = a C1 violation = a void certificate.** Without injectivity `q`
  integrates to `< 1` on folds, `L` drops *below* `H̄`, and (5.1)'s TV-invariance
  step fails. C1 is precisely the hypothesis that keeps the loss honest. This is
  the rigorous statement of "monotonicity must be architectural."

*Validator:* `test_theorem5_oracle_zero_gap` (oracle end: at `r*`, `L ≈ H̄`,
coverage error `≈ 0`, and `L ≥ H̄ − noise` — no folding). `test_theorem5_bound_
holds_for_trained_runs` (scaffold, TODO) loads sweep outputs and asserts measured
`sup_α|cov−α| ≤ sqrt((L−H̄)/2)` across budgets.

---

## 6. Why "everywhere" is strictly harder than (5.1): the average–sup gap

Theorem 5 bounds the **ρ-average** error. "Everywhere" is the **pointwise sup**
`sup_{θ₀}`, and

    sup_{θ₀} TV(p_{θ₀}, q_{θ₀})   ⩽̸   sqrt((L−H̄)/2)   in general.

**Proposition 6 (the μ₂ phenomenon, exactly). [proven characterization]**
*The gap between ρ-average and pointwise-sup coverage error is governed by how
much ρ underweights a region times the local `KL(p_{θ₀}‖q_{θ₀})` there. A model
at the ρ-averaged floor can have large local KL in ρ's tails — central coverage
exact, extreme-θ₀ coverage degraded.* This is not a contradiction with Thm 5; it
is the average-vs-sup gap, and it is exactly the d=5 μ₂ behaviour (PIT KS up to
~0.10 only at box-extreme θ₀; central coverage fine).

Routes to close it, increasing in strength:
1. **Exact realizability** (`q=p`, not `≈`): closed-form / oracle pivots — why
   every oracle-summary Stage-A run calibrated everywhere.
2. **Sup-controlling objective** [conjectural]: minimax / ρ-reweighted NF-MLE
   (importance-weight toward tails), trading average for uniform fit.
3. **Pointwise certificate + local refinement**: the coverage grid as a
   diagnostic, capacity added where (5.1)'s local version fails.

---

## 6½. S2 — a sup-controlling objective gives uniform coverage

Theorem 5 controls `E_ρ`; "everywhere" needs `sup_θ`. The fix is to optimize the
**worst-θ regret** rather than the average.

**Definition (per-θ regret).** `R(θ; r) = E_{T|θ}[−log q_r(T|θ)] − H(T|θ) =
KL(p_θ ‖ q_{r,θ}) ≥ 0`. Plain NF-MLE minimizes `E_{θ∼ρ} R(θ;·)`.

**Theorem S2 (minimax regret ⟹ uniform coverage — the easy implication). [proven]**
*If `r̂` achieves `sup_{θ∈Θ} R(θ; r̂) ≤ ε`, then*

    sup_{θ∈Θ} sup_α | coverage_{r̂}(θ,α) − α |  ≤  sqrt(ε / 2).

*Proof.* Pointwise Pinsker on `TV(p_θ,q_θ)` (Thm 5's per-θ step), uniformly in
`θ`. ∎

**What Theorem S2 does NOT establish (the open substance of S2).** It is a
conditional ("*if* you drive `sup_θ R` small, *then* uniform coverage"). It does
not prove any of:
1. **Constrained-regime realizability** — a bound on `ε* := inf_{r∈ℛ} sup_θ
   R(θ;r)` for a finite-capacity / finite-sample class `ℛ`. At infinite capacity
   `ε*=0` (the KR pivot, Lemma 0), so *all* of S2's content lives in `ε*>0`, which
   is unbounded here. **[open]**
2. **A tractable estimator with convergence** — that minimizing an empirical
   minimax / CVaR objective attains near-`ε*`. **[open]**
3. **Strict improvement over the average.** Only the *certificates* compare:
   `min_r sup_θ R ≤ sup_θ R(r_avg)` is a tautology, so the minimax solution's
   *Pinsker bound* is ≤ the average's — but this does **not** prove its *actual*
   coverage beats the average's actual coverage. The toy below is one instance
   where it does; that is a witness, not a theorem. **[demonstrated, not proven]**

So the target objective is the **distributionally-robust** problem
`min_r sup_{θ∈Θ} R(θ;r)` (in practice: a max over a θ-grid, a `CVaR_β` softening,
or reweighting `w(θ) ∝ R(θ;·)`). The only subtlety is that `R = CE(θ) − H(T|θ)`
and the per-θ entropy floor `H(T|θ)` is θ-varying — minimizing `sup_θ CE(θ)`
alone wrongly over-weights high-entropy θ. Two routes resolve it:

- **Route 1 — known / asymptotic floor.** When `H(T|θ)` is closed-form (Gaussian
  summaries) or, in the `n→∞` regime of §7, equal to the Fisher-information floor
  `H(T|θ) → ½ log((2πe)^d / det I(θ))` for a score/MLE summary, subtract it and
  minimax the regret directly. (This is a second reason the score summary of §7 is
  the right high-d object: it makes the uniform-coverage objective well-posed.)
- **Route 2 — floor-free, flow-only.** With a **fixed** summary, minimize the
  per-θ *calibration discrepancy* `D(θ) = sup_α |coverage(θ,α) − α|` (or a smooth
  pushforward-vs-`N(0,I)` surrogate) — no floor needed, directly the quantity we
  want uniformly small. **Caveat (II-A):** a calibration-only objective collapses
  a *learned* summary (the energy-arm finding), so Route 2 is valid only when the
  summary is fixed — which is exactly the **μ₂** situation (fixed oracle Bartlett
  summary; the ctx-MLP under-fits the affine-index intercept in ρ's tails). For
  *joint* summary+flow, combine an information-preserving term (NF-MLE /
  exact-density) with the `sup_θ` calibration term.

Note the `run.py` coverage grid already *estimates* `sup_θ D(θ)`; S2 turns that
existing certificate into a training signal.

**Witness (one instance — NOT a proof of S2)** —
`test_s2_minimax_controls_sup_coverage`: a closed-form finite-θ instance
(`T|θ ~ N(0,v_θ)`, single-scale pivot) where ρ underweights a high-variance tail.
The ρ-average optimum `s²=E_ρ[v]` leaves **sup-coverage error 0.30**; the minimax
optimum **0.15** — a 2× improvement at *this* `(v,ρ)`. It exhibits the mechanism
of Prop 6 in solvable form; it does not establish items 1–3 above.

*Next (scaffolded `test_s2_minimax_trained`):* the trained analog — cripple a
flow's σ-context capacity to induce a μ₂-style tail gap under ρ-average NF-MLE,
then retrain with minimax/CVaR reweighting and show `sup_θ` coverage error drops.

## 7. Toward a non-oracle, high-dimensional construction

Combine §1–§6. The high-d non-oracle problem **splits**:

- **Validity is already non-oracle and exact.** By Thm 1 + Prop 2, *any* learned
  summary `T = s_φ(X)` plus an injective, sign-correct flow `r(θ;T)` reaching
  `q≈p` yields valid coverage — *no sufficiency required*. The N3 runs already
  exhibit this (`χ²₅ ≈ 0.03` with a learned DeepSets summary). **The validity
  guarantee does not need the oracle.**
- **Efficiency is the open frontier, and it is asymptotic by Pitman–Koopman–
  Darmois.** Outside exponential families no fixed `T∈ℝ^d` is exactly sufficient,
  so the target is *asymptotic* sufficiency: a summary that, as `n→∞`, spans the
  local score `∇_θ log p(X|θ)` (the score is asymptotically sufficient; the MLE /
  efficient-score statistic is the canonical `d`-dim asymptotically sufficient
  summary). 

**Proposed construction (to be made rigorous next). [conjectural]**
A non-oracle CD-SBI procedure with (i) **exact finite-sample validity** and (ii)
**asymptotic efficiency**:

1. Learn `s_φ: 𝒳 → ℝ^d` and an **injective, sign-correct** flow `r_ψ(θ;·)` on
   `ℝ^d` (C1 by construction — single-index / triangular with injected signs).
2. Train by NF-MLE on `(θ, s_φ(X))`. At the population optimum, C2 holds for the
   *induced* `p(s_φ(X)|θ)` ⟹ **exact validity for any φ** (Thm 1).
3. Add an **efficiency objective** on `s_φ` that drives it toward an
   asymptotically sufficient (score-spanning) summary — e.g. maximize
   `I(s_φ(X); θ)` locally, or align `s_φ` with the efficient score. This moves the
   procedure along the *efficiency* axis **without touching validity** (Prop 2).

The decisive theoretical claims to nail down, in order (the relaxation program):

| Step | Claim to prove / check | Tag |
|---|---|---|
| S1 | Validity ⟂ sufficiency in the summary setting (Lemma 0, Thm 1, Prop 1′, Prop 2/2′) | **[proven, with hypothesis (A)]** |
| S2a | *Implication*: `sup_θ R ≤ ε ⟹` uniform coverage `≤ √(ε/2)` (Thm S2) | **[proven — but easy]** |
| S2b | *Substance*: bound `ε*`, strict minimax>average, optimal reweighting (§9½, Thms 8–9) | **[ε*/separation/π_LF PROVEN for single-scale + any KL-convex class; estimator+convergence OPEN]** |
| S3a | Validity holds for a *learned* `s_φ` *when (A) holds*; certifiable post-hoc by Thm 5 (ρ-avg) | **[proven implication + witnessed once trained]** |
| S3b | Efficiency from sufficiency (`sufficient ⟹ smaller E|C_α|`) | **[OPEN in general; proven for clean coarsenings]** |
| #3 | Floor-free coverage objective: correct on fixed summary, collapses on learned (§9) | **[proven — see §9]** |
| S4-found | Efficiency `=` Fisher-info preservation: `I_T⪯I_X` eq. iff sufficient (Thm 10); `log\|C_α\|=−½log det I_T` (Thm 11); collapse `=` `det I_T→0` (§11) | **[PROVEN; Thm 11 exact for location, asymptotic for scale]** |
| S4 | A score-capturing `s_φ` attains `I_{s_φ}→I_X` ⟹ `E|C_α| →` oracle volume | **[reduced to Thm 10–12; estimator + finite-n gap OPEN]** |
| S5 | Honest realizable class jointly in θ (R1-realizability = KR map smooth/monotone in θ) | **[open]** |

**Headline reframing for the experiments.** The multimodal/SLCP frontier is, by
Thm 1, a **validity win** not a loss: with C1+C2 and *no R1*, CD-SBI yields
exactly-covering sets that become honestly **disconnected** — the correct answer a
bimodal posterior demands, where posterior-HPD methods give a wrong unimodal
region. (The earlier "calibration family breaks on SLCP" was the asymptotic
LF2I-*Score* method, not this exact construction.)

*Validator (scaffold):* `test_validity_without_r1_disconnected_sets` (TODO) — a
1-D two-component toy where the exact pivot gives a disconnected, exactly-covering
`C_α`.

---

## 8. Summary of the hypothesis stack (the thing to relax one at a time)

| # | Hypothesis | Role | Enforce by | Verify by |
|---|---|---|---|---|
| C1 | `r(θ;·)` diffeomorphism in summary space, **target-correct signs** | validity + honest loss | injective flow, injected signs | pushforward KS; Jacobian-sign audit |
| C2 | `q(T|θ)=p(T|θ)` | validity | NF-MLE (population) | `L→H̄`; Thm 5 bound vs measured coverage |
| Suff | `T` sufficient | **efficiency only** | oracle now; score-spanning `s_φ` later | `E|C_α|` vs oracle; canonical-corr |
| R1 | `r` monotone-triangular in θ | shape, uniqueness, UMP(U) | UMNN integrand / triangular | `∂r/∂θ>0` audit; CD monotonicity |
| R1* | per-θ KR map smooth/monotone *in θ* | makes R1 realizable | open (multivariate) | Jacobian-recovery vs `L⁻¹` |

The discipline: **validity (C1+C2) is exact and already non-oracle; efficiency
(Suff) is the high-d open problem and is asymptotic; shape (R1, R1*) is separate
and never a coverage condition.** Keep these three from leaking into each other.

---

## 9. #3 — the floor-free coverage objective: correct on a fixed summary, collapses on a learned one

This resolves the question that gates whether S2 even has a usable form with a
learned summary. Define the **coverage objective**

    J(s, r) = sup_{θ∈Θ} D(θ),   D(θ) = sup_α |coverage(θ,α) − α| = ‖G_θ − F_{χ²_d}‖_∞,

where `G_θ` is the CDF of `‖r(θ;T)‖²` under `T~p(·|θ)` (Kolmogorov distance to
χ²_d). By Prop 1′, `D(θ)=0 ⟺ exact coverage at θ`. `J` is floor-free — it never
references `H(T|θ)` — which is what made it attractive as a Route-2 objective.

**Theorem 7 (correctness, well-posedness, collapse). [proven]**

*(i) Correctness.* `J(s,r)=0 ⟺` the procedure has exact conditional coverage at
every `θ`. (Prop 1′ pointwise, then `sup_θ`.) Note `J` targets only the **radial**
law (Prop 1′), strictly weaker than the Gaussianization `C2` that NF-MLE forces.

*(ii) Fixed-summary well-posedness.* Fix `s` with each `μ_θ` satisfying (A). Then
`inf_r J(s,r)=0`, attained by the KR pivot `r(θ;·)=g_θ` (Lemma 0). So with a fixed
summary the objective is correct **and** its optimum is the uniform-coverage
goal — this is exactly the **μ₂ setting** (fixed oracle Bartlett summary, train the
pivot only), where Route 2 is legitimate.

*(iii) Learned-summary collapse.* Minimizing `J` jointly over `(s,r)` is
ill-posed. If the model admits an ancillary statistic `A=a(X)` (law free of `θ`)
with an absolutely continuous distribution — which holds for the location/scale/
exponential families in this project — then `s:=a`, `r(θ;T):=g(T)` (θ-independent,
`g` Gaussianizing `A`'s law) gives `J=0` with

    C_α(X) = { θ : ‖g(a(X))‖² ≤ χ²_{d,α} } ∈ { ∅, Θ } :

the set is the whole space (prob α) or empty (prob 1−α) — **exact coverage, zero
power.** *Proof.* `‖g(A)‖² ~ χ²_d` and is `θ`-free, so `coverage(θ₀,α)=
P(‖g(A)‖²≤χ²_{d,α})=α` for every `θ₀`, and `C_α` is `θ`-independent hence `{∅,Θ}`.
∎

So **coverage is invariant to summary collapse**: `J` alone cannot distinguish an
informative summary from an ancillary one. And the radial softening (Prop 1′)
makes this *worse*, not better — it shrinks the constraint from `r~N(0,I)` to
`‖r‖²~χ²_d`, admitting strictly more degenerate `(s,r)`. (This is the price of the
softening you flagged: easier validity, harder collapse.)

**Corollary 7′ (coverage needs an efficiency/information partner). [proven]**
By Prop 2′, the collapsed procedure has `E|C_α| = α·|Θ|` — *maximal*. So pairing
`J` with either (a) an expected-volume penalty `E|C_α|` (directly Prop 2′; needs
set construction) or (b) an **information-preservation** term — model the full-data
density through an *invertible* summary (exact change-of-variables, floor
`H(X|θ)`), which forces `s` to retain all of `X` — breaks the degeneracy. NF-MLE on
a *summary* does **not** suffice (its floor `H(T|θ)` is attainable by a collapsed
`T`); only full-data exact-density does. This is precisely the empirical split:
the I-A invertible/exact-density arm recovered σ; the II-A calibration-only arm
collapsed it.

**Resolution of #3.** The floor-free coverage objective is *correct* and
*well-posed on a fixed summary* (use it for μ₂), but *degenerate on a learned
summary*. Any learned-summary S2 objective must be **uniformity (§6½) + an
information/efficiency term (Cor 7′)** — efficiency is not polish, it is what makes
the joint problem well-posed. This redraws the §7 construction: step 3's
"efficiency objective" is **mandatory for validity-of-the-procedure**, not just for
sharpness. *(Witnessed, as an illustration only, by
`test_thm7_collapse_exact_coverage_zero_power`: a θ-independent χ²_d statistic
yields coverage ≈ α with whole-space-or-empty sets. The proof, not the test, is the
result.)*

## 9½. S2b (part 1): the minimax–average gap, solved in closed form (#2)

Thm S2 (§6½) is conditional; S2b asked for the *substance* — a bound on
`ε* = inf_ℛ sup_θ R` in a finite-capacity class and a proof that minimax strictly
beats the ρ-average. Here is that theorem for the smallest non-trivial class, with
everything in closed form.

**Setup.** True model `T|θ ~ N(0, v(θ))`, with `v(Θ) ⊆ [a,b]`, `0<a<b`. Constrained
class `ℛ = { r_s(θ;T) = T/s : s>0 }` — a **single shared scale**, the capacity
limit (it cannot adapt `s` to `θ`; the stand-in for the μ₂ ctx-MLP failing in the
tails). The model density is `q_θ = N(0,u)`, `u=s²`, and the per-θ regret is
`R(v;u) = KL(N(0,v)‖N(0,u)) = ½(v/u − 1 + log(u/v))`.

**Theorem 8 (closed-form minimax–average separation). [proven]**
1. *NF-MLE / ρ-average optimum:* `u_avg = argmin_u E_{θ∼ρ} R = E_ρ[v(θ)]` — the
   **arithmetic mean** of the variance under ρ.
2. *Minimax optimum:* `R(·;u)` is strictly convex in `v`, so `sup_{v∈[a,b]}R` is at
   an endpoint, and `u* = argmin_u max(R(a;u),R(b;u))` solves `R(a;u*)=R(b;u*)`,
   giving `u* = (b−a)/log(b/a)` — the **logarithmic mean of the variance *range***
   (endpoints only), with `ε* = R(a;u*) = ½(a/u* − 1 + log(u*/a)) > 0`.
3. *Strict separation:* `M(u):=max(R(a;u),R(b;u))` is strictly quasiconvex with a
   unique min `ε*` at `u*` (and a *corner* there). Hence for every `u_avg ≠ u*`,

       sup_θ R(θ; r_avg) = M(u_avg) > ε* = sup_θ R(θ; r_mm),

   strictly, with the *exact* gap `R(a;u_avg)−R(a;u*)` (if `u_avg≥u*`) or
   `R(b;u_avg)−R(b;u*)` (if `u_avg≤u*`). With Thm S2, the minimax pivot's uniform
   coverage certificate `√(ε*/2)` is strictly below the average pivot's.

**Reading.** *NF-MLE tracks the prior **mean**; minimax tracks the prior's
**range**.* The average solution is mistuned for uniform coverage by exactly the
gap between the arithmetic mean of the bulk and the logarithmic mean of the
extremes — the average-vs-sup pathology, now exactly quantified. (Witness `a=1,
b=4`, tail-underweighting ρ: `u_avg=1.12, u*=2.164, ε*=0.117`, average worst-regret
`0.649` — a 5.5× regret gap; the earlier 2× *coverage* gap is this compressed
through the χ² CDF.)

**Theorem 9 (the general principle: minimax = least-favorable Bayes). [proven]**
In the precision parameter `η=1/u`, `R(v;·)` is convex and the class is a convex
compact `η∈[1/b,1/a]`; `E_π R` is linear (concave) in the prior `π`. By Sion,

    ε* = inf_r sup_θ R = inf_η sup_π E_π R = sup_π inf_η E_π R = inf_η E_{π_LF} R,

so `ε*` is the **least-favorable Bayes regret**. The Bayes-optimal `u` under any
`π` is `E_π[v]`; the least-favorable `π_LF` is the **two-point prior on the extreme
variances `{a,b}`** with weight `w=(u*−a)/(b−a)` (verified: `E_{π_LF}[v]=u*`).
**Corollary.** Plain NF-MLE under training prior `ρ` is uniform-coverage-optimal
**iff `ρ=π_LF`**; for any other `ρ` there is a strict worst-case gap, closed by
reweighting `ρ → π_LF` (concentrate on the regret-extreme `θ`) or by solving the
minimax directly. This is the *principled* form of "importance-weight toward the
tails."

**What #2 closes, and what S2b still leaves open.** Closed (for this class, and by
Thm 9 for any class where KL is convex in the natural parameter — the
exponential-family-summary case): a **bound on `ε*`** (exact), the **strict
minimax>average separation**, and the **optimal reweighting** (`π_LF`). Still open:
a **tractable estimator** that finds `π_LF` / solves the minimax for a *real flow
class* (not the 1-parameter toy), and its **convergence** — that is the remaining
half of S2b, now sharply posed as "estimate the least-favorable prior."

*Closed-form checks (not proofs):* `test_thm8_minimax_average_closed_form`,
`test_thm9_sion_least_favorable` verify `u*`, `ε*`, the balance `R(a;u*)=R(b;u*)`,
the Sion value `sup_π inf_η E_π R = ε*`, and `E_{π_LF}[v]=u*`.

## 10. Honest status of the whole goal

State it plainly so we don't re-fool ourselves:

- **Population optimum (C2 exact):** exact conditional coverage everywhere —
  *proven* (Thm 1, with (A)+Lemma 0), but counterfactual (never exactly attained).
- **Finite samples, ρ-average:** `E_ρ[coverage error] ≤ √((L−H̄)/2)` — *proven*
  (Thm 5).
- **Finite samples, pointwise/uniform ("everywhere"):** **OPEN.** This is the
  property we actually promised. Thm S2 reduces it to driving `sup_θ R` small;
  bounding `ε*` and exhibiting an estimator that does so (S2b) is unproven, and on
  a learned summary the natural objective collapses unless paired with an
  efficiency term (Thm 7). 

So: **we have not proven CD-SBI guarantees conditional coverage everywhere at
finite samples.** We have proven the two endpoints and reduced the middle to two
named, open problems (S2b realizability+estimator; S4 efficiency/anti-collapse).

---

## 11. Foundations for S4: efficiency = Fisher-information preservation

S4 rested on two claims I flagged as conjectural — (1) a score-spanning summary is
asymptotically sufficient; (2) information preservation yields efficiency. **They
are the same theorem**, and the unifying object is Fisher information. This section
proves the foundation and thereby specifies S4's objective exactly.

Notation: full-data score `U_X(θ)=∇_θ log p(X|θ)`; summary score
`U_T(θ)=∇_θ log p(T|θ)`, `T=s(X)`; Fisher infos `I_X(θ)=Cov_θ U_X`,
`I_T(θ)=Cov_θ U_T` (mean-zero scores under regularity).

**Lemma 10 (summary score = conditional mean of the full score). [proven]**
*Under standard regularity (differentiation under the integral),
`U_T(θ) = E_θ[ U_X(θ) | T ]`.*
*Proof.* `p(T|θ)=∫_{s(x)=T} p(x|θ)\,dν`; `∇_θ p(T|θ)=∫ U_X\,p(x|θ)\,dν =
p(T|θ)\,E_θ[U_X|T]`; divide by `p(T|θ)`. ∎

**Theorem 10 (Fisher data-processing; efficiency ⟺ sufficiency). [proven]**
*`I_X(θ) = I_T(θ) + E_θ[ Cov_θ(U_X | T) ] ⪰ I_T(θ)` (Loewner), with equality iff
`Cov_θ(U_X|T)=0` a.s. iff `T` is sufficient (Fisher–Neyman, regular family).*
*Proof.* Law of total covariance for `U_X` given `T`, with `E[U_X|T]=U_T`
(Lemma 10): `Cov(U_X)=Cov(E[U_X|T])+E[Cov(U_X|T)] = I_T + (\text{PSD})`. ∎

**Theorem 11 (asymptotic set-volume `= det I_T^{−1/2}`). [proven for Gaussian/LAN;
leading-order in general]**
*Under LAN at `θ` (DQM regularity, `n` iid replicates, `n→∞`), the
population-calibrated CD-SBI set on summary `T` is, to leading order, the ellipsoid*

    log |C_α(X)| = −½ log det I_T(θ) + (d/2) log χ²_{d,α} + log V_d + o(1),
    V_d = π^{d/2}/Γ(d/2+1).

*Proof.* In the Gaussian-shift limit `T ~ N(θ, I_T^{-1})`, the calibrated pivot is
`r(θ;T)=I_T^{1/2}(θ−T)` (pushes `T` to `N(0,I_d)`), so `|det ∂r/∂θ| = det I_T^{1/2}`.
Change of variables `θ↦r` at fixed `T`: `|C_α| = ∫_{‖z‖²≤χ²_{d,α}} |det ∂θ/∂r|\,dz
= V_d\,(χ²_{d,α})^{d/2}\,det I_T^{-1/2}`. Exact for Gaussian / regular exponential
families; leading-order under LAN (conditionals → Gaussian, pivot → affine). ∎

*Numerically sharpened (and made honest by the witness):* the per-coordinate
identity `E log|∂r_k/∂θ_k| = ½ log I_kk` is **exact for a location coordinate at
every `n`** (the pivot is Gaussian) but **only asymptotic for a scale coordinate**,
with an `O(1/n)` χ²→Gaussian correction. For `N(μ,σ²)` it is exact for `μ`; the `σ`
deficit measured `0.194` at `m=5` and `0.090` at `m=10` — halving as `n` doubles,
confirming the `O(1/n)` rate. So Thm 11 holds exactly on Gaussian directions and
to leading order elsewhere, as stated.

**Corollary 12 (the unification — what S4 optimizes). [proven]**
By Thms 10–11, the excess log-volume over the oracle (full data) is

    E log|C_α|(T) − E log|C_α|(X) = ½ log( det I_X(θ) / det I_T(θ) ) ≥ 0,

zero iff `T` sufficient. Therefore:

- **Validity is independent of `I_T`** — any `(A)`-summary covers (Thm 1).
- **Efficiency ⟺ `I_T(θ)=I_X(θ) ∀θ` ⟺ `T` captures the score** (Lemma 10: residual
  score `Cov(U_X|T)=0`) **⟺ `T` sufficient.** This single equivalence **is** both
  Foundation 1 (score-capture ⟹ asymptotic sufficiency) and Foundation 2
  (information preservation ⟹ efficiency).
- **Collapse (#3) is the singular limit:** a summary blind to a parameter
  direction has `I_T` singular there ⟹ `det I_T→0` ⟹ infinite volume ⟹ the `{∅,Θ}`
  set. "Information preservation" `=` `det I_T` bounded away from `0` `=`
  **Fisher**-information preservation. (Note: this is *Fisher*, not Shannon, info —
  the relevant quantity is the score covariance, not the data entropy.)

**This specifies S4 exactly.** Learn `s_φ` (with an injective sign-correct flow for
C1/validity) by **(a)** NF-MLE on `(θ, s_φ(X))` for validity/C2 **plus (b)** a
score-capture term driving `I_{s_φ}(θ) → I_X(θ)` (equivalently minimizing the
residual score variance `E_θ Cov_θ(U_X | s_φ)`) — which is the *proven* anti-collapse
mechanism (it forbids `I_{s_φ}` singular). Coverage (a) and efficiency (b) are the
two terms #3 proved are *both* required.

**What remains genuinely open in S4:**
- *Existence (finite n).* For non-exponential families no `d`-dim `T` has
  `I_T=I_X` exactly (PKD / Thm 10 equality ⟺ sufficiency); exact efficiency is
  **asymptotic** (`n→∞`, `s_φ→θ̂` the MLE — classical LAN, van der Vaart Ch. 7–9,
  *assembled not reproven*). Bounding the irreducible finite-`n` gap
  `½ log(det I_X/det I_{s_φ})` for the best `d`-dim summary is open.
- *Estimator.* A tractable score-capture loss (score matching / Fisher divergence
  are candidates) **and a proof its optimizer attains `I_{s_φ}→I_X`**. Open.
- *Joint with S2b.* The efficiency term changes the regret landscape, so the
  least-favorable-prior reweighting (Thms 8–9) must be solved *jointly* with
  score-capture. Untouched.

*Witnesses (not proofs):* `test_thm10_fisher_data_processing`
(`I_{first-m}=(m/n)I_X ≺ I_X`, strict) and
`test_thm11_fisher_exact_for_location_asymptotic_for_scale` (location coord matches
`½log I` exactly; scale-coord deficit positive and `O(1/n)`).

---

## 12. S4: the construction (objective + population optimum)

§11 turns S4 from "find a score-spanning summary" into a concrete objective. Over
summary params `φ` and flow params `ψ` (flow injective + sign-correct ⟹ C1):

    L(φ,ψ) = E_{θ∼ρ} KL( p(·|θ) ‖ q_ψ(·|θ;φ) )        [V: validity = NF-MLE]
           − λ · E_{θ∼ρ} log det I_{φ,ψ}(θ),           [E: efficiency]

where `q_ψ(T|θ;φ)` is the flow density of `T=s_φ(X)` and
`I_{φ,ψ}(θ)=Cov_θ(∇_θ log q_ψ(s_φ(X)|θ))` is the summary's model Fisher info.

**The key point — `E` is not an ad-hoc regularizer, it is the set volume.** By
Thm 11, `E_θ log det I_T = −2·E_θ log|C_α| + const`, so maximizing `log det I`
*is* minimizing the expected log-volume of the confidence set. Both terms are
**likelihood-free** (use `q_ψ` + simulator draws; never `p`'s density).

**Theorem 13 (S4 population optimum — realizable case). [proven]**
*Assume (A), and (i) the flow class can represent the KR map of `p(s_φ(X)|θ)` for
each `φ`, and (ii) a `d`-dim sufficient statistic exists (e.g. exponential family).
Then any minimizer `(φ*,ψ*)` of `L` (`λ>0`) is **valid and efficient**:*
- *`V=0 ⟹ q_{ψ*}=p ⟹` C1+C2 ⟹ exact validity (Thm 1);*
- *`s_{φ*}` is **sufficient**, so `I_{φ*}=I_X` and `C_α` has oracle (minimal)
  expected log-volume (Cor 12).*

*Proof.* `V≥0`, `=0` iff `q=p` (each `θ`), achievable for any `φ` by (i)+Lemma 0;
so every minimizer has `V=0`. Given `V=0`, `I_{φ,ψ}` is the true Fisher info of
`s_φ`, `⪯ I_X` (Thm 10), so `E_θ log det I ≤ E_θ log det I_X` with equality iff
`s_φ` sufficient `∀θ`; a sufficient `s_φ` exists by (ii), so the `−λE` term is
minimized exactly at sufficiency. Hence `s_{φ*}` sufficient. ∎

**This is the anti-collapse construction (#3 made concrete).** Coverage alone (`V`)
admits the `det I→0`, `{∅,Θ}` collapse (Thm 7); `E = −2·E log|C_α| → +∞` under
collapse, so it forbids exactly that. Coverage + log-volume are the two terms #3
proved necessary.

**Open (the genuine S4 build + remaining theory):**
- *Non-exp-family / finite `n`:* (ii) fails (PKD) — `max_φ E log det I_{s_φ} <
  E log det I_X`; the gap is the irreducible finite-`n` inefficiency, `→0` as
  `n→∞` with `s_φ→` MLE (Foundation 1, LAN). **Bounding it is open.**
- *Estimator + convergence:* `I_{φ,ψ}(θ)` is a covariance of the flow's `θ`-score
  (autodiff + MC); the joint nonconvex landscape's convergence to `(φ*,ψ*)` is
  unproven, and `E`'s gradient is biased while `q_ψ≠p`. **Open.**
- *Joint with S2b:* `E` reshapes `R(θ;·)`, so least-favorable reweighting (Thms
  8–9) must be solved *with* score-capture. **Untouched.**
- *Build:* `s_φ` (DeepSets/score-net) + injective flow + the two-term loss;
  validate `I_{s_φ}→I_X` and sets `→` oracle on (μ,σ²), then the d=5 Bartlett
  target as the **non-oracle replacement**. This is the experiment.

S4 is now a concrete objective with a proven population optimum and a named list of
finite-`n`/estimator gaps — no longer an open conjecture, but a build with a
theory spine.

---

## 13. Closing the remaining theory gaps (airtight)

Making rigorous — or honestly bounding as open — every piece §11–12 left hanging.
Tags: **[proven here]**, **[assembled from classical, cited]**, **[open]**.

### 13.1 Foundation 1, stated precisely (asymptotic sufficiency) [assembled, cited]
*Hypothesis (DQM).* `X_1,…,X_n` iid `f(·|θ)`; `√f` differentiable in quadratic mean
at `θ` with nonsingular, continuous Fisher information `i(θ)`.
*Theorem (Hájek–Le Cam).* Under DQM the localized experiments `E_n =
{f^{⊗n}(·|θ+h/√n)}_h` converge to the Gaussian shift `G = {N(h, i(θ)^{-1})}_h`; the
central sequence `Δ_n = n^{-1/2} i(θ)^{-1}Σ_i ∇_θ\log f(X_i|θ)` is asymptotically
sufficient, and the MLE satisfies `√n(θ̂_n−θ) = Δ_n + o_p(1)`, hence is
asymptotically sufficient. *(van der Vaart, Asymptotic Statistics, Ch. 7–9; Le
Cam. We invoke, not reprove.)*

### 13.2 The finite-`n` efficiency gap [13(a) proven here; 13(b) assembled]
**Theorem 14.**
*(a) Exponential family* (full rank, `η` a diffeomorphism): `T=Σ_i t(X_i)` is
sufficient for every `n`, so `I_T = I_X` and the gap is `0` — **exactly, no
asymptotics.** (Fisher–Neyman + Thm 10.) This is why oracle Bartlett at `d=5` was
exact.
*(b) Regular non-exponential family:* for the MLE summary,
`Δ(θ;n) := ½ log(det I_X / det I_{θ̂_n}) → 0` as `n→∞`, at rate `O(1/n)` —
`Δ ≈ (2n)^{-1} tr(i(θ)^{-1}Γ(θ))` with `Γ` the statistical-curvature tensor
[Efron 1975; Rao 1963]. The best `d`-dim summary has gap `≤` the MLE's, so a
score-capturing CD-SBI summary is **asymptotically oracle-efficient, exactly so for
exponential families.**
*Proof of `→0`.* In the limit experiment `G` the central sequence is sufficient
with full info `i(θ)`; `θ̂_n`'s local version is asymptotically `N(h, i^{-1})`,
capturing `i(θ)`, so `det I_{θ̂_n}/det I_X → 1`. ∎ (Rate cited.)

### 13.3 S4 decoupling, and the estimator bias is score-level not KL-level
**Proposition 15 (validity ⊥ efficiency decouples). [proven here]**
`{ψ : V(φ,ψ)=0}` is nonempty for every `φ` (Lemma 0), and on it
`E = E_θ log det I_{s_φ}(θ)` — the *true* Fisher info of `s_φ` (since `q_ψ=p`),
independent of `ψ`. Hence `inf_{φ,ψ} L = inf_φ (−λ E_θ log det I_{s_φ})`: efficiency
selects `φ`, validity calibrates `ψ` to it, **with no population-level trade-off**.
So S4 is a clean two-level problem, not a fragile `λ`-balance. ∎

**The training bias is controlled by the score, which KL does not control. [proven
here — negative result]** While `q_ψ≠p`, the efficiency term uses
`Ĩ=Cov(∇_θ\log q_ψ)`, biased from `I=Cov(∇_θ\log p)` by the score error
`∇_θ\log q_ψ − ∇_θ\log p`. This error is **not** bounded by `KL(p‖q_ψ)`: with
`q ∝ p·(1+ε\sin(ωθ\text{-dir}))`, `KL = O(ε²)` for *all* `ω` while the score error
`~ εω → ∞` (verified: `KL≡0.00251`; score-divergence `0.02, 2.0, 50.4` at
`ω=2,20,100`). **Consequence:** validity (NF-MLE) does not by itself make the
efficiency estimate consistent — **S4 needs explicit score control (a Fisher-
divergence / score-matching component), not just `log q`.** This is a concrete
design constraint on the build, proven rather than guessed.
*(Witness:* `test_thm13_kl_does_not_control_score`.*)*
**[Open]** a score-controlled estimator with a convergence guarantee on a nonconvex
flow class — no general guarantee exists for nonconvex NN optimization; empirical.

### 13.4 S2b estimator (convex case) [proven here]
**Proposition 16.** With `R` convex in the natural parameter (the realizable
exp-family-summary regime): the minimax saddle exists (Sion, Thm 9); the
least-favorable prior `π_LF` is supported on the maximizers of `θ ↦ R(θ;η*)`
(complementary slackness) — finite generically, and exactly the *two extreme
variances* in the single-scale family (§9½). The exponentiated-gradient /
mirror-prox iteration `π_{t+1} ∝ π_t e^{γR(·;η_t)}`, `η_{t+1}=Bayes(π_{t+1})`,
converges to the saddle at the standard convex–concave rate. **[Open]** nonconvex
flow class — no general guarantee.

### 13.5 S5 (R1-realizability) reduced to a monotone-likelihood-ratio condition
**Proposition 17. [proven here]** If each conditional `T_k | (θ, T_{<k})` has
monotone likelihood ratio (MLR) in `θ_k`, then `F(t|θ_k,·)` is monotone in `θ_k`,
so the calibrated pivot `r_k = Φ^{-1}(1−F)` is monotone in `θ_k` and R1 holds. MLR
holds for exponential families (natural parameter) and location/scale families ⟹
**R1 is realizable there** (the Gaussian/Bartlett pivots). **[Open]** non-MLR /
multimodal conditionals — R1 fails, but **validity survives** (§1): the calibrated
pivot is non-monotone in `θ`, yielding the honest disconnected sets of the §7
headline. So S5 is not a coverage gap; it is exactly the boundary of the *shape*
axis.

### 13.6 Net status of rigor
Proven here: 14(a), 15, the score-vs-KL bias, 16 (convex), 17 (MLR). Assembled from
classical asymptotics (cited, not reproven): Foundation 1, 14(b). **Irreducibly
open: only optimization/architecture questions** — (i) a score-controlled S4
estimator with nonconvex-convergence, (ii) S2b on a nonconvex class, (iii) R1
beyond MLR. **The inferential chain itself — validity (Thm 1) → finite-sample
certificate (Thm 5) → uniform coverage (Thms S2, 8–9) → efficiency = Fisher-
information preservation (Thms 10–12) → realizable population optimum (Thm 13) — is
airtight.** What remains is not whether the logic is sound, but whether training
attains the proven population optima — an empirical question, correctly outside the
scope of proof.

---

## 14. Rounding out: consolidations to keep the chain clean

Three loose ends in *our* chain (not the cited asymptotics), made airtight before
the build.

### 14.1 The NF-MLE loss certifies coverage only relative to the summary's *own* floor
Thm 5 is `coverage-err ≤ √((L−H̄)/2)` with `H̄(φ)=E_θ H(s_φ(X)|θ)`. Fixed summary:
clean. *Learned* summary: `H̄(φ)` moves, and
**Proposition 18. [proven]** `L(φ,ψ) = H̄(φ) + E_θ KL(p_θ‖q_{ψ,θ})`, so the raw
loss `L` is **not comparable across summaries** — it drops when the summary's
conditional entropy `H̄(φ)` drops, with *no* coverage improvement. The M2 cheat
("naive NF-MLE collapses σ, loss 5.3 below the fixed floor") is exactly
`H̄(φ_collapsed) < H̄(φ_ref)`. Therefore (i) the certificate must use the
*per-summary* floor `H̄(φ)`, never a fixed reference; (ii) the principled
cross-summary regularizer is the Fisher term (§12), which targets *information* and
so cannot be gamed by entropy reduction. This is #3's collapse seen from the floor
side — independent corroboration that S4's two terms are both necessary.

### 14.2 The volume formula is unambiguous in its regime (radial vs Gaussian)
Validity needs only the *radial* law (Prop 1′), yet efficiency (Thm 11) uses full
Gaussianization — which governs? **Resolution. [proven / cited]** `C_α` depends on
the pivot only through the scalar field `θ↦‖r(θ;T)‖²` (so a data-dependent rotation
`R(T)·r` leaves the set unchanged). Thm 11 lives under LAN, where the calibrated
pivot is *asymptotically unique* (Gaussian-shift KR; manuscript Thm A-d) — there
"radially calibrated", "fully Gaussianized", and "the calibrated pivot" coincide,
so `det I_T^{-1/2}` is *the* volume, not a bound. The radial NSC (strictly weaker)
governs the complementary regime (finite-sample non-regular / multimodal), where
uniqueness fails and the content is validity + honest disconnected sets (§7), not
the volume formula. **Validity-theory (radial, broad) and efficiency-theory
(Gaussian, LAN) occupy compatible but distinct regimes; neither undercuts the
other**, and the build uses full NF-MLE, which in the regular regime delivers both.
*(Witness `test_thm11_gaussian_volume_exact`: the 2-D Gaussian-summary set volume
`= V_2 χ²_{2,α} det(Σ_T)^{1/2} = V_2 χ² det I_T^{-1/2}` — predicted 19.139, grid
19.142.)*

### 14.3 S2b and S4 do not conflict — they nest
**Proposition 19 (separation). [proven, population level]** For *any* summary `φ`,
uniform exact validity `sup_θ R=0` is achievable (per-θ KR, Lemma 0); on that
manifold efficiency depends only on `φ` (Prop 15). So the joint problem nests
cleanly: *outer* = efficiency over `φ`; *inner* = S2b least-favorable-reweighting
calibration over `ψ`. The only population coupling is `π_LF = π_LF(φ)` (the
least-favorable prior tracks the summary's induced conditionals). The
finite-capacity coupling is exactly the two already-isolated open items — the score
bias (§13.3, `q_ψ≠p`) and `ε*(φ)>0` (§9½) — **not a new conflict.** So S2b and S4
compose; they do not fight.

### 14.4 Regularity checklist for the build
What the S4 build must ensure for the chain's hypotheses to hold:
- **(A)** `s_φ` induces abs-cont, full-support conditionals (smooth `s_φ`; no
  rank-deficient head) ⟹ C1 / Lemma 0 / validity.
- **nonsingular `I_{s_φ}(θ)`** (the Fisher term enforces it) ⟹ non-collapse +
  (A)-compatibility.
- **score control** on `∇_θ log q_ψ` (§13.3) ⟹ consistent efficiency gradients.
- **MLR conditionals** (§13.5) *iff* nested-set shape (R1) is wanted — optional,
  never required for coverage.

---

## 15. Multimodal: the impossibility is false; R1 is the real (removable) obstruction

We tried to prove the closed-form-CD framework cannot handle multimodal posteriors.
The attempt **refutes** the strong claim and isolates a narrow, removable obstruction.

**Counterexample (multimodal handled exactly). [witnessed]** Sign-unidentifiable
model `X̄|θ ~ N(θ², σ²/n)` (θ and −θ observationally identical ⟹ bimodal posterior).
The **1-dim sufficient** summary `T=X̄` and the calibrated pivot
`r(θ;X̄)=√n(X̄−θ²)/σ` give: `r(θ₀;X̄)~N(0,1)` (valid; KS p≈0.4–0.8), coverage exactly
α, and `C_α = {θ: θ²∈X̄±zσ/√n}` = **two disjoint intervals** `±[√(X̄−·),√(X̄+·)]`
whenever `X̄>zσ/√n`. **Valid, efficient (sufficient summary), correctly disconnected
— a fixed-dim CD-SBI construction on a multimodal model.** So both "multimodal ⟹
impossible" and the §7-era overclaim "the d-dim summary bottleneck *is* the
regularity assumption" are **false** (the latter is herewith retracted).

**Theorem 15 (R1 ⟹ connected sets). [proven]**
*If for fixed `T` the pivot `r(·;T): Θ → ℝ^d` is a homeomorphism onto `ℝ^d` — the
property R1 (monotone-triangular) + calibration guarantees (the Gaussianizing pivot
covers `ℝ^d` as θ ranges over `Θ≅ℝ^d`) — then `C_α(X)=r(·;T)^{-1}(B_α)` is the
continuous image of the connected ball `B_α` under the continuous `r^{-1}`, hence
connected (contractible).* ∎ So an R1 pivot **cannot** produce disconnected sets;
multimodal inference requires `r` non-injective in θ, i.e. **dropping R1**. (The
counterexample's pivot is non-injective `r(θ)=r(−θ)` and not onto `ℝ^d` — it escapes
the hypothesis, which is exactly why it is disconnected.)

**The corrected map:**
- **The casualty is the nested proper-CD (R1), not validity, not the framework.**
  Validity never needed R1 (Thm 1); dropping R1 keeps exact coverage and *gains* the
  disconnected shape. Plain NF-MLE does **not** impose R1 (it is an *extra*
  architectural constraint), so a flexible flow yields the non-monotone calibrated
  pivot automatically where the model is multimodal. **Dropping R1 is free.**
- **Efficiency for multimodal = sufficiency, the same PKD problem as the regular
  case** — not a new barrier. Fixed-dim sufficient stat exists (sign model: `X̄`) ⟹
  efficient multimodal CD works exactly. Residual hardness: the sufficient summary's
  dimension scales with posterior complexity (~ `#modes × d_θ`); at the unbounded
  extreme one models the full conditional (→ NLE), keeping validity with efficiency
  tracking NLE quality.
- **The learning target changes:** the posterior-*mean* regression summary (the
  regular-case device, §S4) fails for multimodal — it averages modes to mush. The
  right target is a *mode-resolving* sufficient summary (posterior moments, or an
  NLE/conditional-density representation), not `E[θ|X]`.

**Upshot.** Multimodality is *not* a fundamental barrier. Relaxing it = (i) drop R1
(free; validity untouched) + (ii) a mode-resolving sufficient summary (the shared
sufficiency problem, scaling with complexity). The "regular-only" pessimism was an
artifact of bundling R1 — a removable architectural choice — with validity.
*Witness:* `test_thm15_multimodal_handled_by_nonmonotone_pivot`.

---

## 16. The mode-resolving summary: posterior-moment regression (not "just NF-MLE")

§15 left one piece open: the non-oracle learning target for a summary that stays
sufficient under multimodality (the posterior *mean* fails — it averages modes). The
answer is **not** "model the full likelihood by NLE"; it is **posterior-moment
regression**, with moment-order as the modality knob.

**The posterior is the right object. [proven]** With a *known* prior `π`, the
posterior determines the likelihood up to a θ-independent constant:
`p(X|θ) ∝ p(θ|X)/π(θ)`. The likelihood function (up to proportionality) is the
*minimal* sufficient statistic (likelihood-equivalence). So **the posterior is
minimal sufficient**, and any summary determining it is sufficient.

**Moments represent it — non-oracle, scaling with modality. [proven + witnessed]**
For moment-determinate posteriors (compact θ-support — our bounded-prior setting),
`{E[θ^⊗k|X]}_{k≤K}` determine the posterior as `K→∞`; a `K`-mode (≈ `K`-component)
posterior is pinned by `~2K−1` moments. Each `E[θ^⊗k|X]` is estimable by **supervised
regression of `θ^⊗k` on `X`** (low-dim target, no likelihood). So **moment-order `K`
is the modality knob.** Witness (sign model, bimodal): the 1st moment `E[θ|X̄]` has
`R²=0.0006` (useless — averages ±modes to 0); the 2nd moment `E[θ²|X̄]` has `R²=0.99`,
`corr=0.999` with the sufficient stat `X̄` — the 2nd moment recovers the sufficiency
the mean discarded.

**The unified construction (one method, two knobs):**
- *Summary:* `T = (Ê[θ|X], Ê[θ^⊗2|X], …, Ê[θ^⊗K|X])` — regressed posterior moments.
- *Pivot:* NF-MLE flow on `(θ,T)`; **R1 ON** (regular: connected sets, `K=1–2`) /
  **OFF** (multimodal: disconnected sets, larger `K`, §15).
- *Regular* (BvM, near-Gaussian posterior): `K=1–2` (mean+cov) suffices — the
  sequential regression-summary strategy is this special case.
- *Multimodal:* more moments resolve modes; R1 off ⟹ disconnected sets.
- *Degrades to NLE / full conditional* only when the posterior is genuinely
  moment-indeterminate (very complex likelihood).

**Why this is not "just NF-MLE."** The summary is *moment regression* (`X →`
posterior moments, supervised) — which scales far better than NLE density-estimation
on high-dim `X` (the OOM bottleneck). NF-MLE is only the *calibration layer* on the
low-dim moment summary. The construction composes an NPE-style posterior summary
(cf. Fearnhead–Prangle's mean; Waldo's moments) with an NF-MLE-calibrated
*non-monotone* pivot — and the synthesis delivers what neither piece does alone:
**exact frequentist coverage + correct disconnected shape + efficiency, with a theory
of when (sufficiency) and how it scales (moment-order ~ modality).** NPE alone has no
coverage; NLE alone has only asymptotic coverage and no reduction.

**The `d_θ`-pivot cap (important refinement).** The NF-MLE pivot is
dimension-preserving, so the clean construction uses a **`d_θ`-dim** summary —
*one* `d_θ`-dim moment/invariant target `m(θ)` (e.g. `θ²` for the sign model:
`d_θ=1`), not a stack of `K` moments. So "moment-order scales with modality" holds
for *sufficiency* but the clean pivot caps at `d_θ`; symmetry-induced multimodality
whose minimal sufficient reduction is `d_θ`-dim (sign model) fits, and the knob is
*which* `d_θ`-dim target (detected by regression informativeness). Genuine
multimodality with sufficient dim `> d_θ` exceeds the clean pivot and requires the
LF2I extension (richer summary + Neyman-calibrated *global* LRT — not the local
score). So the §16 "stack `K` moments" picture is the *sufficiency* statement; the
*constructive* one is "regress the right `d_θ`-dim target."

**Open:** moment-indeterminate / very-complex posteriors (→ NLE/LF2I fallback);
high-order moment-regression variance (low orders well-estimated, high orders noisy
⟹ graceful degradation, not a cliff); the `m(θ)`-target / `R1`-on-off selection rule
(regression-R² as a modality detector). *Witness:*
`test_thm16_posterior_moment_recovers_sufficiency`.

---

### Changelog
- 2026-05-31: initial note. Theorems 1, 5, Lemma 3, Props 2, 6 proven; non-oracle
  construction (§7) and S2/S4/S5 flagged conjectural/open. Validators scaffolded
  in `tests/theory/`.
- 2026-05-31 (S3): Prop 2 (validity ⟂ sufficiency) validated — analytic
  sub-sample-pivot tests (exact coverage from a non-sufficient summary + the
  efficiency cost via E[log|det ∂r/∂θ|]) and a trained lossy-summary flow that
  calibrates ≤0.06 at all probes. Validity is φ-free, confirmed.
- 2026-05-31 (S2): added §6½ Theorem S2 (minimax regret ⟹ uniform coverage) with
  Routes 1/2 (known/asymptotic floor vs floor-free flow-only) and the II-A caveat.
  Closed-form witness (`test_s2_minimax_controls_sup_coverage`): minimax halves the
  sup-coverage error vs ρ-average at one instance.
- 2026-05-31 (rigor pass, after pushback that tests ≠ proofs): (1) added the missing
  hypothesis (A) + Lemma 0 (KR existence) so Thm 1 / "any summary" are not vacuous;
  (2) added Prop 1′ (coverage ⟺ radial χ²_d law — strictly weaker than
  Gaussianization); (3) **retracted** the Neyman–Pearson "proof" of efficiency,
  replaced by Prop 2′ (Fubini: efficiency = power) and demoted "sufficient ⟹
  smaller sets" to conjectural-in-general; (4) honestly split S2 into S2a (proven,
  easy implication) and S2b (open substance); downgraded the "verified" labels;
  (5) added §9 Theorem 7 (#3): the floor-free coverage objective is correct +
  well-posed on a fixed summary but **collapses on a learned one** (exact coverage,
  zero power) — so coverage must be paired with an efficiency/information term;
  (6) added §10 honest whole-goal status (everywhere-at-finite-n is OPEN).
- 2026-05-31 (#2): §9½ Thms 8–9. Closed-form minimax–average separation for the
  single-scale family: NF-MLE optimum = arithmetic mean E_ρ[v]; minimax optimum =
  logarithmic mean of the variance *range* (b−a)/log(b/a); ε* explicit; strict gap
  whenever E_ρ[v]≠u*. General principle (Sion): ε* = least-favorable Bayes regret;
  NF-MLE-uniform-optimal iff ρ=π_LF (= 2-point prior on the extreme variances) ⟹
  "reweight toward tails" = "reweight ρ→π_LF". Closes the ε*/separation/reweighting
  half of S2b; tractable estimator + convergence for a real flow class still open.
- 2026-05-31 (S4 foundations): §11. The two conjectural foundations (score-spanning
  ⟹ asymptotic sufficiency; information preservation ⟹ efficiency) UNIFIED as one
  Fisher-information theorem. Lemma 10 (U_T=E[U_X|T]); Thm 10 (Fisher data-processing
  I_T⪯I_X, eq iff sufficient — proven); Thm 11 (asymptotic set-volume = det I_T^{−½};
  witness showed EXACT for location, O(1/n) for scale); Cor 12 (validity ⊥ I_T;
  efficiency ⟺ I_T=I_X ⟺ score-capture ⟺ sufficient; collapse = det I_T→0). S4's
  objective is now exactly specified: NF-MLE (validity) + score-capture (efficiency
  = proven anti-collapse). OPEN: tractable score-capture estimator + convergence;
  finite-n efficiency gap bound; joint solve with S2b reweighting.
- 2026-05-31 (S4 construction): §12. Concrete objective L = NF-MLE (validity) − λ·
  E log det I_{s_φ} (efficiency), where the efficiency term IS −2·E log-volume
  (Thm 11) — likelihood-free. Thm 13: in the realizable/exp-family case the
  population minimizer is valid (Thm 1) AND efficient (sufficient, Cor 12); the
  log-det-Fisher term is the concrete anti-collapse mechanism #3 demanded. S4 is now
  a build with a proven population optimum; open = estimator/convergence, finite-n
  gap, joint S2b, and the actual training experiment on (μ,σ²)→d=5 Bartlett.
- 2026-05-31 (airtight pass, §13): closed the §11–12 gaps. PROVEN here: Thm 14(a)
  exp-family exact-efficiency (gap 0 ∀n); Prop 15 (validity⊥efficiency decouples,
  no λ-trade-off); the **score-vs-KL negative result** (KL does NOT control the
  score ⟹ S4 needs explicit score control, not just NF-MLE — verified ω-witness);
  Prop 16 (S2b convex saddle + π_LF on the regret-extremes + EG convergence); Prop
  17 (R1-realizability ⟸ MLR conditionals; non-MLR is the disconnected-set regime,
  not a coverage gap). ASSEMBLED/cited: Foundation 1 (Hájek–Le Cam), Thm 14(b)
  O(1/n) gap (Efron/Rao curvature). NET (§13.6): the **inferential chain is
  airtight**; the only irreducibly-open items are optimization/architecture (score-
  controlled estimator + nonconvex convergence, S2b nonconvex, R1 beyond MLR) —
  whether training attains the proven optima, an empirical not a logical question.
- 2026-05-31 (§16, mode-resolving summary): the non-oracle target for multimodal
  sufficiency is **posterior-MOMENT regression**, moment-order = modality knob (NOT
  "just NLE"). Posterior is minimal sufficient (known prior); moments determine it
  (moment-determinate); witness — sign model 1st moment R²=0.0006 (useless) vs 2nd
  moment R²=0.99/corr 0.999 recovers the sufficient stat. Unified construction =
  moment-regression summary + NF-MLE non-monotone pivot; R1-on/K-small = regular
  (the sequential strategy is the special case), R1-off/K-larger = multimodal.
  NF-MLE is only the calibration layer; the summary is NPE-style regression.
- 2026-05-31 (§15, multimodal — tried to prove impossibility, REFUTED it):
  counterexample (sign model `X̄|θ~N(θ²,σ²/n)`: 1-dim sufficient summary +
  non-monotone pivot ⟹ valid + efficient + correctly disconnected). **Retracted** the
  §7-era overclaim "summary bottleneck = regularity". Thm 15 (proven): R1 ⟹ `r(·;T)`
  homeomorphism onto `ℝ^d` ⟹ `C_α` connected; so multimodal needs only *dropping R1*
  (free — NF-MLE doesn't impose it; validity untouched), and efficiency reduces to the
  shared PKD sufficiency problem (summary dim ~ #modes). Multimodal is NOT a
  fundamental barrier. Witness `test_thm15_multimodal_handled_by_nonmonotone_pivot`.
- 2026-05-31 (rounding out, §14): three consolidations keeping the chain clean.
  Prop 18 (NF-MLE loss certifies coverage only vs the summary's OWN floor H̄(φ);
  raw L gameable across summaries by entropy collapse = the M2 cheat = #3 from the
  floor side ⟹ Fisher term is the principled cross-summary regularizer). §14.2
  (volume formula unambiguous: set depends only on ‖r‖²-field; LAN uniqueness makes
  radial=Gaussian where Thm 11 lives; radial NSC governs the complementary
  irregular/disconnected regime — witness: exact 2-D Gaussian volume). Prop 19 (S2b
  and S4 NEST, no objective conflict; only coupling = π_LF(φ) + the already-known
  finite-capacity items). §14.4 build regularity checklist. 26 theory tests pass.
