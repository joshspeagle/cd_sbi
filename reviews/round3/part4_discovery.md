# Round 3 — Part IV (Multivariate) Discovery & Assessment

## Phase A: Discovery

### Top-line goal

Part IV lifts the 1D framework of Parts II–III to `d > 1` by adopting
**autoregressive triangular flows** as the architectural class. Within
that class it proves a multivariate analogue of Theorem A (Theorem A-d):
the calibration manifold collapses to a singleton — the
**Knothe–Rosenblatt rearrangement** — and inference reduces to
slice-by-slice 1D root-finding. The Cholesky corollary
`r^KR = L^{-1}(θ − X)` is the closed-form anchor for the
multivariate-Gaussian validation in §8.3.

### Central arguments

1. **A1 — Architectural specialization is necessary.** Without
   architectural constraints, the multivariate calibration manifold is
   not just infinite-dimensional (the 1D failure mode) but also
   rotation-degenerate: any orthogonal transformation of a calibrated
   pivot remains calibrated because `N(0, I_d)` is rotation-invariant
   (§3.3, lines 549–551). The job of Part IV is to pick a structured
   subclass that is large enough to be useful and small enough to make
   the manifold a singleton.

2. **A2 — The autoregressive triangular structure is the right
   subclass.** Each component `r_k` depends only on `(θ_≤k, X_≤k)` and
   is strictly monotone in `θ_k` (R1^auto) and `X_k` (R2^auto). This
   has three payoffs simultaneously: (a) the Jacobian is
   lower-triangular with positive diagonal, so `log|det ∂_X r| = Σ_k
   log|∂_{X_k} r_k|` is `O(d)` and closed-form, (b) the architectural
   class scales linearly in `d`, and (c) the σ-algebra equivalence
   `σ(θ, X_<k) = σ(θ, r_<k)` makes an inductive proof of uniqueness
   possible.

3. **A3 — Two practical parameterizations span the cases that matter.**
   The **additive form** `r_k = a_k − b_k` handles Gaussian-like models
   (used in §8.1–8.3); the **doubly-monotone UMNN form** handles
   non-additive interactions in `(θ_k, X_k)` (used in §8.4 for the
   exponential rate). Both satisfy (R1^auto) + (R2^auto) by
   construction; the KR theorem applies to both.

4. **A4 — Theorem A-d gives a multivariate uniqueness result that
   Theorem A + Theorem C cannot.** Theorem A is intrinsically 1D
   (single pivot `r`, target `N(0,1)`); composing it `d` times gives
   `d` separate 1D uniqueness statements but not a joint uniqueness
   statement on `r = (r_1, …, r_d)` calibrated against `N(0, I_d)`.
   Theorem A-d closes that gap by induction on `k`, with two
   essential ingredients beyond a per-coordinate 1D argument:
   (i) σ-algebra equivalence at previous levels, and (ii) the
   *general-source* form of Lemma 4.2 (with `F_k^{(θ)}(· | X_<k)`,
   not `N(θ, 1)`, as the source measure).

5. **A5 — The Gaussian corollary `r^KR = L^{-1}(θ − X)` is the
   numerical anchor.** It is the unique closed-form benchmark in the
   manuscript for which "did the network find KR?" is verifiable
   coordinate by coordinate. §8.3 reports `E[J_θ] = L^{-1}` to 1–2 %.

6. **A6 — Inference inherits exact frequentist coverage.** With `r`
   calibrated, `‖r(θ_0; X)‖² | θ_0 ~ χ²_d` for every `θ_0`, so
   `C_α(X_obs) = {θ : ‖r‖² ≤ χ²_{d,α}}` is an exact confidence set.
   The set is *connected* under (R1^auto) + `C¹`, and is computed by
   slice-by-slice 1D root-finding rather than a `d`-dimensional grid.

7. **A7 — KR depends on coordinate ordering, but every ordering is a
   valid CD pivot.** The architectural autoregressive order picks one;
   data with natural ordering (time series, spatial sweep) should use
   that order, and unordered data may pick any (open problem, §11.2).

### Supporting evidence per argument

- A1 → §3.3 already establishes the unconstrained-manifold size; §6
  intro recapitulates the implication. Direct.
- A2 → (R1^auto)/(R2^auto) statements + §6.2 Jacobian; the
  load-bearing-clauses paragraph (lines 1309–1315) explicitly names
  the two payoffs of the *dependence* clause and one of the
  *monotonicity* clause.
- A3 → Two explicit definitions in §6.1 (additive eq. line 1333,
  doubly-monotone eq. line 1338); the doubly-monotone form's
  `∂_{X_k} r_k > 0` argument is spelled out in lines 1340–1346.
- A4 → §6.3 proof: σ-algebra equivalence (lines 1432–1439),
  general-source Lemma 4.2 (lines 1447–1460). The "Remark (general
  source)" was added in round 1 (lines 868–878 of §4.2) to support
  this lift.
- A5 → §6.3 Corollary (lines 1462–1495), proof via
  `Z = L^{-1}(X − θ) ~ N(0, I_d)` route rather than the
  Cholesky-regression-coefficient algebra. The 2D numerical
  verification is in §8.3.
- A6 → §6.4 (lines 1545–1568); the §6.2 "Connected CD level sets"
  paragraph (lines 1372–1380) supplies the geometric well-definedness.
- A7 → Remark 1 of §6.3 (lines 1499–1510).

### What Theorem A-d gives that Theorem A + C cannot

A single sentence the reader could be told but currently isn't: a
joint uniqueness statement on `r: Θ × X → R^d` calibrated against
`N(0, I_d)` is genuinely more than `d` separate 1D claims. The 1D
claim at level `k` requires conditioning on `(θ, X_<k)`, and the
inductive argument needs that this is *equivalent* to conditioning on
`(θ, r_<k)` — which depends on (R2^auto) at all previous levels and
on the σ-algebra identity. Theorem A + Theorem C cannot supply that
identity; it is the new ingredient.

---

## Phase B: Accessibility assessment

### B.1 Undefined-or-thinly-defined terms

| Term | First use in Part IV | Defined? | Severity |
|---|---|---|---|
| Autoregressive flow | line 1273 (§6 intro) | One-sentence def. — "the *k*-th output depends only on inputs with index ≤ k, so the Jacobian is lower-triangular." No worked example. | High — this is the central architectural object and the one-sentence definition asks the reader to mentally combine "autoregressive masking" + "triangular Jacobian" + "product determinant" all at once. |
| Triangular Jacobian | line 1275 | Same one sentence — the *why* (chain rule on a function whose `k`-th output ignores `X_>k`) is not unpacked until §6.2. | Medium — the §6.2 unpacking is just a piecewise formula (lines 1355–1358), not a derivation. |
| Knothe–Rosenblatt rearrangement | line 1389 | Defined inline: "the unique transport map of triangular form: each coordinate transformation conditions on previously-transformed coordinates." The *idea* (sweep one coordinate at a time, conditioning on what has already been transformed) is stated but never *drawn* with `d = 2` as a worked example. | High — the name appears 9 times in §6.3 alone. |
| Brenier map | line 1391 | Contrastive definition: "the unique transport map that is the gradient of a convex function (the L²-optimal transport plan). KR depends on coordinate ordering; Brenier does not." Reads as "interesting context," and that is genuinely all it is — Brenier appears nowhere else in Part IV. | Low — but the *role* of the contrast (why mention Brenier at all?) is not stated. Right answer: "to flag that we are *not* using the canonical OT map; we are using a different one that aligns with autoregressive architectures." |
| Conditional CDF `F_k^{(θ)}(· | X_<k)` | line 1403 | Notation introduced via the Theorem A-d hypothesis. No prose unpacking ("the CDF of `X_k` after we condition on `X_1, …, X_{k-1}` at parameter `θ`"). | Medium — the manuscript assumes the reader is comfortable with conditional CDFs as objects to be manipulated. Astronomers may be more comfortable with conditional densities. |
| σ-algebra equivalence | line 1432 | Used as a working tool in the proof. "Conditioning on `r_<k` given `θ` is therefore equivalent to conditioning on `X_<k` given `θ`." | Medium-High — the term σ-algebra appears nowhere else in the manuscript; an astronomer reader will parse this as "is independent of the way we describe the conditioning info" but the manuscript should say that explicitly. |
| Cholesky factorization | line 1464 | Single inline definition: "`LL^T = Σ` (`L` lower-triangular with positive diagonal)." | Low — Cholesky is reasonably standard. But the *role* (`L^{-1}` is the multivariate analogue of dividing by σ in 1D) is not stated. |
| Bijective measure-preserving map | not directly used; *measure-preserving rearrangement* in the (R2^auto) gloss, line 1314 | Not defined. | Low — used only in the gloss; can be removed or replaced with "monotone match-up of the two distributions." |
| (R1^auto) / (R2^auto) | line 1294–1306 | Defined formally with measurability + monotonicity clauses. The *names* echo (R1)/(R2) of §2.3–2.4 but the autoregressive specialization is not motivated beyond "specialization." | High — see B.2 below. |
| UMNN sub-structure | line 1334+ | UMNNs are formally defined in §7.1 (line 1578), *after* Part IV uses them. Forward reference is implicit. | Medium — Part IV would benefit from a one-paragraph teaser of "what a UMNN does (positive integrand → monotone function with closed-form derivative)" at first use. |
| KR ordering | "the architectural choice of autoregressive order in the network implicitly selects one" — Remark 1 | Stated, not motivated with an example. | Medium — what does it actually feel like to pick the wrong order? §11.2 (open problem) acknowledges this. |

**Knothe–Rosenblatt as an idea.** The textual definition (line 1390:
"each coordinate transformation conditions on previously-transformed
coordinates") is *correct* but compressed. The reader is expected to
realize that this is the natural multivariate analogue of "PIT
followed by `Φ^{-1}`": apply `F_{1}` to get a uniform-`U_1`, then
apply `F_{2|U_1}` to `X_2` conditioned on `U_1`, etc. — i.e., the
*Rosenblatt transform*. The connection to the 1D PIT story already
developed in §1.2 (`Φ(θ − X)` for the location-normal CD) is not
drawn.

**Brenier as load-bearing or not.** Brenier appears once as
contrastive. The contrast is genuinely *not load-bearing* — Part IV
never needs the Brenier map, and an astronomer reader who has never
heard of Brenier will not lose anything by skipping that sentence.
The manuscript could be more explicit about this ("for context only —
this paper uses KR, not Brenier"). Conversely, the contrast does
implicitly tell the careful reader that *there is a unique OT map
without coordinate ordering* (Brenier) — but this paper does not use
it because the architecture has a built-in coordinate ordering.

### B.2 Oracle-style passages

The flagged suspects, in order of severity:

1. **§6 intro one paragraph (1272–1281)** — currently defines
   autoregressive flow + triangular Jacobian + product-of-diagonals +
   linear-in-`d` scaling + cites three references in one paragraph.
   **Verdict: too dense.** No motivating example. A reader who has
   not seen MAF/NSF before will not have a mental picture by the end
   of this paragraph. Recommend splitting into:
   - One paragraph: "Here is what an autoregressive triangular flow
     *is*, with a `d = 2` worked picture."
   - One paragraph: "Here is why the Jacobian is triangular and what
     that buys us."
   - One paragraph: "Standard references."

2. **§6.1 (R1^auto)/(R2^auto) statements (1294–1315)** — the
   *dependence* and *monotonicity* clauses are both stated. The
   load-bearing-clauses paragraph (1309–1315) is *new and excellent*
   (added in round 1): it tells the reader *why* each clause is
   needed before the proof in §6.3 uses them. **Verdict: this is the
   right pattern; preserve it.** Could be slightly improved by adding
   a one-line forward pointer ("see §6.3 induction step for where
   each is used"); could also be improved by stating in plain words
   *what fails* when each clause is dropped — currently it says "the
   Jacobian formula breaks" and "the σ-algebra identity fails," but
   doesn't say what the *user-facing* failure mode is (non-uniqueness
   of the trained network, autograd-Jacobian instability of §8.4).

3. **§6.1 two parameterizations (1326–1350)** — the additive form and
   the doubly-monotone form are *listed* with their formulas, but the
   *story* "you use the additive form when interactions are mostly
   linear in (θ, X) — typical Gaussian-like models; you use the
   doubly-monotone form when the dependence of `r_k` on `θ_k` itself
   varies with `X_k` — typical for non-additive likelihoods like the
   exponential rate" is not told before the formulas. The parenthetical
   `(\S\ref{subsec:8.1}--8.3)` and `(\S\ref{subsec:8.4})` hint at this
   but don't unpack it. **Verdict: pedagogically weak. The reader is
   not told which to pick for their problem.** Add a one-paragraph
   "which form when?" decision rule before the two displays.

4. **§6.2 Jacobian computation (1352–1370)** — the triangular Jacobian
   is *stated* with a piecewise formula. The *why* (because `r_k`
   doesn't depend on `X_>k`, so `∂_{X_ℓ} r_k = 0` for `ℓ > k`) is
   never said in prose. **Verdict: oracle-style. Two-sentence fix.**
   "Because `r_k` is a function of `X_≤k` alone (the *dependence*
   clause of (R2^auto)), `∂_{X_ℓ} r_k = 0` for every `ℓ > k`. The
   Jacobian `∂_X r` therefore has only diagonal and below-diagonal
   entries — it is lower-triangular. The determinant of any
   triangular matrix is the product of its diagonal entries, giving
   `log|det ∂_X r| = Σ_k log|∂_{X_k} r_k|`."

5. **§6.3 Theorem A-d proof (1413–1460)** — the round-1 rewrite
   added σ-algebra equivalence + general-source L-4.2. The structure
   (base case → inductive step with three labelled sub-stages
   "σ-algebra equivalence," "calibration ⇒ conditional Gaussianity,"
   "1D rigidity") is *much* improved over the pre-round-1 version
   and reads more like a story than asserted moves. **Verdict:
   structurally sound; preserve the three-sub-stage scaffolding.**
   Two residual oracle moments:
   - The inductive hypothesis statement is *implicit* ("Assume
     `r_1, …, r_{k-1}` are uniquely identified") — the reader is
     left to recall that "uniquely identified" means "equal to the
     KR rearrangement at their respective level." Add three words.
   - The 1D rigidity sub-stage assumes the reader will look back to
     the §4.2 general-source remark. Add a parenthetical pointer
     `(see §4.2 remark)`.

6. **§6.3 Cholesky corollary (1462–1495)** — the round-1 rewrite uses
   the direct `Z = L^{-1}(X − θ)` route. **Verdict: cleaner than the
   Cholesky-regression-coefficient algebra would have been, but the
   *motivation* for the direct route is not stated** — a reader who
   tries to derive the corollary via the explicit conditional-mean
   route (the way an astronomer who has done Gaussian-process
   conditioning might) will not be told "the direct route bypasses
   the Cholesky-regression-coefficient algebra." Currently the proof
   says only "the cleanest route bypasses the Cholesky-regression
   algebra," which assumes the reader already tried it and gave up.
   Add a sentence stating *what the regression algebra would have
   looked like* (or cite Pourahmadi 1999, per the round-1 reviewer's
   suggestion).

7. **§6.3 three remarks (1497–1543)** — each remark is a tighter
   claim. The *headers* tell the reader what's coming, which is good.
   But the *opening sentence* of each remark does not say *why the
   remark is here*:
   - Remark 1 (ordering): why does this matter? Because users will ask
     "does my choice of variable order in the network's autoregressive
     mask matter?" Answer: yes, but every ordering is correct, and
     §11.2 is an open problem on selecting one. Could open with: "A
     natural first question: does the autoregressive ordering matter?"
   - Remark 2 (architectural non-uniqueness): why does this matter?
     Because §8.3 reports pointwise residuals around the unique target,
     and a careful reader might worry "didn't you just say the
     calibration manifold is a singleton?" Could open with: "Theorem
     A-d's singleton is at the level of the function `r`, not at the
     level of the network parameterization. Here is what that
     distinction means in practice." This was the *exact issue* C-6.3
     in the round-1 inventory was about — the remark answers it but
     does not first ask it.
   - Remark 3 (R2^auto role): why does this matter? Because the
     manuscript wants to identify *which clause* of (R2^auto) supplies
     which step of the induction, to clarify that both clauses are
     genuinely needed. This *does* open with a clear motivating
     sentence ("Both clauses... are needed in the induction, but for
     distinct reasons"). Preserve as is.

### B.3 Proof-role clarity for Theorem A-d

Currently the reader meets Theorem A-d at line 1399 with no
preamble: "**Theorem A-d (Multivariate uniqueness within the
autoregressive triangular class).**" The *role* of the theorem —
"without this, the multivariate calibration manifold is infinite-
dimensional plus rotation-degenerate; with this, it is a singleton
and equals the Knothe–Rosenblatt rearrangement" — is not stated as
a setup sentence. §3.3 says it (briefly), but only `d − 1` sections
back. Recommend a one-paragraph "what this theorem accomplishes
and why we need it" *immediately before* the theorem statement,
echoing the structure of §4 (Theorem A) where the role is set up by
§3.3 immediately before. A model:

> "From §3.3, the multivariate calibration manifold is enormous:
> infinite-dimensional in general plus closed under rotations of
> `N(0, I_d)`. We now show that the autoregressive triangular class
> closes it to a singleton — and identifies that singleton as the
> Knothe–Rosenblatt rearrangement."

### B.4 Concept-introduction gaps

**Missing: a worked `d = 2` example of an autoregressive triangular
flow.** §6.1 gives two formulas; §6.2 gives a Jacobian schematic;
§6.3 gives a uniqueness theorem; §8.3 reports a *trained* `d = 2`
example. But the reader is never shown a *concrete* `d = 2` instance
of the additive form (or the doubly-monotone form) with the
networks replaced by simple functions — e.g., "for illustration,
suppose `a_1(θ_1) = θ_1`, `b_1(X_1) = X_1`, `a_2(θ_2; θ_1, X_1) = θ_2
+ 0.5 θ_1`, `b_2(X_2; θ_1, X_1) = X_2 + 0.5 X_1`, then `r_1 = θ_1 −
X_1` and `r_2 = (θ_2 − X_2) + 0.5 (θ_1 − X_1)` — note how the
conditioning on `X_<k` lets the second component pick up correlation
structure." This would give the reader something concrete to point
the formal definitions at.

**Does the Cholesky corollary land as an example?** Partly. It
*is* a worked closed-form, but the form `L^{-1}(θ − X)` is written
as a matrix-vector product rather than coordinate-by-coordinate.
The 2D explicit form `r_2 = 1.155(θ_2 − X_2) − 0.577(θ_1 − X_1)` is
in the proof body (line 1494) and also in §8.3 (line 1849) — but
only in the proof, after the corollary is stated abstractly. Reorder
the corollary so the 2D explicit form appears *at first statement*,
with the general `L^{-1}` form as the abstraction. Alternatively:
expand the corollary into a half-page "worked example: 2D
correlated Gaussian," giving the reader something to anchor the
abstract theorem to.

**Missing: a connection back to the §1.2 PIT story.** §1.2 establishes
that the 1D CD `H^*(θ; X) = Φ(θ − X)` is the PIT of the pivot
`θ − X`. The multivariate analogue is exactly the Rosenblatt
transform: `Φ_d(r^{KR}(θ; X))` is the product PIT, taking each
conditional CDF in turn. Drawing this bridge — "what we called the
PIT in §1.2 generalizes to the Rosenblatt transform in `d > 1`" —
would land KR for a reader who already understands the 1D story.
This is a *one-paragraph* fix, and it is the single best pedagogical
upgrade Part IV could make.

### B.5 Narrative flow

§6 has four subsections: class → Jacobian → uniqueness → inference.
The build is logical, but the *bridges* between subsections are thin:

- §6 intro → §6.1: clean (intro motivates, §6.1 specifies).
- §6.1 → §6.2: a one-sentence bridge is missing. Currently §6.2
  begins "In either parameterization, both Jacobians are lower-
  triangular by construction" — that's a fine bridge, but the
  reader is left to infer "we now compute the Jacobian we needed
  for the NF-MLE loss." Add a sentence.
- §6.2 → §6.3: jarring. §6.2 is computational; §6.3 opens with
  "Optimal-transport context, briefly" which jumps to a different
  register entirely. The OT context paragraph itself is fine, but
  there is no bridge sentence saying "we have the architecture; now
  we ask whether it pins down `r` uniquely." Add one.
- §6.3 → §6.4: clean (theorem → inference).

**Does §6.3 overwhelm?** It is the longest subsection by a wide
margin (~160 lines vs. ~30–60 for the others), containing the
theorem + proof + corollary + 3 remarks. **Verdict: yes, mildly
overwhelming, and the proof is the worst offender.** Options:
- Split §6.3 into §6.3 (Theorem A-d statement + proof) and §6.4
  (KR consequences: Cholesky corollary + remarks), with §6.5 →
  inference. This gives the reader a breath between the proof and
  the remarks.
- Or: keep the structure but visually separate the corollary +
  remarks under a sub-header like "**Consequences of Theorem A-d**"
  to signal the gear-shift.

The corollary and the three remarks are not load-bearing for the
proof — they are downstream payoffs. Splitting them off would let
a reader who only wants the theorem skip ahead without losing the
inference section, and let a reader who wants the Cholesky form
(astronomers) navigate to it directly.

---

## Prioritized recommendations

In rough priority order (highest pedagogical payoff first):

1. **Add a worked `d = 2` autoregressive-triangular-flow example**
   *between* §6.1's two parameterizations and §6.2's Jacobian
   computation. Use the additive form with simple polynomial `a_k`,
   `b_k` (no neural networks) so the reader sees concretely what the
   triangular structure does. This is the single biggest accessibility
   gain available in Part IV. (One half-page.)

2. **Draw the PIT → Rosenblatt-transform bridge** explicitly. Add
   one paragraph either in §6 intro or at the top of §6.3 saying:
   "The 1D PIT story of §1.2 (`Φ(θ − X)` is the PIT of the location-
   normal pivot) generalizes to `d > 1` as the *Rosenblatt
   transform*: apply the first conditional CDF, then the second
   conditioned on the first, and so on. The Knothe–Rosenblatt
   rearrangement is this transform composed with `Φ^{-1}` at each
   step." (One paragraph.)

3. **Add a one-paragraph "what Theorem A-d accomplishes" before its
   formal statement.** Echo the §3.3 framing: without architectural
   structure the manifold is infinite-dimensional + rotation-
   degenerate; Theorem A-d says the autoregressive triangular class
   closes it to a singleton. (One paragraph.)

4. **Rewrite §6 intro paragraph as two paragraphs** (definition with
   `d = 2` mental picture; then properties + references). The current
   one-paragraph version packs too much in. (One paragraph → two.)

5. **Add a "which parameterization when?" decision rule** before the
   two displays in §6.1. Currently the parenthetical pointers
   `(\S8.1–8.3)` and `(\S8.4)` hint at this; make it explicit.
   "Use the additive form when the dependence of `r_k` on `(θ_k,
   X_k)` is approximately a difference of two monotone functions —
   typical for Gaussian-like models. Use the doubly-monotone form
   when `r_k`'s dependence on `θ_k` itself varies with `X_k` —
   typical for non-additive likelihoods like the exponential rate.
   The KR theorem applies to both." (One paragraph.)

6. **Add 2-sentence "why triangular ⇒ product-of-diagonals
   determinant" prose** at the top of §6.2. The piecewise formula is
   not a derivation; it is a definition followed by an assertion.
   (Two sentences.)

7. **Open Remark 2 with the question it answers.** "If the
   calibration manifold is a singleton, why do we see pointwise
   residuals in §8.3?" Then the existing content. (One sentence.)

8. **Split §6.3 into §6.3 (theorem + proof) and §6.4 (KR
   consequences: Cholesky corollary + remarks)**, with §6.5 →
   inference. Or, less invasively, add a sub-header "**Consequences
   of Theorem A-d**" before the corollary. (Structural; cosmetic if
   sub-header.)

9. **Cholesky corollary: lead with the 2D explicit form**, then the
   abstract `L^{-1}` form. Currently the 2D explicit form is buried
   in the proof body. (Reorder.)

10. **Replace "σ-algebra equivalence" with a working-statisticians'
    paraphrase** in the proof, or add a parenthetical gloss: "the
    information about `θ` carried by the previously-transformed
    coordinates `r_<k` is the same as the information carried by the
    original observations `X_<k`." Keep the technical term for the
    statisticians; add the paraphrase for the astronomers. (One
    parenthetical.)

11. **Demote "Brenier" to a footnote or parenthetical**, since the
    contrast is purely cultural and not load-bearing. The main text
    can say "we use the Knothe–Rosenblatt rearrangement (the unique
    triangular monotone transport map; we use this rather than the
    L²-optimal Brenier map because triangular structure aligns with
    autoregressive masking)." (One sentence; current treatment is
    longer.)

12. **Forward-define UMNN** with a half-sentence at first use in §6.1:
    "UMNNs (defined formally in §7.1) parameterize a scalar
    monotone function as the integral of a positive integrand, giving
    closed-form derivatives." Currently the reader meets "UMNN" in
    §6.1 with no inline definition and a forward pointer to §7.1
    that they have to chase. (Half-sentence.)

---

## Existing strengths to preserve

- **The "load-bearing clauses" paragraph in §6.1 (1309–1315)** is
  *excellent* round-1 work. It tells the reader why each clause of
  (R2^auto) is needed *before* the proof uses it. Preserve and use
  as the model for upgrading (R1^auto)'s motivation too.

- **The three labelled sub-stages of the Theorem A-d inductive
  step** ("σ-algebra equivalence," "Calibration ⇒ conditional
  Gaussianity," "1D rigidity") read as a story, not a wall of moves.
  Preserve the labelling scheme.

- **The Cholesky corollary's direct route** (`Z = L^{-1}(X − θ) ~
  N(0, I_d)`, so `−Z = L^{-1}(θ − X) ~ N(0, I_d)`, then verify
  (R1^auto)/(R2^auto), then invoke Theorem A-d uniqueness) is much
  cleaner than the Cholesky-regression-coefficient algebra would
  have been. Preserve the route; just signpost it.

- **The OT-context opening of §6.3** is a clean register-shift
  that gives the reader cultural orientation before the theorem.
  Preserve, but trim Brenier to a parenthetical.

- **The "linear scaling in `d`" callout in §6.2** (`O(d)` vs `O(d³)`)
  is a clean, concrete payoff that astronomers will appreciate.
  Preserve.

- **The §6.4 worked 1D example** (`C_α = [X − z_α, X + z_α]`) is a
  sanity-check that lands well — the reader sees the multivariate
  machinery degenerate gracefully to the textbook `z`-interval.
  Preserve.

- **Remark 3's two-clause failure analysis** (what fails without the
  *dependence* clause vs. without the *monotonicity* clause of
  (R2^auto)) is precisely the kind of "*which* hypothesis is doing
  *what* work" exposition this manuscript needs more of. Preserve
  and use as a model for similar two-clause analyses elsewhere.
