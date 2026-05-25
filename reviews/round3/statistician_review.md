# Round 3 Final Review — Statistician Reader

## Summary

The manuscript is technically sound after round 3. The core load-bearing
arguments — strict propriety of NF-MLE (Thm 3.2), the three-lemma proof of
Theorem A, the lift to exponential families (Thm C), the Bin(3,θ)
counterexample driving (R4), the Knothe–Rosenblatt induction of Thm A-d,
and the Cholesky corollary — are correct, with hypotheses stated cleanly
and proofs that match the conclusions claimed. The §3.5 four-step
re-framing of the loss-below-truth mechanism preserves the underlying
Gibbs / KL-with-normalizer-shift argument intact; the §8.4 ablation
narrative is rigorous and directly grounded in the §3.5 algebra. The
round-3 pedagogical additions (the "What is X" preambles in §1.2, §3.1,
§5.7, §6.3, §7.1; the "What we are about to prove" theorem preambles;
the §6.3 example-first treatment of the Cholesky corollary) are
overwhelmingly framing-only — they preview the formal content without
substituting for it. There are, however, a small number of places where
either the preamble overstates the theorem's reach, where a definition
is loose enough to admit nitpicking, or where a "by inspection" move
elides a step that would benefit from one extra sentence. None are
blockers; they are tightenings, and the manuscript can be submitted with
or without them addressed. The most consequential residual issue is in
§6.4 (lift from ρ-a.e. calibration to every θ₀), where the manuscript
asserts the pointwise lift but does not actually prove it — this was
true in v6 as well, but the new framing makes the gap more visible.

## Technical rigor: where it holds up

- **Theorem 3.2 (strict propriety, §3.2).** The conditional KL
  decomposition `L(r) = E_ρ[KL(p ‖ p̂_r)] + E_ρ[H(X|θ)]` is correctly
  stated and depends only on (R2). The conclusion that minimizers are
  exactly `M_F` (and that the minimum equals the ρ-averaged conditional
  entropy) is precisely the right statement of strict propriety in this
  setting, and the hypothesis "every element of F satisfies (R2)" is
  the right one. The promotion from in-paragraph claim to labelled
  theorem improves the manuscript's auditability without changing the
  content.
- **Three-lemma proof of Theorem A (§4.2–§4.3).** Lemma 4.1 (1D monotone
  rearrangement, citing Villani 2003 ch.2), Lemma 4.2 (level-set
  rigidity via pushforward-density blow-up at a critical point, with
  the `|v − v_*|^{−1/2}` rate worked out explicitly), and Lemma 4.3
  (NF-MLE strict propriety conditional) are all correctly stated. The
  Lemma 4.2 remark on "general source" (any continuous-positive source,
  any continuous-finite-density target) is used cleanly in Thm C and Thm
  A-d; the manuscript flags this dependency explicitly, which is good
  practice.
- **Theorem A* (§4.4) and (R3).** The reason a separate Lipschitz
  theorem is needed (a Lipschitz V-shape gives a bounded jump, not a
  blow-up) is articulated correctly. The proof correctly invokes
  Rademacher + bi-Lipschitz + Lemma 4.1 in place of Lemma 4.2. The
  observation that (R3) is automatic for C¹ networks but must be
  imposed for vanilla ReLU is technically accurate.
- **§5.7 Bin(3,θ) counterexample.** The four-tile reordering, the
  algebraic identity `∂_θ A_2 = −3(2θ − 1)²` (the "perfect-square
  accident"), and the explicit demonstration that (R1)+(R2)+(R3_U)
  hold while (R4) fails are all verifiable by direct calculation. The
  ruling-out of Bin(1,θ) and Bin(2,θ) as too small is correct and
  matches the genuine combinatorial picture.
- **Theorem A-d induction (§6.3).** The base case is Theorem A applied
  to `(θ₁, X₁)`. The σ-algebra equivalence `σ(θ, X_{<k}) = σ(θ, r_{<k})`
  is correctly derived from (R2^auto) applied autoregressively. The
  step "calibration ⇒ conditional Gaussianity" — that joint
  `r | θ ∼ N(0, I_d)` plus the σ-algebra identity gives
  `r_k | θ, X_{<k} ∼ N(0, 1)` — is the right move (the conditional law
  of a coordinate of a standard normal given the other coordinates is
  N(0,1) by independence, and the σ-algebra identity makes the
  conditioning equivalent). The application of the general-source Lemma
  4.2 to `F_k^{(θ)}(· | X_{<k})` is justified by the hypothesis that the
  conditional CDF is `C¹` and strictly increasing.
- **§3.5 four-step loss-below-truth decomposition.** The decomposition
  `L(r) = E_ρ[KL(p ‖ p̂_r/Z)] + E_ρ[H(X|θ)] − E_ρ[log Z(θ)]`
  is correct. The KL identity applied to the normalized density
  `p̂_r/Z(θ)` is the standard move, and the `−E_ρ[log Z]` shift is
  exactly what folding produces. The numerical claim that
  `0.88 − 0.56 = 0.32 ⇒ Z ≈ e^{0.32} ≈ 1.38` follows from this
  decomposition.
- **Cholesky corollary proof (§6.3).** The "cleanest route" given —
  verify `L⁻¹(θ − X)` is in the autoregressive calibration manifold by
  direct construction, then invoke Thm A-d's uniqueness — is correct
  and avoids the more involved Cholesky-regression algebra.
- **Class 4 (CRPS directional propriety, §3.7).** The collapse-to-median
  conclusion is correctly derived via the kernel-score form
  `CRPS(F, y) = E_{Z∼F}|Z − y| − ½ E|Z − Z'|`: the second term is
  independent of `G`, the first is the `L¹` distance between `G` and
  `F`, and minimizing the `L¹` distance from a fixed `F` to a point
  mass over the location of the point mass yields `δ_{m(F)}` where
  `m(F)` is the median of `F`. The elementary expansion is consistent.

## Technical rigor: places where round-3 edits introduced imprecision

- **§1.2, definition of CD.** The opening sentence says "a confidence
  distribution (CD) is the frequentist analogue of the Bayesian
  posterior: a data-dependent distribution function `H(·; X)` on Θ
  whose α-quantile is the upper endpoint of an exact one-sided
  α-confidence interval." This conflates the CD with its quantile
  function and is mildly inaccurate even on its own terms: the
  α-quantile of `H(·; X)` is the upper endpoint of an
  α-confidence interval *under the convention that `H` is increasing in
  θ and α refers to the upper one-sided level*. The Schweder–Hjort
  definition immediately afterward (their Def 3.1) does the correct
  work; the opening should either be dropped or rephrased to say
  "encodes confidence intervals at every level via its quantiles."
- **§1.2, "Astronomer-friendly comparison."** The bullet on the
  Bayesian posterior says "credible intervals contain the true θ₀ with
  probability α *averaged over the prior* — coverage holds when one
  integrates over the same prior used to define the posterior. There is
  generally no guarantee of frequentist coverage *pointwise at a given
  θ₀*." This is correct but slightly imprecise: in fact the standard
  statement is even stronger (Bayesian credible intervals have average
  coverage over the prior *only on average over data drawn from the
  prior-marginal model*, not in a frequentist sense at all). A working
  statistician will not be misled, but the phrasing makes the result
  look more frequentist-shaped than it is.
- **§3.1, "What is a normalizing flow."** The pedagogical preamble
  describes an *unconditional* flow `r: X → R^d`. The transition "From
  flow to conditional flow to NF-MLE" then re-introduces the
  conditional version with `r(θ, X)`. A reader who skipped or read
  quickly might walk away with the impression that NF-MLE is on
  `X | θ` *unconditionally on θ in the loss*, when in fact the
  expectation is over `(θ, X) ∼ ρ ⊗ P_θ`. This is not a wrong
  statement, but the linkage to the loss could be a half-sentence
  clearer.
- **§3.2 Theorem 3.2, hypothesis.** "Let F be a class of measurable
  functions ... every element of which satisfies the invertibility
  condition (R2)" — the (R2) of §2.2 is stated as a `C¹` diffeomorphism
  condition, with the Lipschitz variant (R3) deferred to §4.4. The
  proof uses only that `p̂_r` is a normalized density; the Lipschitz
  case satisfies that under (R2 Lipschitz) + (R3), but as written the
  theorem's hypothesis nominally restricts to `C¹` pivots. The
  parenthetical "(or its autoregressive analogue (R2^auto)); the two
  are equivalent for our purposes" is helpful but does not cover the
  Lipschitz extension. A small "(or the Lipschitz pair (R2),(R3) of
  §4.4)" addition would tighten this.
- **§3.5 Step 2.** "The regions of `X`-space where `r` folds are
  precisely where `|det ∂r/∂X|` is large (the local volume rescaling
  goes up to compensate for the under-counting)" — the word "precisely"
  is too strong: large Jacobian magnitude is *necessary* for non-
  bijectivity at fixed `θ` (a small Jacobian implies a small
  neighborhood, hence local injectivity by inverse-function arguments
  in the smooth case), but the implication does not run "fold ⇒
  Jacobian is large" in a tight quantitative sense. "Folding makes the
  surrogate's local Jacobian factor inflate in the folded region" is
  the safer statement.
- **§3.7 Class 1 counterexample.** The construction
  `r(θ, X) = sign(θ) · |X − θ|` on `Θ = {±1}` with uniform `ρ` is
  presented as failing conditional calibration while passing marginal
  PIT. The conditional law of `r` given `θ = +1` is `|X − 1| = |Z|`
  (Z standard normal), i.e. half-normal on `[0, ∞)`. Given `θ = −1`,
  `r = −|X + 1| = −|Z|`, supported on `(−∞, 0]`. The mixture is a
  *folded normal mixture* with density `|Z|`-like spread on `(−∞, 0)`
  and `(0, ∞)`. Whether this mixture is exactly `N(0, 1)` is what the
  manuscript asserts "by symmetry." It is in fact exactly `N(0,1)`:
  given `θ = +1`, `r ∼ |Z|`; given `θ = −1`, `r ∼ −|Z|`; the
  ½–½ mixture has the same density as `|Z|` on `[0,∞)` and `|Z|` on
  `(−∞, 0]`, which is `φ(z)` everywhere. The argument is correct, but
  the "by symmetry" leaves a reader two seconds of work; one extra
  sentence ("the mixture density at `z > 0` is `½ · 2φ(z) = φ(z)`,
  and similarly for `z < 0`") would make the point indisputable on
  first read.
- **§6.3 Theorem A-d, "calibration ⇒ conditional Gaussianity" step.**
  The argument "the full calibration constraint
  `r(θ; X) | θ ∼ N(0, I_d)` implies that all components are jointly
  standard normal. Combined with the σ-algebra identity,
  `r_k | θ, X_{<k} ∼ N(0, 1)`." is correct but quietly uses two facts:
  (i) the components of a standard normal vector are independent, so
  the conditional law `r_k | r_{<k}` is `N(0, 1)`; (ii) by the
  σ-algebra identity, conditioning on `r_{<k}` equals conditioning on
  `X_{<k}` (both given θ). Spelling out the independence step (one
  half-line) would make this air-tight; as currently written, a
  hostile referee might object that "jointly standard normal" was
  conflated with "marginally + conditionally standard normal" without
  the independence step being named.
- **§6.4 lift from ρ-a.e. to every θ₀.** The parenthetical "the NF-MLE
  training argument delivers calibration only ρ-a.e.; the lift to every
  θ₀ uses continuity of `r` in θ under the architectural class plus
  ρ's full support on the inferential region" is the right *idea*, but
  it is asserted without proof. The claim that pointwise frequentist
  coverage holds at every θ₀ — which is the central selling point of
  CD-SBI vs. Bayesian alternatives — rests on this lift, so handwaving
  it costs the manuscript credibility relative to the strong claims
  elsewhere. A short lemma ("if `r` is continuous in θ on supp(ρ) =
  closure of inferential region, and `r(θ; X) | θ ∼ N(0, I_d)` for
  ρ-a.e. θ, then by continuity of `H_r` and of the law of
  `r(θ; X) | θ` in θ, calibration extends to every θ in supp(ρ)")
  would close this. Note this is a known v6 issue, not new to round 3;
  but the round-3 emphasis on the "pointwise, every θ₀" promise makes
  the gap more conspicuous than it was.

## Theorem statements: any that mislead or misframe?

- **Theorem A preamble (§4.1):** "the framework, properly architected,
  picks out the classical answer rather than something exotic" — fair
  framing. The strategy preview is accurate.
- **Theorem 3.2 preamble (§3.2):** "the population-level minimizers of
  `L` are precisely the calibrated pivots in `F`, nothing else" —
  accurate.
- **Theorem A-d preamble (§6.3):** "the autoregressive triangular
  restriction (R1^auto) + (R2^auto) is enough to collapse the manifold
  back to a singleton" — accurate. The strategy ("induction on the
  coordinate index — base case is Theorem A applied at fixed θ₁, and
  the inductive step identifies `r_k` by applying the general-source
  form of Lemma 4.2") matches the proof.
- **Theorem C preamble (§5.2):** "the proof works because the
  manuscript's three-lemma argument never used anything specific about
  Gaussian shape beyond two facts: source has continuous positive
  density, target has finite density" — accurate.
- **Theorem C\* preamble (§5.7.2):** the framing that (R4) is needed in
  the discrete case because permutations of T-atoms can preserve
  calibration + within-atom monotonicity is correct; the Bin(3) example
  then carries the load.
- **Proposition 4.5 (§4.5):** "the unique element of `M ∩ F_{C¹}`
  produced by CD-SBI is identically the Schweder–Hjort UMP-unbiased
  CD" — accurate; this is what Theorem A and the Schweder–Hjort
  textbook entry together imply.

No theorem statement misleads or overstates. The preambles are
faithful previews.

## Proofs: any that have gaps not previously flagged?

- **§6.3 inductive step (joint normality ⇒ conditional N(0,1)) — half-
  line gap.** As noted above, the independence-of-coordinates step is
  implicit. Easy fix; not a substantive gap.
- **§6.4 ρ-a.e. → every-θ₀ lift — asserted, not proved.** This is the
  load-bearing pointwise coverage claim; the manuscript should either
  prove it with a short lemma or scope the coverage claim to "ρ-a.e.
  θ₀ + continuous extension to supp(ρ) by the architectural class."
- **§3.5 Step 2 "precisely" phrasing.** Not a proof gap, just an
  overstrong direction in the prose. The actual argument (folding
  inflates the Jacobian factor → drives `Z(θ) > 1` → loss shifts by
  `−log Z`) is correct.
- **§6.3 corollary (Cholesky).** The proof bypasses the
  Cholesky-regression algebra by direct verification that `L⁻¹(θ − X)`
  is in the autoregressive calibration manifold. This is fine, but the
  Worked Example preceding it ("the KR pivot in this case is just
  `L⁻¹(θ − X)`") asserts the conclusion before the proof — which is
  the intended "example first" pattern. The example does not in itself
  prove the formula; it illustrates it. A reader could misread the
  Worked Example as the proof if not careful. The formal proof
  (1 paragraph) is present, so this is a matter of signposting rather
  than rigor.
- **§3.7 Class 4 elementary expansion.** The expansion
  `E_{Y∼G}[CRPS(F, Y)] = ∫ F² dz + ∫ G(z)(1 − 2F(z)) dz` is
  individually divergent (both integrals diverge at `±∞`); the
  manuscript flags this with "after suitable regularization." A
  skeptical reader will accept the kernel-score derivation as
  authoritative and treat the elementary expansion as illustrative;
  the manuscript correctly relies on the kernel-score form for the
  primary argument.

No new (non-v6) gaps. The two pre-existing soft spots (the §6.4 lift
and the implicit independence step in §6.3) are visible to a careful
reader.

## What landed well (the pedagogical edits that didn't compromise rigor)

- **§1.2 "What is a PIT?"** Correct, concise, and the algebraic check
  `P(G(Y) ≤ u) = u` is the right level of detail. The transition to
  "the PIT is what makes calibration condition (ii) non-circular" is
  helpful framing.
- **§1.2 location-normal worked example.** Anchors the abstract goal
  in the simplest case before any architecture or training is
  introduced. The verification `Φ(θ₀ − X) = Φ(−Z) ∼ U(0,1)` is
  air-tight.
- **§3.1 "What is a normalizing flow?"** Conveys the change-of-
  variables idea cleanly. The two-factor decomposition (base-density
  factor + Jacobian factor) is correct framing.
- **§5.7.1 "What is the intuition behind randomized PIT."** The
  sub-interval-tiling description is the right mental model for
  Lancaster's construction and prepares the reader for (R4) by
  hinting at the within-vs-between-atom distinction.
- **§6.3 KR vs Brenier sidebar.** Correct, brief, and helpful: KR
  depends on ordering, Brenier doesn't; KR aligns with autoregressive
  masking. Both citations (Knothe 1957; Rosenblatt 1952; Brenier 1991)
  are accurate.
- **§7.1 "The idea, in one sentence" for UMNN.** Captures the FTC-via-
  positive-integrand idea cleanly; the implementation substitutions
  (softplus + Gauss–Legendre vs. ELU+1 + Clenshaw–Curtis) are flagged
  correctly.
- **§8.4 four-step ablation narrative.** Preserves the §3.5 algebra
  (Steps 3 and 4 directly invoke the `Z(θ) > 1 ⇒ −log Z` shift) while
  giving the reader a clear empirical hook (0.88 vs 0.56 vs the
  conditional-entropy floor). The "smoking gun" framing is justified
  by the algebra.
- **Theorem preambles in general.** They preview what the theorem
  proves and which lemmas / strategies do the work, without claiming
  more than the formal statement delivers. This is exactly what a
  pedagogical preamble should do.

## Specific recommendations (priority-ranked)

1. **Close the ρ-a.e. → every-θ₀ lift in §6.4.** Either with a short
   lemma (one paragraph: continuity of `r` in θ + ρ full support ⇒
   law of `r(θ; X) | θ` is continuous in θ in total variation ⇒
   calibration set is closed ⇒ ρ-a.e. + closed = full support) or by
   scoping the coverage claim to "exact frequentist coverage at every
   θ in the closure of supp(ρ) where `r` is continuous." Highest
   priority because this is the central inferential promise of the
   framework. *(Pre-existing v6 issue, not new to round 3.)*

2. **Spell out the independence step in §6.3 Theorem A-d inductive
   step.** Half a line ("since the components of `N(0, I_d)` are
   independent, the conditional law of `r_k | r_{<k}` is again
   `N(0, 1)`") makes the σ-algebra-then-conditional-normality
   argument fully explicit.

3. **Reword §3.5 Step 2's "precisely where" to "in regions where" or
   "where the Jacobian factor is inflated."** Avoids the implicit
   converse direction. Minor but worth the keystrokes.

4. **In Theorem 3.2's hypothesis statement (§3.2), add the Lipschitz
   variant explicitly.** Either change "(R2) of §2.2 (or its
   autoregressive analogue (R2^auto))" to "(R2) of §2.2 (or the
   Lipschitz pair (R2)+(R3) of §4.4, or its autoregressive analogue
   (R2^auto))", or be explicit that the strict-propriety argument
   needs only "p̂_r is a valid normalized density for ρ-a.e. θ" as a
   hypothesis, which all three variants supply.

5. **Tighten the §1.2 opening definition of CD** so that the relation
   between `H(·; X)` and confidence intervals is stated as "encodes
   one-sided and two-sided intervals via its quantiles" rather than
   pinning to the α-quantile and the upper endpoint specifically. The
   formal Schweder–Hjort definition immediately afterward does the
   work; the pedagogical lede should not over-promise on which side.

6. **Add the "by direct calculation" line for the §3.7 Class 1 mixture.**
   `½ · 2φ(z) = φ(z)` on each half-line, by symmetry of `Z`. One
   half-sentence prevents an unnecessary "is this right?" beat in an
   otherwise crisp counterexample.

7. **Consider whether the §6.3 "Worked example" should be labeled more
   clearly as illustration-not-proof.** The current "The KR pivot in
   this case is just `L⁻¹(θ − X)`" before the corollary proof is fine
   for a mixed audience but could leave a technical reader briefly
   uncertain about which is the formal claim. A "We verify this in
   the corollary below" pointer would resolve.

8. **§3.7 Class 4 elementary expansion: either include a one-line
   note on the regularization, or drop the second derivation.** The
   kernel-score derivation is sufficient on its own; the elementary
   expansion's "after suitable regularization of the (individually
   divergent) integrals" is correct but reads as hand-waving when set
   next to the rigorous kernel-score argument. Either is fine; both
   together makes the second look weaker than it is.

None of these are blockers. Items 1 and 2 are the substantive ones; 3–8
are polish. The manuscript is in submittable shape and the round-3
pedagogical pass did not damage technical rigor in any load-bearing
place.
