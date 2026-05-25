# Round 3 — Global Structure & Narrative Review

Scope: whole-manuscript reading of `cd_sbi_v7.tex` (2,427 lines, 7 Parts).
Focus: cross-Part coherence, narrative arc, pedagogical scaffolding, and
the two failure modes the user flagged (astronomer accessibility +
oracle-style writing). Per-Part findings are out of scope here.

## Through-line and connections

The intended argument arc is clear and, on the whole, the manuscript
delivers it: §1.1 names the problem (overconfident SBI), §1.2 introduces
the alternative target (a CD), §1.3 states the top-line goal in
operational language (`r(θ; X)` with `H_r = Φ_d ∘ r` pointwise
calibrated), §2 defines the calibration manifold `M`, §3 supplies a loss
that selects `M`, §4–6 prove uniqueness within architectural classes,
§7–8 demonstrate finite-sample reachability, §9 turns the theory into a
recipe, §10 places it in the SBI taxonomy, §11 enumerates what's left.
That arc is exactly the right shape for the work.

Where the through-line succeeds:
- **§1.3's longtable mapping objectives to sections** is the manuscript's
  best piece of global scaffolding. It tells the reader, upfront, what to
  expect and where each piece lives. Nothing else in the document plays
  this role at scale.
- **§8.5's "Synthesis" table** mapping experiments back to theoretical
  claims closes the experimental loop cleanly.
- **The "Status of claims" table at the end** is the third pillar that
  holds the structure together — it lets a reader audit the whole
  argument in one page.
- The reduction-to-classical statements (Proposition 4.5; the `L⁻¹` KR
  corollary verified in §8.3) work as "anchors" — moments where the
  abstract framework becomes a recognizable classical object.

Where the through-line frays:

1. **Part II (§3) is a heavier load than its position suggests.** §3.1
   defines normalizing flows in three sentences; by §3.2 the reader must
   already be reasoning about KL identities, change-of-variables, and the
   distinction between unconditional and conditional density matching.
   Then §3.7 (the five-class taxonomy of strictly proper objectives)
   detours into a substantial design-space comparison that, while
   excellent in content, is *positioned* as if the reader has already
   chosen NF-MLE and just wants to see the alternatives. A reader who
   came in via §1 has not made that choice.

2. **§4 → §5 → §6 is monotonically increasing in classical-statistics
   prerequisites.** §4 needs only `N(θ, 1)` and monotone transport. §5
   needs one-parameter exponential families, sufficient statistics, MLR,
   PIT, and the discrete extension needs Lancaster randomization. §6
   needs Knothe–Rosenblatt, Brenier, the Cholesky bridge, and an
   inductive σ-algebra argument. The escalation is not flagged for the
   reader; an astronomer who survived §4 may not realize §5 is
   intentionally a different reading level.

3. **The §3.4 "SNL is the same loss" identification gets buried.** This
   is potentially the single biggest connection the manuscript draws to
   the SBI literature — *we use a well-understood loss, the contribution
   is the architectural prescription*. It deserves more prominence (or
   to be re-asserted as a refrain in §10), not a subsection inside the
   loss chapter.

4. **§9 (the implementation recipe) is too short for what it carries.**
   It is the bridge between the proof-heavy Parts III–IV and the open
   problems. As written it's a 6-bullet checklist; the reader does not
   get a worked illustration of "I have a new model, here is how I
   proceed end-to-end."

5. **§11 reads as a list of open problems, but the manuscript never
   states whether *any* of them is required for the paper's claims to be
   trustworthy.** §11.4 (misspecification) and §11.5 (higher-d) are
   particularly load-bearing for the paper's practical pitch. The reader
   has no way to tell from §11 alone which open problems are "future
   work, nice to have" vs. "we promise calibration on 2D Gaussians and
   one 1D non-Gaussian; everything else is conjecture."

## Audience signaling

The §1.2 "astronomer-friendly comparison" subsection is the only place
in the manuscript where the astronomer reader is explicitly named.
After §1.2 the framing disappears entirely. The astronomer-flavored
bridges (e.g. relating CD to "a distribution-valued summary of the
evidence") do not recur. By §3.2 the prose has become standard
mathematical-statistics writing with no further accommodation.

This is the user's first flagged failure mode in concrete form: the
astronomer is welcomed in §1.2 and then abandoned. The §3.7 brief
glossary ("Cramér–von Mises distance is...", "HSIC is...") is an
exception — it briefly resurrects the astronomer-aware voice — but
it's tucked inside the design-space taxonomy where an astronomer might
already have bailed.

Specific moments where the astronomer reader is left without a hand-rail:
- §3.2's strict-propriety argument assumes fluency with the KL identity
  `E[-log q] = H + KL(p||q)`.
- §4.2 Lemma 4.2's blow-up proof assumes the reader will follow
  `|v - v_*|^{-1/2}` divergence as a standard maneuver.
- §5.1 introduces "MLR" and "regular exponential family" in a single
  paragraph and asks the reader to retain both for the rest of the Part.
- §5.7 introduces randomized PIT, mid-p, and the Lancaster reference
  without saying *why* discrete models are even being discussed here
  (most astronomer applications are continuous).
- §6.3's KR/Brenier/Cholesky paragraph is dense even by statistical-ML
  standards.
- §11.4 mentions "forward-KL projection" and "M-projection in the
  modern Bregman-geometry sense" with no foothold for an astronomer who
  doesn't already know Bregman divergences.

## Section/Part structure

Seven Parts is on the high side for ~35 pages, and the boundaries are
not all carrying equal load:

- **Part I (Framework, §1–2)**: well-scoped. §1 motivates, §2 makes the
  pivot definition formal. Good.
- **Part II (Loss design, §3)**: this is a single section in its own
  Part. The §3.7 taxonomy reads as if it wants to be a sibling Part —
  "Part III: The design space of objectives" — but is currently lumped
  in with the loss derivation.
- **Part III (1D theory, §4–5)**: also well-scoped, but §5.7 (discrete)
  is a sub-sub-section under §5 and yet structurally is the most
  involved single result in Part III (uses (R4), the Bin(3,θ)
  counterexample, etc.). It could justify its own §6.
- **Part IV (Multivariate, §6)**: one section. Same pattern as Part II.
- **Part V (Experiments, §7–8)**: balanced. §7 is architecture, §8 is
  four experiments + synthesis. This works.
- **Part VI (Practice + position, §9–10)**: the *pairing* feels arbitrary.
  §9 is forward-facing (here's how to use it); §10 is comparison-facing
  (here's where it sits). They share a Part by elimination, not by
  natural cohesion.
- **Part VII (Open questions, §11)**: a Part containing one section.

Recommendations to consider:
1. **Promote §3.7 ("design space of strictly proper objectives") to its
   own Part**, before or after Part III, as a substantive
   "alternatives-considered" chapter. As-is, this 100-line subsection
   does heavy positioning work that §10 partially repeats.
2. **Move §10 (Position in the SBI literature) much earlier.** Most of
   §10's content is comparison-table — it could live in §1 as a
   "context" subsection after §1.1, framing the entire document as the
   addition of one row (CD-SBI) to a known taxonomy. An astronomer
   reader trying to triage *whether to read further* would be served by
   this earlier.
3. **Merge §11 into a "Limitations and open questions" subsection of
   §9** (or, conversely, expand §11 into a full discussion). A Part
   containing one section is structurally weak.
4. **Promote §5.7 (discrete extension) to its own section §6** so that
   the (R4) result and Bin(3,θ) counterexample are not buried under a
   §5.x heading; this matches its centrality after round 1.

## Abstract and introduction

The abstract (lines 57–129) is genuinely good — it states the problem,
the framework, the three ingredients, the four theorems, and the
empirical results in roughly that order. A reader who reads only the
abstract would know what this paper claims.

What the abstract does *not* do, and where an astronomer skimming might
get stuck:

- It does not say what a CD *is* until it uses the phrase "calibrated
  *confidence distribution* (CD) in the Schweder–Hjort / Fraser
  tradition." For an astronomer who has heard of Bayesian posteriors but
  not CDs, that's a closed-vocabulary signal. One additional sentence —
  "a distribution-valued summary of the evidence for θ whose quantile-
  based intervals contain θ₀ at exactly the nominal rate, for every θ₀
  — the frequentist analogue of the Bayesian posterior" — would resolve
  this without lengthening the abstract by much.
- It does not surface the SNL connection. Given that §3.4's central
  message is "this is the SNL loss with an architectural addition", the
  abstract burying this in "This is the same objective used by Sequential
  Neural Likelihood" — without flagging the strategic significance —
  understates the contribution.
- The phrase "pointwise in θ₀, not merely on average over the proposal"
  appears in italics in the abstract but is never re-anchored
  pedagogically afterward. A confused reader cannot trace this phrase to
  a specific section.

§1.1's "state of SBI" paragraph is well-aimed at someone who already
knows what NPE/NLE/NRE are and just wants the Hermans-trust-crisis
citation. An astronomer reader would need *one more sentence* per
method, or at minimum a forward-pointer to §10's comparison table (the
table that, per the previous section, ought to be moved up anyway).

## Forward/backward references

The manuscript uses dense `§X.Y` cross-references. Sampling:
- §2.2 forward-refs §6 for (R1^auto), §3.3 for the explicit
  characterization of the calibration manifold, §11.1 for the related
  open problem, §4.3 for the sign convention.
- §3.3 forward-refs §4 and §6.3 for uniqueness theorems.
- §3.5 forward-refs §8.4 for the empirical demonstration of (R2)'s
  necessity.

Forward-refs to material the reader hasn't yet seen are *frequent* and
sometimes *load-bearing* (the reader is being told "this is justified
later," not "see also later"). The §3.5 forward-ref to §8.4 is the most
striking case: the *justification* for an architectural prescription
in §3.5 lives 25 pages later in §8.4. A reader who stops at §3 has been
told "trust me, the experiment shows this" without seeing the
experiment.

Backward-refs are mostly clean and serve the reader well. The main
issue is forward-ref density in Parts I–II.

## Pedagogical scaffolding missing globally

### Glossary — RECOMMEND

A two-page front-matter glossary (before §1, or as Appendix A) covering:
- confidence distribution, pivot, calibration manifold, PIT, randomized PIT
- (R1), (R2), (R3), (R3_U), (R4), (C1), (C2) with one-line definitions
  and their first appearance
- KL divergence, monotone likelihood ratio, sufficient statistic,
  exponential family, ancillary statistic
- UMNN, normalizing flow, autoregressive flow, triangular flow
- Knothe–Rosenblatt rearrangement, Brenier map, Cholesky factorization
- UMP-unbiased CD, Schweder–Hjort UMPU CD
- NPE/NLE/SNL/NRE (one line each)
- M, F, ρ, P_θ, H_r, r*, r^KR, T(X), C_α, M_F

The manuscript already inlines several of these (e.g. §5.3's ancillary-
statistic gloss, §3.7's brief CvM/HSIC/CRPS glossary, §5.1's exponential-
family gloss). Lifting these into a single front-matter glossary would
do several things at once: (a) free the body text from repeating them,
(b) give the astronomer reader a single page to reference back to, (c)
make the (R1)/(R2)/(R3)/(R4) zoo navigable. The current state — where
(R3) means one thing in §4.4 and (R3_U) means a related thing in §5.7
and the manuscript explicitly notes the conflict inline — is exactly
the place where a glossary earns its keep.

### Roadmap diagram or table — STRONGLY RECOMMEND

The manuscript has *three* tables that collectively map the argument
(the §1.3 "objectives" table, the §8.5 "synthesis" table, the closing
"Status of claims" table) but none of them is a single global roadmap.
A one-page diagram of the form

```
Problem (§1.1) → Target (§1.2) → Pivot definition (§2)
                                      ↓
                          Loss that selects M (§3)
                                      ↓
            ┌────── 1D uniqueness (§4–5) ──────┐
            │                                  │
     loc-normal (§4)        exp family (§5)   discrete (§5.7)
                                      ↓
                       d-dim uniqueness (§6, KR)
                                      ↓
                          experiments (§7–8)
                                      ↓
                          recipe (§9) ←→ position (§10)
                                      ↓
                          open problems (§11)
```

— with which theorem applies where, which open problem maps to which
section, and where the experiments enter — would let an astronomer
decide on a reading path in under a minute. This is the single highest-
leverage pedagogical addition the manuscript could make.

### Running example — RECOMMEND

The location-normal model `X ~ N(θ, 1)` is already a de facto running
example: it appears as a worked instance in §1.2, §2.2, §4 (full theorem),
§5.3 (Student-t generalization with unknown variance), §6.3 corollary
(multivariate `L⁻¹(θ − X)`), §6.4 (z-interval inversion), §8.1
(empirical), §8.2–8.3 (multivariate empirical). The thread is already
there. What's missing is one paragraph at the top of §1 (or end of §1.3)
that *names* the running example and tells the reader "every Part of
this document will return to this model as a sanity check."

### Skip-able sidebars — CONSIDER

Each Part could open with a 2–3 line "What's in this Part / what to skim
if you trust the proofs" box. Specifically:
- Part III, opening: "If you trust that the Schweder–Hjort UMPU CD is
  the right target in regular 1D, the punchline is Proposition 4.5; the
  rest of Part III is the proof."
- Part IV, opening: "If you trust that triangular autoregressive flows
  exist and have positive triangular Jacobians, the punchline is the KR
  corollary in §6.3; the rest is the uniqueness proof."

This is a moderate intervention. The risk is that "skim if you trust
me" sidebars become an excuse not to argue. Recommend only if Parts III
and IV remain in their current proof-heavy form.

### Executive summary / "main results in one paragraph" — RECOMMEND

The abstract is good but is structured as a 5-paragraph narrative. A
*separate*, one-paragraph "main results" block — either at the top of
§1.3 or in place of the current §1.4 ("Scope") — would let a hasty
reader extract the four theorems + four experiments in one block. The
"Status of claims" table at the end does this but in tabular form and
buried at the back.

## Tone and voice

The §1.2 "astronomer-friendly comparison" framing is genuinely warm:
"A CD plays the same role as a posterior: a distribution-valued summary
of the evidence for θ." Compare to §3.2's opening: "Write the model
density implied by r as p̂_r(X | θ) := φ_d(r(θ, X)) · |det ∂r/∂X|. Under
(R2), p̂_r(·|θ) is a normalized probability density on X at every fixed
θ..."

These two passages are written for different readers. The shift happens
abruptly at the §2 → §3 boundary and is permanent thereafter. The §3.7
brief glossary block (one of the rare exceptions) is the only spot in
Parts II–IV that audibly addresses a non-specialist.

There are natural anchor points to re-engage the astronomer voice:
- Start of each Part (currently bare `\section*{Part X — title}` lines
  with no prose). Each Part-divider page could carry a 3–5 sentence
  "what this Part is for" paragraph in the §1.2 voice.
- The reduction-to-classical moments (Prop 4.5; §6.3 corollary). These
  *should* be the most astronomer-friendly passages — "here is where
  our abstract machinery gives back something you recognize" — but
  they read in the same dense voice as the surrounding proofs.
- The "worked example" lines (§5.3, §6.4) are also natural re-anchor
  points; currently they're terse.

Recommend: keep §1.2's voice for Part-opening paragraphs and for each
"reduction-to-classical" passage; keep current technical voice for
proofs themselves.

## Global oracle-style patterns

The user's second flagged failure mode appears at the *global* level
in several forms:

1. **"Calibration = pointwise frequentist coverage" is asserted in §1.2
   and §1.3 and then assumed throughout.** Sections §3 onward reason as
   if the reader has internalized that conditional `r(θ_0; X) | θ_0 ~
   N(0, I_d)` is *the* right target. The motivation lives only in §1.2's
   half-page; for a reader who didn't fully buy that section, the rest of
   the manuscript is asking them to trust the framing. Recommend: at the
   start of each Part, a one-sentence re-anchor ("Recall: we want `r`
   such that `r(θ_0; X) | θ_0 ~ N(0, I_d)` for every `θ_0`. This Part
   establishes...").

2. **"Architecture, not penalty" is asserted but never compared on a
   shared problem.** The manuscript repeatedly states that monotonicity
   must be enforced architecturally, not via penalty terms. §8.4's
   ablation is real and load-bearing, but it tests "architecture vs.
   no enforcement," not "architecture vs. penalty." A reader who
   defaults to Lagrangian thinking ("just add a penalty term") is given
   §3.5's theoretical mechanism but no head-to-head empirical
   comparison. This is a place where the manuscript *sounds* like it
   has settled the question but has only settled the easier version.

3. **"NF-MLE is the SNL loss" is presented as a near-throwaway
   identification.** It is actually the manuscript's central
   contribution-positioning claim: the loss is existing, the
   architecture is new, the resulting object is calibrated by
   construction. §3.4 makes this clear, but it does not return as a
   refrain. Sections §10 should explicitly use this framing again.

4. **"The proposal ρ is not a prior" — addressed well in §1.3 and §2.1,
   then quietly assumed.** Astronomer readers (who interact heavily
   with Bayesian SBI tools) will keep wanting to ask "but isn't ρ
   doing the work of a prior?" The manuscript answers this *once*. A
   re-anchor at the start of §8 ("the experiments use ρ = U[-7, 7];
   coverage is reported pointwise in θ_0, not averaged over ρ")
   would help.

5. **The implicit assumption that "monotone normalizing flow + KL
   training works" is taken as background.** §3.1's three-sentence
   normalizing-flow definition assumes the reader already trusts that
   diffeomorphism-via-NN-with-positive-Jacobian is a thing. This is
   fine for an ML-statistics reader; for an astronomer it's an
   un-motivated leap.

## Prioritized cross-cutting recommendations

In rough order of leverage-per-effort:

1. **Add a one-page roadmap diagram at the top of §1 (or as Figure 1).**
   Single biggest improvement to whole-document navigability. Pairs
   with the §1.3 objectives table; doesn't displace it.

2. **Add a two-page glossary as Appendix A (or front-matter), pulling
   together (R1–R4), (R3_U), the SBI-method acronyms, classical-stats
   primitives, and notation.** Lets the body text stop inlining
   definitions and gives the astronomer one page to reference back to.

3. **Promote §3.7 (design-space taxonomy) and move §10 (literature
   position) earlier — ideally as a paired "Where this work sits"
   section right after §1.1.** The current ordering (motivate →
   define → derive → 30 pages later: position) defers context the
   astronomer reader needs upfront. As a corollary, §3 becomes leaner
   and stays focused on the loss.

4. **Open each Part with a 3–5 sentence pedagogical paragraph in §1.2's
   voice, plus a "what to read carefully / what to skim" hint.** This
   re-engages the astronomer reader at five natural breakpoints and
   gives proof-heavy Parts a soft landing.

5. **Name the location-normal as the running example explicitly, in
   one paragraph at the end of §1.3.** The thread is already there;
   it just needs naming.

6. **Resolve forward-ref density in Part II by either inlining the
   §8.4 ablation result as a one-paragraph "preview" in §3.5, or by
   moving §3.5 itself later (after experiments).** Status quo asks the
   reader to take §3.5 on faith for 25 pages.

7. **Re-anchor "ρ is not a prior" at the start of §8 and again in §10.**
   Single highest-recurrence misreading risk for the astronomer
   audience; cheap to mitigate with two sentences placed where readers
   will encounter them at the moments they're most likely to slip.
