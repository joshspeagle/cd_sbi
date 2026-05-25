# Round 3 — Part II (Loss design) Discovery & Assessment

Scope: `cd_sbi_v7.tex` §3, lines 482–793. Subsections 3.1–3.5 + 3.7
(§3.6 is intentionally skipped by `\setcounter` and not a bug).

## Phase A: Discovery

### Top-line goal(s)

Part II answers two yoked questions: (i) *what loss should we use to
train the pivot `r`*, and (ii) *why is that loss the right one*. The
goal is to introduce **NF-MLE** (the normalizing-flow maximum-likelihood
objective), prove it is strictly proper for the calibration manifold
`M_F`, identify it with the SNL loss to inherit existing SBI tooling,
flag the architectural monotonicity-in-`X` requirement on which
propriety silently depends, and situate the resulting loss within a
five-class taxonomy that the manuscript itself constructs.

### Central arguments

1. **(§3.1) The loss is just maximum likelihood under the
   change-of-variables density implied by `r`.** Define
   `\hat p_r(X | θ) = φ_d(r(θ, X)) · |det ∂r/∂X|`; NF-MLE is the
   negative log-likelihood under `\hat p_r`. The box
   `L(r) = E[½‖r‖² − log|det ∂r/∂X|]` is the operational statement.

2. **(§3.2) NF-MLE is strictly proper for `M_F` via the standard
   forward-KL identity.** Under (R2), `\hat p_r(· | θ)` is a normalized
   density, so `E[−log \hat p_r] = H(X|θ) + KL(p ‖ \hat p_r)`. Averaging
   over `ρ` makes `L(r)` equal `E_ρ[KL] + const`; KL non-negativity gives
   the population minimum exactly on `M_F`.

3. **(§3.3) NF-MLE picks an element of `M`; the architectural class
   picks *which one*.** The unconstrained calibration manifold is
   infinite-dimensional (any sign-flip / measure-preserving rearrangement
   / orthogonal rotation stays in `M`). Restricting `F` to `C¹`-monotone
   (1D) or autoregressive-triangular (multivariate) shrinks
   `M_F` to a singleton — Theorems A and A-d.

4. **(§3.4) NF-MLE is identical to the SNL training loss.** A
   three-column SNL-vs-CD-SBI table makes the point: same loss, different
   architecture, different inference path, different coverage guarantee.
   Practical payoff: CD-SBI inherits SNL's optimization tooling.

5. **(§3.5) Without architectural monotonicity in `X`, the loss is
   unbounded below — the central cautionary tale.** Autograd-computed
   Jacobians let the network "fold" the `X`-map, breaking bijectivity;
   `Z(θ) = ∫ \hat p_r dX > 1`; the `−log Z(θ)` term breaks the
   conditional-entropy lower bound; loss falls below the truth's loss.
   Empirical demo deferred to §8.4 (0.56 < 0.88).

6. **(§3.7) Among five candidate loss families, NF-MLE (Class 5) is the
   one that scales, has a strong per-sample gradient, and is in the
   *correct direction* of strict propriety.** Class 1 (marginal-only)
   fails by Hermans-trust-crisis counterexample; Class 2 (marginal +
   HSIC) has weak gradients and balance hyperparameters; Class 3
   (stratified) scales as `K^d`; Class 4 (CRPS/pinball against `N(0,I)`)
   has the propriety direction reversed and collapses to a point mass at
   the median.

7. **(§3.7, positioning subsection) NPE, SNL, NRE, LF2I, WALDO, Box CD
   are all locatable inside the five-class taxonomy.** SNL and CD-SBI
   share Class 5 on `X | θ`; CD-SBI's contribution is the architectural
   prescription that converts the SNL output into a frequentist CD.

### Supporting evidence per argument

- **Arg 1**: the change-of-variables formula, presented in line (496),
  and the boxed loss in (501). One paragraph (487–504).
- **Arg 2**: the KL identity in (513), the equality `L(r) = E_ρ[KL] +
  const` in (515), the `argmin = M_F` statement in (525). No worked
  example or pedagogical decomposition beyond the symbols.
- **Arg 3**: an English summary of the size of the unconstrained `M`
  (rotations, sign flips, rearrangements) at (545–551); a bulleted
  forward reference to Theorems A and A-d at (557–568).
- **Arg 4**: the longtable at (583–608); SNL citation
  `\citep{PapamakariosEtAl2019}`; the parenthetical at (610–617) that
  "(a) exposes a monotone pivot directly, (b) makes pivot inversion
  well-defined."
- **Arg 5**: a derivation of the `−log Z(θ)` shift at (635–643); a
  pointer to §8.4 for the empirical loss-below-truth (645).
- **Arg 6**: per-class verdicts at (678–765), including a fully worked
  collapse-to-`δ_0` argument for CRPS using both the kernel-score
  representation (727–740) and the elementary `F²` expansion (742–748).
- **Arg 7**: the SBI-positioning paragraph at (767–790) with named
  citations for each method.

## Phase B: Accessibility assessment

### Failure mode 1: Undefined terms for non-specialists

Audience: an astronomer who has seen MCMC, MLE, and posteriors, but not
necessarily flows or scoring-rule theory. Terms catalogued:

- **Normalizing flow** (line 489). Defined in one sentence: "a
  parametric family of diffeomorphisms `r : X → ℝ^d` that transforms a
  complicated source density into a simple base density." For an
  astronomer this is *terse*. Three follow-on terms — *diffeomorphism*,
  *change-of-variables formula*, *base / source density* — are used in
  the same paragraph without their own gloss. Net effect: a reader who
  doesn't already know what a flow is gets a citation
  (`PapamakariosEtAl2021`) and a single sentence, and is then immediately
  asked to parse a boxed loss containing `log|det ∂r/∂X|`. **The
  user-flagged failure mode lives here.** Even one schematic diagram
  ("flow transports `X | θ` to `N(0, I)`; loss = `−log` of pulled-back
  density") would unblock the reader, but none is provided.

- **Diffeomorphism** (489). Never defined. Adequate gloss for an
  astronomer is roughly two clauses: "smooth, invertible, with smooth
  inverse." Absent.

- **Change-of-variables formula** (494, then again at 511, 625, 760).
  The formula `|det ∂r/∂X|` appears with no derivation, no analog
  ("Jacobian of a coordinate transform — the same thing that puts the
  `r²sinφ` in spherical integration"), and no statement of why it is
  required for `\hat p_r` to integrate to 1. Critical because the
  *entire* §3.5 argument is about what goes wrong when the formula's
  precondition (bijectivity) fails.

- **Jacobian determinant**. Used in the loss box (501) but never named
  as such, never connected to the elementary intuition that it measures
  local volume change. An astronomer who took multivariable calculus
  twenty years ago needs the connection spelled out.

- **Strict propriety** (§3.2, line 532). Defined inline, well: "a loss is
  strictly proper for a target `M` if its population-level minimum, over
  the class of admissible `r`, is attained exactly on `M` and nowhere
  else." This definition is the model for what the other terms should
  get and currently don't.

- **Kullback–Leibler divergence / KL** (513). Used as if standard; the
  identity `E[−log \hat p_r] = H + KL` is invoked with the bare word
  "standard." For astronomers, KL is the *less* familiar of the two
  intuitive notions of "distance between distributions" (the other being
  `χ²` or RMS). A sentence — "KL is non-negative, zero only when the two
  densities agree almost everywhere; this is Gibbs's inequality" — would
  let an astronomer follow the propriety argument without a side trip to
  Cover & Thomas.

- **Forward-KL projection.** Never named, but the §3.2 identity is
  exactly the forward-KL projection. Stating that connection explicitly
  ("we are projecting the true `p(· | θ)` onto the model family in
  forward-KL — the same direction MLE uses") would orient anyone who has
  seen variational inference.

- **Gibbs's inequality.** Mentioned once at line 642 with no
  introduction, as the bound that's broken by folding. An astronomer
  will not know what was broken.

- **Cramér–von Mises, energy distance, HSIC, CRPS, pinball loss** —
  §3.7 has a "brief glossary" block (666–676) covering CvM, energy
  distance, HSIC, CRPS, pinball. **This is the one place the draft does
  the right thing.** The glossary should be the model for the rest of
  §3. Even so, *energy distance* is glossed as "an `L²`-distance between
  CDFs in higher dimensions," which is a partial truth — the kernel
  form actually used in the Class-4 argument (line 728) is not mentioned
  in the glossary, so the glossary doesn't actually equip the reader to
  parse the proof a paragraph later.

- **Class 1–5 taxonomy.** The manuscript invents this taxonomy and uses
  it as a load-bearing organizational device for §3.7 and the
  SBI-positioning paragraph. Verdict: the taxonomy *does* land, because
  each class has a single sentence of construction and a bolded one-line
  verdict ("Insufficient on its own", "Doesn't scale", "Directional
  propriety error"). But there is no preview list before Class 1 —
  the reader is reading a taxonomy without first being told there are
  five classes and how they relate. Adding a half-paragraph "we will
  consider five families: (1) marginal-only, (2) marginal + independence,
  (3) stratified, (4) scoring rules against `N(0,I)`, (5) per-sample log
  density — and only the last avoids all three of marginal-vs-conditional
  collapse, exponential bin blowup, and propriety-direction reversal"
  would lift §3.7 from a list to a comparison.

### Failure mode 2: Oracle-style passages

§3 has three concentrated oracle-style passages.

- **§3.2 KL-identity derivation (line 513).** The reader is shown a
  one-line identity and told its consequence. They are not walked
  through the logical chain "log-likelihood under a normalized model →
  cross-entropy → entropy + KL → averaging over `ρ` preserves the
  decomposition → minimizing log-likelihood is minimizing KL." For an
  astronomer who knows MLE but not its variational interpretation, the
  whole §3.2 mechanism reads as algebra-without-story. The English
  bridge that's needed is: "training to maximize likelihood is exactly
  training to minimize a forward-KL divergence; the conditional entropy
  `H(X | θ)` is a constant of the data that we cannot reduce; the
  remaining piece is the KL, which is zero iff our model matches the
  truth." This is a four-line addition that would convert §3.2 from a
  one-step derivation into a told story.

- **§3.5 `−log Z(θ)` mechanism (lines 622–645).** This is the most
  important explanatory passage in Part II and currently the most
  oracle-style. The reader is asked to absorb, in 23 lines:
  (i) without (R2) `\hat p_r` is not a density; (ii) folding inflates
  the local Jacobian; (iii) the surrogate's total mass exceeds 1;
  (iv) the KL identity then picks up a `−log Z(θ)` shift; (v) Gibbs's
  inequality no longer gives a conditional-entropy lower bound;
  (vi) loss can fall below the truth's loss; (vii) §8.4 confirms with
  `0.56 < 0.88`. Every link is correct, every link is one sentence
  with no analogy. The causal chain the user asked for —
  *folding → inflated Jacobian → inflated `\hat p_r` → `Z > 1` →
  `−log Z(θ) < 0` → loss below entropy bound* — is exactly the chain
  in the manuscript, but it is buried in symbols. **An astronomer
  reading this will register only the bad conclusion.** What is needed
  is a story: "imagine `r` folds the `X` axis on itself, so two values
  of `X` map to the same value of `r`. The Jacobian `|∂r/∂X|` at each
  preimage is then artificially small, so reciprocally `\hat p_r`
  (which is `φ(r)·|∂r/∂X|^{-1}` in the local form) inflates. The
  network has invented free probability mass. Plugging the inflated
  `\hat p_r` back into the KL identity adds a `−log Z(θ) < 0` term
  that lowers the loss. The optimizer happily walks the network into
  this regime, achieving a loss below what the *true* `r*` achieves —
  the autograd-Jacobian trick rewards self-folding."

- **§3.7 per-class verdicts.** The Class-1 verdict ("Insufficient on
  its own") *is* given a counterexample (lines 684–692, the `θ = ±1`
  half-normal mixture). This is the gold standard within §3 — a
  one-line claim followed by an explicit construction. Class 2's
  verdict ("empirically fragile in two ways") is also adequately
  unpacked (lines 698–707). Class 3 is dispatched in two lines without
  development; this is fine because the failure (`K^d` bins) is itself
  short and self-evident. Class 4 is *over*-developed by the §3 norm —
  two independent derivations (kernel-score and elementary expansion)
  both showing the same collapse — but the *reason* the reversal
  matters is not bolded out: an astronomer may not immediately see
  why "minimizing over the sample distribution with `F` fixed" is the
  wrong direction. A single sentence ("the sample is supposed to be
  *given* and the forecast *learned*; here it is the opposite, so the
  loss rewards the learner for collapsing instead of spreading") would
  cap the section. Class 5 ("strictly proper, scales, identical to
  SNL") is one paragraph, which is right because the work was already
  done in §3.2.

### Failure mode 3: Proof-role / definition-role clarity

- **§3.2's strict-propriety theorem is implicit.** The result
  `argmin L(r) = M_F` at (525) is the strict-propriety theorem of Part
  II, but it is not labelled as a theorem, not given a name, and is not
  previewed before the derivation. The reader gets to the boxed
  conclusion and then has to look back to identify what was proved.
  Either elevating to a labelled proposition ("Proposition 3.1
  (Strict propriety of NF-MLE within `F`)") or at least prefacing the
  derivation with "we now prove that NF-MLE is strictly proper for
  `M_F`" would make the proof's role explicit.

- **§3.5 has no theorem-statement.** The `−log Z(θ)` mechanism is
  presented as an explanation, not a result. Given that this mechanism
  is the *empirical* central claim of the whole paper (§8.4 is built
  around it), elevating it to a labelled "Failure mode 3.1 (Loss below
  entropy bound under autograd Jacobian)" would advertise its
  importance. This is a stylistic call, but the round-3 brief
  explicitly named §3.5 as a candidate for oracle-style failure, and
  the absence of a labelled result contributes to that effect.

- **§3.7 Class-4 collapse derivation.** The CRPS-collapses-to-`δ_0`
  result *is* a non-trivial calculation occupying lines 720–748. It
  deserves its own labelled proposition. As written, an astronomer
  reading the section first sees "Directional propriety error" in bold
  and then a wall of integrals; without the result-statement up front,
  there is no anchor for what the wall is proving.

- **§3.4 SNL table.** Role is clear (a comparison). Adequate.

- **§3.3 selection-from-`M` claim.** Role is clear (a forward
  reference). Adequate, modulo the issue under Failure mode 4 below.

### Failure mode 4: Concept-introduction gaps

- **Normalizing flow.** One paragraph (487–498), already covered under
  Failure mode 1. This is the largest single accessibility gap in Part
  II. Astronomers should be told (i) "a flow is a learned smooth
  invertible map from data space to a simple Gaussian latent space";
  (ii) "training the flow means adjusting the map so that the implied
  density of the data, computed by the change-of-variables formula,
  matches the empirical distribution"; (iii) "we use the flow as a
  pivot — the latent space is the standardized residual space where
  calibration is just Gaussianity." A figure showing
  `X | θ` (curved or messy) being transported to `N(0, I)` (round)
  would do the work of all three sentences.

- **Change of variables.** Already covered. The single missing
  intuition is "Jacobian = local volume scaling." Without it §3.5 is
  unreadable.

- **Calibration manifold `M_F`.** Defined in Part I §2.2 (line 408),
  used continuously throughout §3 (`M_F` appears at lines 525, 561,
  566, 661, 761). The concept is well-motivated in Part I as "the set
  of pivots whose induced law on `r(θ_0; X) | θ_0` is `N(0, I)` for
  every true `θ_0` in the proposal's support." Part II re-uses without
  re-establishing. **Recommendation**: a one-line callback at the
  start of §3.2 or §3.3 — "recall `M_F` is the set of pivots that are
  exactly calibrated for every `θ_0`; the question is which loss
  recovers it" — would re-anchor the reader who has been away for a
  Part. The cost is one line. The current draft assumes Part I is
  still loaded in the reader's head.

- **Architectural class `F`.** Used at lines 525, 526, 554, 561, 566.
  Astronomers are unlikely to have an intuition for "function class"
  as a mathematical object. The needed gloss is "`F` is the set of
  flows your network can possibly express — UMNN, autoregressive
  triangular, etc.; restricting `F` is what your architecture does
  for you." A single sentence at the first use of `F` in §3.3 would
  prevent the slogan "architecture picks out a unique element"
  (line 570) from being content-free.

- **(R1), (R2), (R3).** Defined in Part I §2.2/§2.3/§2.4 and used in
  §3 as labels. Recall is light — `(R2)` is invoked at line 509 with
  "Under (R2), `\hat p_r(· | θ)` is a normalized probability density."
  A reader who has forgotten which regularity hypothesis is which
  loses the argument. A parenthetical recall — "(R2: `X ↦ r(θ, X)` is
  a `C¹` diffeomorphism at fixed `θ`)" — at first reuse of each label
  would cost three short clauses and prevent flipping back.

- **SNL.** Lines 573–581 do a reasonable job introducing SNL as
  "sequential by default, refits on simulator outputs near the
  current posterior estimate, proposal `ρ` changes between rounds." For
  an astronomer who has not seen SNL, "refits the flow on simulator
  outputs drawn near the current posterior estimate" is the only piece
  of context they have. Adequate; could be one sentence richer
  ("SNL trains a likelihood surrogate `\hat p(X | θ)` rather than a
  posterior `p(θ | X)` and runs MCMC on the surrogate at the end") but
  Part I §1.1 already does that. Pass.

### Failure mode 5: Narrative flow

The section structure is logical on paper:
3.1 here is the loss → 3.2 here's why it's proper → 3.3 here's what it
selects → 3.4 it's the SNL loss → 3.5 here's why (R2) matters → 3.7
here's the design space.

Three flow weaknesses:

- **§3.1 → §3.2 is too abrupt.** §3.1 is one paragraph and a box; §3.2
  immediately starts a KL derivation. The reader has not had time to
  internalize what `\hat p_r` *is* before being told the population
  minimum is on `M_F`. A two-sentence bridge — "we now check that this
  loss does the right thing: its population minimum is exactly the set
  of pivots that are calibrated for every `θ_0`" — would let the
  reader follow §3.2 as a confirmation rather than an opaque
  derivation.

- **§3.5 → §3.7 transition.** After the loss-below-truth alarm of
  §3.5, the reader is dropped into a six-page taxonomy comparison
  with no transitional sentence. The current draft uses a horizontal
  rule (`\medskip\hrule\medskip`, line 655). A bridge — "§3.2–§3.5
  argued that NF-MLE under (R2) is strictly proper for `M_F`; this
  subsection asks whether any *other* loss could play the same role"
  — would make §3.7 feel like a justification of an earlier choice
  rather than a comparison-shopping detour.

- **§3.7 is internally a long list.** Five classes plus a positioning
  paragraph, ~130 lines, no internal preview or summary. As noted
  under Failure mode 1, an opening sentence enumerating the five
  classes and the three independent failure modes (marginal-vs-
  conditional, exponential bin blowup, propriety-direction reversal)
  would convert the section from "five paragraphs of unrelated
  verdicts" into "three obstructions, four classes that hit one of
  them, NF-MLE that avoids all three." Same content, far more
  navigable.

The longtable at §3.4 (583–608) is a strong piece of structural
writing; it stays.

## Prioritized recommendations

In rough priority order — highest first, lowest at the bottom.

1. **Expand §3.1's introduction of normalizing flows to one
   pedagogical page.** Add the three sentences listed under
   Failure mode 4 (data-to-`N(0, I)`, training-by-change-of-variables,
   pivot-as-latent-residual). Add a small figure (boxed `X | θ`
   transported to a round `N(0, I)` blob with a learnable arrow)
   if production allows. Define *diffeomorphism* in a phrase, *change
   of variables* with the "local volume scaling" analogy, *base
   distribution* explicitly. This is the single highest-impact change
   in Part II and the one the user flagged.

2. **Rewrite §3.5 as a story, then derive.** Add a 4–6 sentence
   English narrative at the top — folding → inflated Jacobian →
   inflated `\hat p_r` → `Z(θ) > 1` → `−log Z(θ) < 0` shift →
   loss below entropy bound — and keep the derivation as the formal
   backup. Optionally elevate to a labelled "Failure mode" environment
   so the result has the prominence the §8.4 experiment gives it.

3. **Add a §3.7 opening preview.** Half a paragraph enumerating the
   five classes and the three independent failure modes; this converts
   the section from a list into a comparison. While you're there, add
   a one-sentence "why does the reversal matter" gloss at the end of
   Class 4 and a one-sentence summary at the end of Class 5 ("so we
   choose Class 5, with the architectural prescription of §3.5").

4. **Add four short term-glosses on first use**: *KL divergence*
   (in §3.2, one sentence including "Gibbs's inequality says it's
   `≥ 0` with equality iff densities agree"), *Jacobian determinant*
   (in §3.1, one phrase: "the local volume-scaling factor"),
   *diffeomorphism* (in §3.1, "smooth, invertible, with smooth
   inverse"), *Gibbs's inequality* (in §3.5, where it's invoked).

5. **Elevate the strict-propriety result to a labelled
   proposition.** §3.2's `argmin L = M_F` is currently inline; making
   it Proposition 3.1 advertises its role and lets §3.3, §3.5, §3.7,
   and Parts III–IV refer to "Proposition 3.1" by name instead of
   "the argument of §3.2."

6. **Add the forgotten-from-Part-I recalls.** A one-line restatement
   of (R1)/(R2) at first reuse in §3, and a one-line restatement of
   the calibration manifold `M_F` at the start of §3.2 or §3.3.
   Trivial cost, big readability win.

7. **Bridge sentences between subsections.** Two-sentence bridges
   between §3.1→§3.2 and §3.5→§3.7. Replace the silent `\hrule`
   between §3.5 and §3.7 with prose.

8. **Expand the §3.7 glossary to include the kernel-score form of
   CRPS** so the Class-4 derivation at line 728 is self-contained,
   and rename "brief glossary" to something more inviting (e.g.,
   "A vocabulary aside"). Minor.

9. **Optionally: elevate the §3.7 Class-4 CRPS-collapse-to-`δ_0`
   derivation to a labelled proposition.** This is genuinely a
   small theorem and deserves the visibility.

## Existing strengths to preserve

- The **§3.4 SNL/CD-SBI longtable** is a model of comparative
  exposition: same loss, different architecture, different inference
  path, different guarantee. Keep verbatim.

- The **§3.7 Class-1 counterexample** (the `θ = ±1` half-normal
  mixture) is exactly the right level of concreteness; it's the
  template every other class would benefit from.

- The **§3.7 brief-glossary block** (lines 666–676) is the only place
  in Part II that pre-defines its vocabulary. It should become the
  template for the rest of §3.

- The **§3.5 `Z(θ) > 1` derivation** is mathematically clean and
  pointed; the recommendation is to *prepend* a narrative, not to cut
  the derivation.

- The **§3.7 SBI-positioning paragraph** (767–790) does important work
  — placing NPE, SNL, NRE, LF2I, WALDO, Box CD inside the same
  taxonomy. This earns the length of §3.7 even before any pedagogical
  improvements.

- The **boxed loss at (501)** with the irrelevant constant explicitly
  dropped is the right kind of statement: it's the one thing the
  reader needs to remember from §3.1 and it's visually marked as such.

- The **"no auxiliary penalties needed" closing of §3.2** (538–541)
  is the right kind of editorial point — calibration *and* sharpness
  are already in the per-sample log-density. Don't lose it.
