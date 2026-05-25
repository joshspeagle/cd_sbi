# Round 3 — Part I (Framework) Discovery & Assessment

Scope: `cd_sbi_v7.tex` lines 140–481 (§§1.1–2.4). The job is to assess
pedagogy and accessibility for a mixed audience of statisticians and
astronomers, with attention to undefined terms, oracle-style assertion,
proof/definition-role clarity, concept introduction, and narrative flow.

## Phase A: Discovery

### Top-line goal(s) of Part I

1. **Reorient the reader from the Bayesian SBI default to a frequentist
   target.** Convince a reader steeped in NPE/NLE/NRE (or in Bayesian
   astronomy) that there is a different object — the confidence
   distribution — that one can target *primarily* from a simulator, and
   that miscalibration in mainstream SBI is the empirical motivation.
2. **State precisely what the rest of the paper produces, and what
   structural conditions that object has to satisfy.** Concretely:
   define the pivot `r`, the induced CD `H_r = Φ_d ∘ r`, and the
   calibration manifold `M_F`; introduce the two axioms (R1) and (R2)
   that the rest of the paper will treat as architectural, not
   penalty-based, requirements.

### Central arguments

1. **Mainstream SBI is systematically miscalibrated, and retrofitted
   calibration is a workaround rather than a fix.**
   - **Where:** §1.1, lines 145–192.
   - **Supporting evidence:** citation to `HermansEtAl2022` (the "trust
     crisis" paper) plus a survey-style citation list of retrofit
     methods (Delaunoy 2022; Falkiewicz 2023; LF2I; WALDO; Bortolato &
     Ventura 2025; Patel 2023). No worked example or quantitative
     illustration — entirely citation-supported.
2. **A confidence distribution is the right primary target if you want
   pointwise frequentist coverage, and it has a long pre-SBI lineage.**
   - **Where:** §1.2, lines 193–224.
   - **Supporting evidence:** Schweder–Hjort Def. 3.1 invoked verbatim
     (lines 207–212); citation chain Fisher 1930 → Cox 1958 → Efron
     1993/1998 → Singh 2007 → Xie–Singh 2013 → Schweder–Hjort 2016;
     contrast with Bayesian posterior via Fraser 2011.
3. **In the location-normal model the optimal CD has a closed form, and
   it is the textbook z-interval.** This is the proof-of-concept that
   the framework is not vacuous.
   - **Where:** §1.2, lines 242–253.
   - **Supporting evidence:** Direct one-line calculation
     `Φ(θ₀ − X) = Φ(−Z) ∼ U(0,1)`; inversion to recover
     `[X − z_{1−α/2}, X + z_{1−α/2}]`. The only worked example in
     Part I.
4. **The top-line goal of the paper can be stated as two characterizing
   properties (C1) and (C2).** This is the contract Part I makes with
   the rest of the document.
   - **Where:** §1.3, lines 258–293, plus the "subordinate objectives"
     table at lines 296–324.
   - **Supporting evidence:** Definitional. No proof — the table
     enumerates which later sections discharge which obligations.
5. **The proposal ρ is a sampling device, not a prior.** This is the
   *load-bearing* conceptual claim that distinguishes CD-SBI from
   Bayesian SBI methods that look superficially similar (SNL in
   particular).
   - **Where:** §1.3, lines 326–335.
   - **Supporting evidence:** Informal argument plus pointer to §11.4
     for the misspecified / under-parameterized case where the claim
     becomes false. The full strict-propriety argument that makes the
     ρ-independence rigorous is in §3.2 (Part II).
6. **The "pivot" object is defined by (R1) monotonicity in θ and (R2)
   invertibility in X.** These are stated as the only structural
   requirements on `r` going forward.
   - **Where:** §2.2, lines 377–404.
   - **Supporting evidence:** Definitional plus the location-normal
     `r*(θ, X) = θ − X` worked through (R1), (R2) — lines 414–422.
7. **(R1) is needed because calibration alone leaves an
   infinite-dimensional ambiguity in the choice of `r`, and (R2) is
   needed because without it the NF-MLE loss is not even well-posed.**
   - **Where:** §2.3, lines 424–455 and §2.4, lines 457–478.
   - **Supporting evidence:**
     - §2.3: Informal characterization of the ambiguity (sign flips,
       measure-preserving rearrangements, `O(d)` rotations) and forward
       pointer to §3.3 for the explicit version.
     - §2.4: Sketch of the change-of-variables breakdown (preimages
       summed wrongly; loss unbounded below); forward pointer to §3.5
       (formal) and §8.4 (empirical demonstration).

## Phase B: Accessibility assessment

### Failure mode 1: Undefined terms for non-specialists

The audience is mixed: a statistician will know most of these, an
astronomer will know few. Listing terms in order of first appearance.

- **"implicit" model** at line 147: defined inline ("defined by a
  stochastic simulator … but for which the likelihood … is intractable
  or unavailable in closed form"). *Sufficient.*
- **"surrogate"** at line 152: used in the generic ML sense but not
  defined. For an astronomer not steeped in ML this is jargon. A
  one-clause gloss ("a neural-network approximation of an otherwise
  intractable function") would carry the rest of §1.1.
- **"posterior", "likelihood", "likelihood ratio"** at lines 153–154:
  not glossed. A trained Bayesian astronomer knows them; a frequentist
  reader may not flinch but a junior reader will. Probably fine to
  leave, but the *contrast* with the CD object only lands if the reader
  knows what a posterior is.
- **"credible interval"** at line 183: undefined; central to the
  miscalibration claim. Astronomers often conflate "credible" and
  "confidence" intervals in practice. One sentence distinguishing them
  here would do double duty as motivation for §1.2.
- **"confidence distribution" (CD)** at lines 197 and 199: this *is*
  defined immediately ("the frequentist analogue of the Bayesian
  posterior: a data-dependent distribution function `H(·; X)` on Θ
  whose α-quantile is the upper endpoint of an exact one-sided
  α-confidence interval"). The definition is technically correct but
  *dense*: "α-quantile is the upper endpoint of an exact one-sided
  α-confidence interval" requires the reader to mentally invert the
  CDF, recognize the duality between CDs and one-sided CIs, and parse
  "exact" (= equality, not bound). A worked illustration (e.g. point
  to the location-normal example *forward* before stating the abstract
  definition) would help. Currently the abstract definition lands
  before the concrete one.
- **"pivot"** at line 218 (used informally before being defined) and
  again at line 267 (defined). The line-218 first use ("applied to a
  correctly chosen *pivot*") slips by, and the line-267 definition is
  the classical one ("function of data and parameter whose distribution
  does not depend on the parameter"). This is the right place to define
  it, but the wording "slightly stronger property" elides what is
  actually being strengthened (the distribution under truth is
  *specifically* `N(0, I_d)`, not just θ-free). A reader may not
  realize that the choice of base distribution is essentially
  conventional / WLOG.
- **"probability integral transform (PIT)"** at line 216–218: glossed
  inline ("the mapping that converts any continuous random variable
  into a uniform (0,1) variable via its own CDF"). *Sufficient* for a
  statistician; astronomers who have not seen the PIT will at least
  have a one-line operational handle.
- **"UMPU" (uniformly most powerful unbiased)** at line 244: spelled
  out once but not motivated. The reader is told there is an optimal
  CD without being told what "optimal" means here (smallest expected
  CI length? best power against alternatives in the dual testing
  problem?). The footnote-level cost of saying "optimal in the sense
  of having the most powerful unbiased size-α test in the dual testing
  problem; see §5" would clear this up. As a hook for a non-specialist
  this matters because it is the *answer* to "why bother with CDs."
- **"calibration manifold"** at line 213 (cross-reference) and lines
  408–410 (definition). The definition uses set-builder notation with
  `Law(r(θ₀; X) | θ₀)` — that conditional-law notation is unusual in
  astronomy. Glossing the line above ("the set of `r` that, evaluated
  at each true θ₀ and on data drawn from `P_{θ₀}`, has the right
  sampling distribution") would carry it.
- **"normalizing flow"** at line 213 (alluded to as NF-MLE) — *not
  defined in Part I*. The reader has to wait until §3.1 (line 489) to
  learn what an NF is. But NF-MLE is mentioned by name at lines
  213–214, 326, 333, and 477, all in Part I, including in load-bearing
  conceptual statements ("NF-MLE drives the network toward [exact
  coverage]" at lines 213–214). For a Part-I-only reader this is a
  reference into a void.
- **"change-of-variables Jacobian / formula"** at line 460: stated as
  the Jacobian formula directly with `|det ∂r/∂X|`. A non-statistical
  reader knows this from multivariable calculus but may not connect it
  to density transformations. The sentence at lines 459–464 is dense
  and would benefit from one sentence of intuition ("the determinant
  factor corrects for how much `r` locally stretches volume").
- **"Knothe–Rosenblatt (KR) transport / rearrangement"** at line 290
  and lines 452–454: not defined in Part I. The reader is told the
  autoregressive form *is* a KR transport but no operational handle is
  given. Since this is the named concept that does the heavy lifting in
  Part IV, even a one-sentence preview ("an autoregressive ordering of
  the coordinates that turns a multivariate transport into a chain of
  conditional 1D transports") would help.
- **"strict propriety / strictly proper"** — *not in Part I at all*,
  but the §1.3 table at line 314 names "NF-MLE" as the loss whose
  minimizers are `M`, which is the strict-propriety statement
  rephrased. A footnote or a one-sentence preview here would prepare
  the reader for §3.2.
- **"forward-KL projection / weighted KL projection"** at line 333:
  this is buried but it is the precise statement of what happens under
  misspecification. The reader is asked to take it on faith here. A
  half-sentence gloss ("the element of F that minimizes the average
  KL divergence from the true model, weighted by ρ") would help.
- **"monotone likelihood ratio (MLR)"** at line 242: stated by name
  without definition. This term is from textbook hypothesis testing
  and will be opaque to most astronomers. A bracketed gloss ("MLR: the
  likelihood ratio between two parameter values is a monotone function
  of the sufficient statistic") would suffice.
- **"sufficient statistic"** at line 243: used without definition.
  Same problem as MLR. Often co-occurs with MLR in stats texts but is
  itself nontrivial.
- **"location-normal model"** at line 245, line 414: used freely. A
  statistician parses it; an astronomer may not know it is shorthand
  for `X ∼ N(θ, 1)` with θ the location parameter. The model is
  *displayed* on the next line, so context resolves it, but the term
  could be glossed at first use.
- **`C¹ diffeomorphism`** at line 397: a heavy term to drop on an
  astronomer. The conceptual content is "a smooth invertible map with
  smooth inverse" — saying this in words first, then naming the
  technical term, would be standard pedagogical practice.
- **"full-rank Jacobian"** at line 398: glossed contextually but the
  asymmetry (full-rank a.e. ≠ injectivity for Lipschitz maps) is
  flagged at lines 400–403 in a way that is highly technical without
  the §4.4 context. This is a forward-pointer to a subtle point that
  the reader cannot evaluate yet — and the parenthetical "for ReLU
  network arguments" is opaque if the reader has not seen §4.4.
- **`O(d)` (orthogonal group)** at line 447: used without definition.
  For a statistician this is shorthand; for an astronomer "orthogonal
  group" may or may not register depending on background. The
  statement "(the entire orthogonal group `O(d)` in `d > 1`)" can be
  glossed as "any rotation or reflection of `R^d`" inline.
- **"measure-preserving bijection of `N(0, I_d)` to itself"** at lines
  444–445: technical phrasing. Spelling out one example (a sign flip;
  a rotation) before the abstract phrase would carry it.

### Failure mode 2: Oracle-style passages

These are passages where the conclusion is asserted in friendly
language but the reader cannot reconstruct why the move is the right
one. Each item gives the quote, what is taken on faith, and what would
have to be added.

- **Lines 195–197 ("We instead change the primary target of inference.
  Rather than estimating a posterior and correcting it, we aim to
  produce — from the simulator alone — a confidence distribution.")**
  The reader is told that "the primary target" changes, but the
  preceding paragraph (§1.1) does not actually establish that *changing
  the target* is the right move. The §1.1 indictment is that
  retrofitted calibration is a workaround; but the reader is not shown
  *why* a CD is structurally easier to calibrate than a posterior, or
  why correcting a posterior is hopeless rather than just hard. This is
  the central sales pitch and it is asserted, not earned. *Fix:* a
  one-paragraph bridge explaining that frequentist coverage is a
  *pointwise* property whereas Bayesian coverage is a *prior-averaged*
  property, and that the latter cannot be turned into the former by any
  post hoc procedure that doesn't itself sample from the truth — so
  the choice is structural, not aesthetic.
- **Lines 207–214 ("Schweder–Hjort require two properties: (i) ...,
  (ii) the calibration condition exact coverage: `H(θ₀; X) ∼ U(0,1)`
  when `X ∼ P_{θ₀}` ... Property (i) is enforced architecturally in
  this paper by (R1) (§2.2); property (ii) is what NF-MLE drives the
  network toward.")** The reader is told (i) is architectural and (ii)
  is driven by training, but at this point in the document the reader
  doesn't know what "architectural" means in this context, what
  NF-MLE is, what the network is, or what "drives toward" means
  operationally. Three forward pointers in two sentences, none of
  which the reader can resolve. *Fix:* either move this sentence to
  the end of §1.3 once the (C1)/(C2) decomposition is in hand, or
  expand to one paragraph that explains the decomposition before
  pointing forward.
- **Lines 219–224 ("Astronomer-friendly comparison. A CD plays the
  same role as a posterior: a distribution-valued summary of the
  evidence for θ.")** This passage *almost* works — it sets up the
  comparison cleanly. But the punchline ("there is generally no
  guarantee of frequentist coverage pointwise at a given θ₀") is
  delivered as a one-clause aside in the bullet list. The astronomer
  reader is asked to internalize that pointwise frequentist coverage is
  the desideratum *and* that Bayesian posteriors generically lack it,
  in one sentence. *Fix:* add a half-sentence concrete illustration —
  e.g. "even a 'flat' prior on θ generally fails this; the coverage of
  Bayesian credible sets is a function of the prior, not of the model
  alone."
- **Lines 242–246 ("In regular one-parameter exponential families with
  monotone likelihood ratio in a continuous sufficient statistic, the
  optimal (uniformly most powerful unbiased, "UMPU") CD is explicit
  and unique (the precise hypotheses are stated in §5.1; the
  location-normal model is the simplest case).")** This sentence
  bundles five technical conditions, names the optimal object UMPU,
  and offers no operational sense of *what is being optimized over*.
  The reader is asked to believe there is a uniqueness theorem and
  to wait until §5.1 to see it. *Fix:* either omit the umbrella
  sentence and let §1.2 stand on the location-normal example alone, or
  add one sentence: "Optimality here means: among CDs whose level-α
  sets are confidence intervals with exact coverage, this CD is the
  one whose intervals are shortest in expectation (the dual statement
  of the UMP unbiased test)." Round-1 already flagged C-1.2-LocNorm in
  this spirit; the round-3 fix is a stronger version of that one.
- **Lines 254–256 ("Our aim is to compute such a CD from a simulator
  alone, when no closed form is available.")** Clean as a closer, but
  the reader has been shown a *closed-form* CD and is now told the
  paper will compute one *without* a closed form. No bridge: how does
  one even formulate "compute a CD" as an optimization problem? *Fix:*
  one sentence saying "the next sections introduce a function class
  rich enough to contain the closed-form CD when it exists, and a loss
  whose minimizers are the calibration manifold within that class."
- **Lines 287–293 ("(C2) Monotonicity in θ … the autoregressive form
  is what makes the joint Φ_d(r(·;X)) a Knothe–Rosenblatt-type
  transport with connected level sets; coordinate-wise monotonicity
  alone (with all other θ_j held fixed) is insufficient in d > 1.")**
  KR transport is invoked by name in §1.3 to motivate (C2), but the
  reader has no operational sense of (a) what a KR transport is, (b)
  why connected level sets matter, or (c) what would go wrong with
  coordinate-wise monotonicity. The "is insufficient" is asserted, no
  counterexample. *Fix:* either defer the d > 1 statement of (C2) to
  Part IV and state (C2) for d = 1 here, or add a one-paragraph
  preview of KR with a 2D pictorial intuition (an L-shaped
  rearrangement that conditions coordinate 2 on coordinate 1).
- **Lines 326–335 ("(C1) is a frequentist property … in the population
  limit, provided the architectural class F contains the calibration
  manifold and ρ has full support, the CD pivot does not depend on ρ.
  Under misspecification or with an under-parameterized architecture,
  the population minimizer of NF-MLE is the ρ-weighted KL projection
  onto F and does depend on ρ; see §11.4.")** This is the most
  pivotal paragraph in §1.3 and reads as the densest. It asserts
  ρ-independence, then immediately qualifies it with two conditions
  the reader has no way to check, then names a "weighted KL
  projection" that the reader has no way to interpret. Round 1 already
  flagged C-1.3-rho in a similar vein. *Fix:* this paragraph should be
  unpacked over three short paragraphs — (a) the population statement
  with both conditions made operational; (b) what goes wrong without
  saturation; (c) a forward pointer with one sentence summarizing the
  §11.4 punchline. As written it does double-duty as both a claim and
  a hedge, and the hedge swallows the claim.
- **Lines 459–478 (§2.4 in its entirety)** The section title is "why
  monotonicity in X is needed," but the content is "why the
  change-of-variables formula needs injectivity in X." The reader is
  told that without injectivity the loss has no lower bound, but the
  *connection between injectivity and monotonicity* is never made
  explicit — in 1D injectivity-of-a-continuous-function and
  monotonicity are equivalent, but in `d > 1` "monotonicity in X" is
  not even defined coordinate-wise (X is data, not parameter, and is
  multivariate). The §2.4 title is misleading. The text talks about
  injectivity throughout but is labelled "monotonicity." *Fix:* either
  rename §2.4 to "the role of (R2): why injectivity in X is needed,"
  or explicitly justify the title (e.g. by noting that in the
  autoregressive class of §6, (R2) reduces to monotonicity in each
  `X_k` coordinate at fixed `X_{<k}`, θ).
- **Lines 469–478 ("…for architectures that admit unbounded
  `|det ∂r/∂X|` in folded regions — e.g. ReLU / UMNN networks with no
  built-in monotonicity — the NF-MLE loss has no lower bound at all.
  The formal version is in §3.5; the empirical demonstration in §8.4
  (where the autograd-Jacobian ablation attains loss 0.56 against the
  true model's 0.88). Enforcing (R2) architecturally fixes this.")**
  The empirical numbers (0.56 vs 0.88) are dropped without
  interpretation. Is 0.56 *better* or *worse* than 0.88? Lower is
  better for NLL, so 0.56 is "better" — but that is the *symptom* of
  the unboundedness pathology, not the diagnostic. The reader has to
  reverse-engineer the sign convention to read the numbers right.
  *Fix:* add the half-sentence "lower is better for this loss, so the
  pathological model appears to beat the truth — which is the symptom
  of the loss being unbounded below."

### Failure mode 3: Proof-role / definition-role clarity

Part I states definitions, not theorems; the question is whether the
*role* of each definition or axiom is clear before the formal statement.

- **The pivot definition (§2.2, lines 377–404)**: the term "pivot" is
  introduced *as* the (R1)+(R2) object, but as round-1 D-pivot flagged,
  the conventional meaning of "pivot" includes the calibration
  property. A reader who has seen "pivot" before will be confused about
  whether the §2.2 object is calibrated by definition. The text needs
  one sentence at line 379 saying "we use 'pivot' for the (R1)+(R2)
  admissible candidate; the calibrated subset is the calibration
  manifold below." Currently the reader has to infer this from the set-
  builder definition of `M_F`.
- **(R1) at line 385**: stated formally with the d > 1 autoregressive
  version inlined, but the role of (R1) is not given *before* the
  statement. §2.3 then gives the role *after* the statement. Standard
  pedagogical practice is one sentence before the formal axiom: "(R1)
  is the axiom that pins down the orientation of the CD and makes
  `H_r(·; X)` a proper CDF on Θ — see §2.3 for the full motivation."
- **(R2) at line 396**: same problem. (R2) is stated formally as a
  diffeomorphism condition, then §2.4 motivates it. The motivation
  should at least be previewed in one sentence at line 397.
- **The calibration manifold (§2.2, lines 408–412)**: defined cleanly
  with one line of set-builder notation. But the role — that this is
  the *target* the loss will drive toward, that uniqueness within an
  architectural class is the goal of Parts III/IV, etc. — is not
  stated. The reader is told only "any element of `M_F` satisfying (R1)
  is a valid CD pivot," which is a *consequence* not a *role*. *Fix:*
  one bracketing sentence: "The rest of the paper aims to (i) write a
  loss whose population minimizers are `M_F` exactly; (ii) restrict F
  so that `M_F` is a singleton."
- **The (C1)/(C2) characterization (§1.3, lines 274–293)**: the
  (C1)/(C2) decomposition is excellent — round-1 already flagged this
  as a strength — but the *relationship* between (C1)/(C2) (top-line
  goal in §1.3) and (R1)/(R2) (structural axioms in §2.2) is not made
  explicit. They are not the same thing: (C1) is the calibration
  desideratum (the *output property*), while (R1)/(R2) are
  architectural axioms on the *input class*. The reader has to infer
  that (R1) ≈ (C2) (modulo the autoregressive subtlety) and (R2) is a
  *new* condition that (C1)/(C2) don't capture. *Fix:* a small table
  or paragraph in §2.2 mapping (C1) ↔ calibration manifold membership,
  (C2) ↔ (R1), and explaining where (R2) comes from (the
  change-of-variables formula, not the desideratum).

### Failure mode 4: Concept-introduction gaps

- **The astronomer-friendly tone (§1.2)** is set and then dropped.
  §1.2 has the "Astronomer-friendly comparison" subhead at line 220,
  which suggests the tone will continue. But §1.3 onward shifts into
  formal definition-theorem mode without bridging back. The reader
  who was sold on the framing in §1.2 has to re-acclimate at §1.3 with
  no narrative cue. *Fix:* one sentence at the start of §1.3
  ("Section 1.2 said *what* a CD is and *why* it is the right object;
  this section says precisely *which* CD we will compute and *what
  conditions* it has to satisfy.") and one at the start of §2 (similar
  bridge from §1.3's (C1)/(C2) goal to §2's (R1)/(R2) axioms).
- **NF-MLE is named before it is defined** (lines 213, 326, 333, 477)
  but is not defined until §3.1 line 497. For a Part-I-only reader
  this is a void; for a sequential reader it is a series of
  forward-pointers that read as "trust me." *Fix:* one
  parenthetical at line 213 — "(NF-MLE = maximum-likelihood training
  of the normalizing flow `r`; see §3.1)" — would carry every
  subsequent use in Part I.
- **The "proposal ρ vs prior" distinction** (line 326 and lines
  359–366) is the most important conceptual claim distinguishing
  CD-SBI from Bayesian SBI. It is made twice — once in §1.3 (informal,
  with the saturation caveat) and once in §2.1 (notation list,
  one-clause aside). Neither place gives it room. *Fix:* a dedicated
  one-paragraph mini-subsection ("Why ρ is not a prior") with the
  full unpacking — currently the claim is spread thin across two
  locations and a forward pointer.
- **The autoregressive structure** (lines 287–293, 387–392, 452–454)
  is named four times in Part I before §6 defines it. Each invocation
  reads as "see §6 for what this means." A 3-sentence preview in §1.3
  or §2.2 — what autoregressive means, why the ordering matters, what
  goes wrong without it — would replace the four floating references
  with one anchor.
- **Confidence sets / confidence regions** at lines 432–438: stated
  as `H_r(·; X)^{-1}([α/2, 1−α/2])` but the geometric content is
  abstract. The full confidence-set formulation in terms of `‖r‖²` is
  not given until Part III/IV. For a reader trying to picture what
  the *output* of a CD-SBI model looks like at inference time, Part I
  is silent. *Fix:* one display showing
  `C_α(X_obs) = {θ : ‖r(θ; X_obs)‖² ≤ χ²_{d, α}}` with one sentence
  saying this is the operational form used in practice.
- **The location-normal example reappears three times** (lines
  246–253, 414–422, then implicitly via `r* = θ − X`) without explicit
  callback. The third appearance at §2.2 says "the function
  `r*(θ, X) = θ − X` is a pivot in this sense" without noting "this
  is the same `r*` we saw in §1.2." A small narrative thread —
  "returning to the location-normal example from §1.2" — would tie
  these together.

### Failure mode 5: Narrative flow

- **§1.1 → §1.2 transition (line 191 → 193).** Works moderately well:
  the §1.1 indictment ("treats calibration as a correction") sets up
  §1.2 ("change the primary target"). But the *logical leap* — that
  changing the target rather than the correction is the right
  response — is asserted, not argued. (Same issue as oracle-passage 1
  above.) The transition is rhetorical, not deductive.
- **§1.2 → §1.3 transition (line 256 → 258).** The bridge is "compute
  such a CD from a simulator alone." Then §1.3 jumps straight to the
  formal (C1)/(C2) decomposition with no explanation of how the
  reader got from §1.2's verbal definition of a CD to (C1)/(C2). A
  one-sentence bridge ("To make 'compute a CD' precise, we name two
  properties the output has to have:") would smooth this. The
  astronomer-friendly tone of §1.2 is dropped abruptly.
- **§1.3 → §1.4 transition (line 335 → 337).** §1.4 is a one-paragraph
  scope statement, which is fine, but it interrupts what could be a
  more natural flow into §2. §1.4 could be folded into §1.3's closing
  ("Subordinate objectives") table caption, or moved to a preface.
  Currently it reads as administrative.
- **§1.4 → §2 transition (line 345 → 346).** A horizontal rule and a
  new section title. The reader has just been told the document's
  scope, and is now thrown into "Notation and standing assumptions"
  with no bridge from the (C1)/(C2) goal of §1.3 to the (R1)/(R2)
  axioms of §2.2. The reader has to infer that (R1) is the structural
  realization of (C2) and that (R2) is a *new* axiom that wasn't in
  the (C1)/(C2) goal. *Fix:* one paragraph at the start of §2: "§1.3
  stated the target as two properties (C1)/(C2) of the output. To
  realize that target, we impose two structural axioms (R1)/(R2) on
  the function class. (R1) is the architectural realization of (C2);
  (R2) is the additional condition needed for the change-of-variables
  formula to be well-posed. The rest of §2 motivates each axiom in
  turn."
- **§2.1 → §2.2 transition (line 376 → 377).** Notation list, then
  formal definitions. No bridge. The notation list buries the "ρ is
  not a prior" claim in a single inline clause (line 362) — that
  claim should be its own paragraph, as flagged above.
- **§2.2 → §2.3 transition (line 422 → 424).** "The framework we
  develop is: find the analogue of r* when no closed form is
  available, by training a neural surrogate." Then §2.3 dives into
  "(R1) ensures `H_r(·; X)` is a proper CDF on Θ." There is no
  signpost that §2.3 and §2.4 are the *motivation* for the two axioms
  that were just stated in §2.2. *Fix:* one sentence at the start of
  §2.3 — "Sections 2.3 and 2.4 motivate (R1) and (R2) in turn: why
  each axiom is needed and what fails without it." Currently the
  reader has to infer the structure.
- **§2.3 → §2.4 transition (line 455 → 457).** Direct title-to-title
  with no narrative connection between the two roles. §2.3 ends on an
  open-problem forward-pointer (§11.1, the rotational ambiguity); §2.4
  opens fresh. *Fix:* a half-sentence bridging the two axioms — "(R1)
  fixes which calibrated `r` we get; (R2) is what makes the loss that
  selects calibrated `r`s well-defined in the first place."
- **§2.4 → Part II transition (line 478 → 482).** §2.4 ends on an
  empirical reference (§8.4 ablation, 0.56 vs 0.88) and then the
  horizontal rule and Part II title. No closing paragraph for Part I.
  *Fix:* a 3–4 sentence Part I summary closing — "Part I has set the
  target: produce a pivot satisfying (C1)/(C2), realized
  architecturally as (R1)/(R2). Part II writes the loss whose
  population minimizers are the calibration manifold; Parts III/IV
  prove that within the right architectural class, that minimizer is
  unique."

## Prioritized recommendations

In rough order of leverage (highest impact first):

1. **Add a §2.0 bridge paragraph** that maps (C1)/(C2) (the
   *desideratum*) to (R1)/(R2) (the *architectural axioms*), and
   says explicitly that (R2) is a new condition that (C1)/(C2) do
   not capture. This single addition would resolve much of failure
   modes 3 and 5, because the reader currently has to derive the
   relationship by themselves. (~half-page insertion.)
2. **Add a one-paragraph "ρ is not a prior" mini-subsection** as
   its own subsection (call it §1.3.5 or §2.1.5), unpacking the
   ρ-independence claim, the saturation/full-support qualifier, and
   the misspecified case in plain terms. Currently the most
   load-bearing conceptual distinction in the paper is buried
   across two locations. This is the single biggest
   accessibility/oracle-style win.
3. **Add a brief Knothe–Rosenblatt + autoregressive preview**
   (3–5 sentences in §1.3 or §2.2), so the four forward-pointers to
   §6 stop reading as "see later." Include the 2D pictorial
   intuition (transport coord 1, then coord 2 conditional on coord 1).
4. **Rename or rewrite §2.4** so the title-content mismatch is
   resolved. Either retitle to "The role of (R2): why invertibility
   in X is needed" (cleanest) or insert one paragraph at the top
   explaining the equivalence of injectivity and monotonicity in 1D
   and the autoregressive coordinate-wise reading in d > 1.
   Currently the title promises monotonicity, the body delivers
   injectivity, and the reader has no anchor for the difference.
5. **Add a brief "what does inference look like" preview** showing
   the confidence-set formula
   `C_α(X_obs) = {θ : ‖r(θ; X_obs)‖² ≤ χ²_{d, α}}` somewhere in §1.3
   or §2.2. Currently a Part-I reader has no concrete picture of the
   *output* of a CD-SBI model. One display equation closes the gap.
6. **Audit forward-pointer density.** Specific worst offenders:
   §1.2 lines 213–214 (forward to (R1) and NF-MLE without
   definition); §1.3 lines 285–293 (forward to §6 KR autoregressive);
   §2.4 lines 469–478 (forward to §3.5 *and* §8.4 in one paragraph,
   with the empirical 0.56 vs 0.88 unmotivated). Each can be
   resolved with a one-clause inline gloss; the cumulative effect on
   readability is substantial.
7. **Glossing pass on technical vocabulary at first use.** Specific
   terms identified in failure-mode 1 above: MLR, sufficient
   statistic, UMPU (with a one-clause "optimal means..."),
   diffeomorphism, orthogonal group `O(d)`, change-of-variables
   determinant. Each gets a parenthetical, not a footnote — total
   added length probably ~1/3 page.

## Existing strengths to preserve

- **The (C1)/(C2) characterization in §1.3 (lines 274–293).** Crisp,
  load-bearing, repeatedly referenced — round-1 already endorsed
  this as the cleanest part of Part I. Any restructuring should keep
  this as the anchor.
- **The "Astronomer-friendly comparison" framing in §1.2 (lines
  220–240).** The Bayesian-posterior contrast is the right hook for
  the target audience and works as written. The fix is to *extend*
  this tone into §1.3 and §2, not to disturb it.
- **The location-normal worked example (lines 246–253).** Concrete,
  self-contained, demonstrates that the framework reduces to the
  textbook in the simplest case. The recurrence at §2.2 (lines
  414–422) is well-placed and should be preserved (just signposted
  as a callback).
- **The "subordinate objectives" forward-pointer table in §1.3
  (lines 296–324).** Gives the reader a road map of which sections
  discharge which obligations — unusually explicit. Astronomer
  readers in particular benefit from this kind of structural
  scaffolding.
- **The "(C1) is a frequentist property" punchline at line 326.**
  The *sentence* is right; the surrounding paragraph just needs
  unpacking (see recommendation 2).
- **The empirical anchor in §2.4 (the §8.4 0.56 vs 0.88 ablation,
  lines 477–478).** The fact that the abstract claim about
  unbounded NF-MLE loss has an empirical demonstration is exactly
  what an astronomer reader wants to see. The numbers just need
  interpretation (see oracle-passage 8 above).
- **Schweder–Hjort grounding (lines 207–214).** Anchoring the CD
  definition in the textbook synthesis is the right move for
  credibility with the statistical reader; the citation chain
  Fisher → Cox → Efron → Singh → Xie–Singh → Schweder–Hjort is
  appropriate and complete.
- **The careful distinction between "C¹ + full-rank Jacobian" and
  "Lipschitz + uniform lower bound" (lines 396–404).** This is a
  technical subtlety that round-1 reviews praised, and it should
  remain. The fix is to flag *that* it is a technical subtlety the
  reader can defer to §4.4, not to remove it.
