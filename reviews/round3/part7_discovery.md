# Round 3 — Part VII (Open questions) Discovery & Assessment

Scope: §11.1–§11.8 (lines 2223–2363) and the closing "Status of claims"
table (lines 2364–2417) of `cd_sbi_v7.tex`.

## Phase A: Discovery

### Top-line goal(s)

Part VII has two top-line goals: (i) enumerate the questions that are
genuinely open in the CD-SBI framework as of v7 — separating "open
because no one has tried it" from "open because the obstruction is
unresolved" — and (ii) honestly mark which load-bearing claims in the
manuscript are theorems vs. conjectures vs. engineering aspirations,
so readers can calibrate trust in the rest of the document. The
"Status of claims" table is the second goal's deliverable.

### Central arguments (one per open problem)

Each subsection is its own micro-argument. The shared shape is "We have
result X (with cross-reference); the natural extension Y is missing;
here is the obstruction." Read as a set:

- **§11.1 Uniqueness beyond autoregressive triangular.** Theorem A-d
  pins down uniqueness *within* the autoregressive class via the
  Knothe–Rosenblatt singleton; outside that class, `N(0, I_d)` is
  `O(d)`-invariant and (R1)-monotonicity-in-θ only kills the
  coordinate-wise sign-flip subgroup `Z_2^d`. The residual symmetry
  group of `M` under weaker (non-autoregressive) monotonicity
  constraints is unknown. Practical takeaway: pick an ordering.
- **§11.2 KR ordering selection.** Under autoregressive (R1), each
  permutation of coordinates is a valid pivot; all live in `M`. Whether
  some orderings train more stably / give lower-variance gradients is
  unstudied (in SBI or in NF more broadly). The Carlier et al. (2010)
  anisotropic-quadratic continuation from KR to Brenier maps motivates
  a conjecture but does not resolve it.
- **§11.3 The C⁰ residual in 1D.** Lemma 4.2's `C¹` argument derives
  (R3) (strict X-monotonicity, a.e.) from (R1) + calibration via a
  pushforward-density blow-up; the blow-up fails in the Lipschitz case
  (V-shape gives a bounded density jump). So Theorem A\* needs (R3) as
  a separate hypothesis. Whether (R3) is implied by (R1) + Lipschitz +
  calibration alone — without `|∂_X r| ≥ c > 0` — is the precise open
  question. The text notes that this is moot in practice because UMNN /
  triangular architectures enforce (R3) by construction.
- **§11.4 Misspecification.** Under model misspecification, the
  population NF-MLE minimizer is the forward-KL ("M-")projection of the
  true conditional `p(·|θ)` onto the architectural class `F`, and
  `r(θ₀; X) | θ₀ ~ N(0, I_d)` no longer holds. The coverage degradation
  of this projection is unanalyzed. Existing robust-SBI ideas (RoPE,
  generalized/power posteriors, MMD posterior bootstrap) target related
  questions via *different* mechanisms — RoPE uses real-data calibration
  pairs; Schmon et al. reweight ABC; Dellaporta et al. resample via MMD.
  None of these is an analysis of the forward-KL projection itself.
- **§11.5 Higher-dimensional scaling.** Autoregressive flows scale
  linearly in `d` for the Jacobian, and the broader NF literature
  reaches `d ~ 100`+ (Papamakarios, Durkan, Lueckmann). What is specific
  to CD-SBI is that the *triple* {calibration, monotonicity, tractable
  Jacobian} must be maintained jointly. Whether this triple stays
  tractable past the 2D experiments, and whether sparse-autoregressive
  / attention conditioning becomes necessary, is open. Empirical
  validation past `d = 2` is the natural next milestone.
- **§11.6 Sequential / amortized variants.** Two sub-questions, framed
  separately and very deliberately: (a) Naive sequential CD-SBI
  (adaptive `ρ` near `X_obs`) almost certainly loses pointwise
  calibration — Hermans et al. 2022 already documented empirical
  miscalibration of all naive sequential SBI, and the `ρ`-independence
  argument of §1.3 breaks. (b) Whether an importance-reweighted or
  APT-style corrected sequential procedure can restore the guarantee
  is genuinely open. Falkiewicz et al. 2023 (Calibrated NPE) is the
  closest existing work; they add a coverage regularizer without a
  proof.
- **§11.7 Discrete data without a scalar sufficient statistic.** The
  §5.7 randomized-PIT construction relies on a fixed scalar MLR
  sufficient `T(X)`; under a learned feature `g_φ(X)`, sufficiency
  fails, atom probabilities `p_θ(t)` become φ-dependent, and (R4)'s
  order-coherence is no longer architectural because the ordering of
  `T`-values is itself learned. Open whether the Lancaster randomization
  story extends.
- **§11.8 Efficient computation of C_α in d ≥ 3.** Explicitly labeled
  "implementation challenge, not a research problem." Three standard
  approaches (autoregressive root-finding, HMC/SMC/rejection,
  constrained pushforward) all work; what is undeveloped is efficient
  visualization / reporting in `d ≥ 3`. This is correctly downgraded
  from open problem to engineering note (it was flagged "✗ not actually
  open" in round 1 and the heading now reflects that).

### Supporting evidence per problem

Each open problem now backs its "is open" claim with one or more of:
forward references to the relevant earlier section that *did* resolve a
neighboring case (§11.1 → §6.3; §11.3 → §4.4; §11.7 → §5.7); citations
to literature that addressed an adjacent question without resolving the
one at issue (§11.2 → Carlier et al.; §11.4 → Wehenkel et al., Schmon
et al., Dellaporta et al.; §11.5 → Lueckmann, Papamakarios, Durkan;
§11.6 → Hermans, Lueckmann, Falkiewicz); and an explicit "to our
knowledge" disclaimer where the literature search came up dry (§11.2,
§11.4, §11.6). After round 2's citation pass, every open problem
except §11.3 (which is internal-only by design) carries at least one
external anchor.

### Coherent set vs. unconnected list?

Round 1 and round 2 substantially rewrote several open problems (§11.1
fixed the rotation-symmetry-vs-(R1) conflation; §11.3 corrected the
Lemma 4.2-vs-4.3 cross-reference; §11.4 dropped the wrong
"moment-projection" term and added the RoPE / generalized-posteriors
context; §11.5 dropped the unjustified `d ≳ 10` cutoff and added
NF-benchmark scaling references; §11.6 split into naive vs. corrected;
§11.8 was downgraded from "open" to "implementation"). The set is now
fairly coherent as a *list of independent questions*, but it is not yet
organized as a *roadmap*. Reading §11.1 → §11.8 linearly, the implicit
order is: §11.1–§11.3 are theoretical loose ends (uniqueness extensions,
regularity); §11.4 is robustness; §11.5 is scaling; §11.6 is amortized
vs. sequential; §11.7 is a different model class (discrete +
high-dim observations); §11.8 is engineering. That ordering is
defensible but not signposted, and there are no headings to group them.

### The "Status of claims" table

The closing table maps each "Theorem / Empirical / Open" claim in the
manuscript to its location and a one-line status. It functions as a
machine-readable summary of which load-bearing pieces of the framework
are proven, which are validated only empirically, and which are open.
For a reader who wants to know what they can cite vs. what is
aspirational, this is genuinely useful — and the inclusion of the
"Open" rows linking to §11 ties the table to Part VII directly. It is
not a leftover artifact.

## Phase B: Accessibility assessment

### 1. Undefined terms for non-specialists

The astronomer/non-specialist reader hits the following terms:

- **"KR ordering"** (§11.1, §11.2): the body of §11.2 implicitly defines
  "ordering" as the autoregressive order of coordinates `θ_1, ..., θ_d`,
  but a non-specialist coming straight from the executive summary might
  not know that "KR" = Knothe–Rosenblatt and that the ordering choice
  is what selects a particular triangular structure. Definition is
  inherited from §6.3; no reminder here.
- **(R3), C⁰ residual, Lipschitz case** (§11.3): "C⁰ residual" in the
  heading is jargon — it refers to the gap between the C¹ result
  (Theorem A) and the Lipschitz result (Theorem A\*). The body explains
  this in passing but uses "the Lipschitz case" as a callout without
  reminding the reader what fails. A one-line "in the Lipschitz case,
  Lemma 4.2's pushforward-density blow-up argument fails" is present
  but compressed.
- **Forward-KL projection / M-projection** (§11.4): the body defines
  the forward-KL projection with a formula, then parenthetically calls
  it "the M-projection in the modern Bregman-geometry sense". For an
  astronomer this terminology cascade (forward-KL / M-projection /
  Bregman geometry) is dense; the formula carries the load but the
  surrounding language is dense.
- **RoPE** (§11.4): named with one inline definition ("an
  optimal-transport calibration method that uses a small real-world
  calibration set of (θ, X) pairs"). Sufficient for someone who knows
  OT calibration; thin for someone who doesn't.
- **Generalized-Bayes posterior / power posterior** (§11.4): mentioned
  in passing ("generalized posteriors / power-posterior reweighting in
  the ABC setting"). Not defined.
- **MMD posterior bootstrap** (§11.4): named with no definition.
- **Amortized vs. sequential SBI** (§11.6): defined inline at the top
  of the subsection ("single network for all X_obs" vs. "refine around
  a specific X_obs"). Good — this is the one place the manuscript does
  the right thing for the non-specialist.
- **Lancaster randomization** (§11.7): named but not re-explained;
  refers back to §5.7 (Theorem C\*). A reader who skimmed §5.7 will
  not remember the construction by §11.7.
- **Anisotropic-quadratic-cost continuation** (§11.2): jargon-dense
  phrase from the OT literature. The parenthetical "taking the cost
  weights λ_1 ≫ ... ≫ λ_d as the limit recovers KR" is helpful but
  assumes the reader knows what a quadratic transport cost is.
- **APT-style** (§11.6): named without expansion (APT = Automatic
  Posterior Transformation, Greenberg et al. 2019). A non-SBI reader
  will not recognize this.

### 2. Oracle-style passages

Open-problem paragraphs are inherently terse, and §11 paragraphs are
all compact. The "oracle" failure mode here is each problem being
described in 3–5 sentences that pack "what's open, why, what's known
nearby" into one block. By subsection:

- **§11.1**: motivates well — explicit reference to Theorem A-d, then
  states which subgroup of `O(d)` is killed by (R1) and which is not.
  The practical takeaway sentence at the end ("some triangular ordering
  must be chosen architecturally") is a strong signal of "why this
  matters." Good.
- **§11.2**: motivates the question (no natural ordering for unordered
  data, all orderings give valid pivots) before posing it. The OT
  paragraph is the densest part — a reader unfamiliar with KR /
  Brenier / OT cost will skim it. The "we are not aware of any
  published study" sentence is helpful framing for an astronomer who
  might assume the answer must exist somewhere.
- **§11.3**: heavily compressed. "(R3) is automatic" requires reading
  §4.4; the open question itself is a single sentence; the failure
  mode (the Lipschitz blow-up) is a half-sentence parenthetical. Hardest
  subsection to read without §4 in working memory.
- **§11.4**: long enough to be readable, but the term-density per
  sentence is high (forward-KL projection, M-projection, RoPE,
  generalized posteriors, MMD posterior bootstrap, all introduced
  within one paragraph). A reader who pauses on each term will spend
  more time on §11.4 than on the rest of §11 combined.
- **§11.5**: well-motivated. Concrete numbers (`d_θ ~ 10`, `d_X ~ 100`)
  ground the reader; the "calibration–monotonicity–Jacobian triple"
  phrasing names the specific thing CD-SBI adds. Good.
- **§11.6**: well-structured (the explicit split into naive vs.
  corrected sub-questions is a model of how to write an open-problems
  paragraph). The §1.3 reference for "ρ-independence" is correct and
  load-bearing; a reader who didn't internalize §1.3 will be lost on
  why naive sequential breaks. APT is unexplained.
- **§11.7**: the obstruction description (sufficient `T` becomes
  feature-extractor-dependent, (R4) loses architectural cleanness) is
  precise but assumes the reader remembers (R4) from §5.7. For a
  reader who is in §11 because §11 was advertised in the executive
  summary as the open-problems map, the references back to §5.7 are a
  significant context demand.
- **§11.8**: the explicit "implementation challenge, not a research
  problem" label and the enumerated list of three standard approaches
  make this the most reader-friendly subsection.

### 3. Proof-role clarity (analog for open problems)

The "we know X; we don't know Y; here's why Y matters" pattern is
followed cleanly in §11.1, §11.5, §11.6, §11.8. It is followed in
compressed form in §11.2, §11.4, §11.7. It is followed but obscured
in §11.3 (the "why it matters" is "moot in practice" — which is true,
but undercuts the motivation). Overall the framing is consistent
enough to be skimmable.

### 4. Concept-introduction gaps

The most acute back-references that may strand a linearly-skimming
reader:

- §11.1 → "Theorem A-d (§6.3)" and "rotation symmetry of the standard
  Gaussian"; the reader needs §6.3 (Knothe–Rosenblatt singleton) and
  §3.3 (rotation symmetry vs. calibration) in working memory.
- §11.3 → "(R3) … Lemma 4.2"; the reader needs §4.4 + Lemma 4.2.
- §11.4 → "the forward-KL projection"; the reader needs the connection
  between NF-MLE and KL that lives in §3.2 / equation (515) of §3.
- §11.6 → "ρ-independence-in-the-population-limit argument of §1.3";
  the reader needs §1.3.
- §11.7 → "the discrete extension of §5.7" + "(R4) order-preservation";
  the reader needs §5.7 and the (R4) definition (line 1129).

These back-references are correctly cross-cited, but a reader who is
treating §11 as a standalone "roadmap" section will find them
demanding. A one-line in-place reminder per subsection (e.g., "(R3) is
the assumption that `|∂_X r|` is bounded away from zero a.e.") would
reduce the context demand sharply without changing the substance.

### 5. Narrative flow

The eight subsections are currently presented as a flat list. The
implicit grouping is:

- **Theoretical extensions** (§11.1 uniqueness outside autoregressive;
  §11.2 ordering selection; §11.3 C⁰ residual)
- **Robustness** (§11.4 misspecification)
- **Scaling and practice** (§11.5 higher d; §11.6 sequential; §11.7
  discrete with no scalar sufficient statistic)
- **Engineering** (§11.8 efficient C_α)

This grouping is implicit but not signposted. Three sub-headings would
let a reader scanning for "is the misspecification question addressed?"
or "what about high dimensions?" land in the right block without
reading all eight headings. The order within each group is defensible;
the §11.5 → §11.6 → §11.7 sequence in particular (scale up in `d`,
then in iteration structure, then in observation type) reads naturally.

The "Status of claims" table at the end serves readers well: it
provides a final at-a-glance view of what is proven (theorems +
empirics) vs. what is open, and the §11 cross-references close the
loop. It is not a leftover. A small accessibility improvement: the
table mixes "Proven" / "Validated" / "Open" in the Status column —
splitting it into Theory vs. Empirical vs. Open sub-blocks (or color-
or symbol-coding the rows) would make it easier to scan. Three of the
"Open" rows (R3 / higher-d / sequential) are also explicitly
"in-practice-resolved or-near-resolved" in the body of §11; the table's
flat "Open" label slightly oversimplifies these. A more precise label
(e.g. "Open (architecturally moot)" for §11.3) would communicate the
nuance.

## Prioritized recommendations

1. **Add three sub-headings to §11** to group the eight open problems
   into "Theoretical extensions" (§11.1–§11.3), "Robustness" (§11.4),
   "Scaling and practical extensions" (§11.5–§11.7), "Engineering"
   (§11.8). This converts the flat list into a roadmap with negligible
   re-writing.
2. **One-line in-place reminders for §11.3, §11.4, §11.6, §11.7**
   summarizing the load-bearing prior concept (`(R3)`, forward-KL /
   M-projection, ρ-independence argument, Lancaster randomization).
   This is the single highest-leverage accessibility intervention for
   Part VII — currently these subsections are unreadable without
   §4.4 / §3 / §1.3 / §5.7 in working memory.
3. **Soften the terminology load in §11.4**. Define (or briefly gloss)
   RoPE, generalized-Bayes / power posterior, and MMD posterior
   bootstrap on first use. A footnote-style one-sentence definition
   per term would suffice; alternatively, drop the SchmonEtAl /
   DellaportaEtAl mentions if they are not load-bearing for the
   open-problem framing.
4. **Refine the "Status of claims" table**: split or annotate the
   "Open" column to distinguish "open and architecturally moot" (§11.3)
   from "open and the natural next milestone" (§11.5, §11.6) from
   "open and unaddressed" (§11.4). A second sub-status column or a
   parenthetical in the Notes column suffices.
5. **Add a one-sentence opening to §11** stating what the section is
   for ("this section enumerates the questions left open by the
   framework, grouped by …") and reminding the reader that the
   "Status of claims" table at the end gives the cross-reference index.
   Currently §11 begins with §11.1 cold.
6. **§11.6 small fix**: expand "APT-style" to "APT-style
   (Automatic Posterior Transformation, Greenberg et al. 2019)"
   inline, since this is the only place APT is mentioned and it is
   load-bearing for the open question.

## Existing strengths to preserve

- **The two-part structure of §11.6** (naive sequential vs. corrected
  sequential, each with its own evidence and citation) is the cleanest
  open-problem framing in the manuscript and is a template for how
  the others could be sharpened. Keep this.
- **The §11.8 reframing as "implementation challenge, not a research
  problem"** is exactly the right epistemic humility — round 1 flagged
  it as not actually open, and the v7 framing honors that. Keep.
- **The "to our knowledge" disclaimers** in §11.2, §11.4, §11.6 are
  honest and useful — they tell the reader the literature was checked
  and the gap is real. Keep these and consider adding one to §11.7.
- **The "Status of claims" table** is a strong closing artifact for
  this manuscript. Many theory-heavy SBI / CD papers leave the reader
  guessing what is proven vs. conjectured; this table answers that
  directly. Keep, with the refinement in recommendation 4 above.
- **The §11.1 practical takeaway sentence** ("some triangular ordering
  must be chosen architecturally") and the §11.5 "calibration–
  monotonicity–Jacobian triple" phrasing are both examples of the
  manuscript naming the specific thing CD-SBI contributes. Preserve
  and replicate this style in §11.2 and §11.4 where it is currently
  missing.
- **Forward references from §11 back into the body** (§11.1 → §6.3,
  §11.3 → §4.4, §11.6 → §1.3, §11.7 → §5.7) are correct and complete.
  The accessibility issue is not that they are wrong but that they
  require the reader to actually follow them; recommendation 2 (in-place
  reminders) addresses that without removing the cross-references.
