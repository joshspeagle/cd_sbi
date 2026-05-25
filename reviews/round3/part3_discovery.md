# Round 3 — Part III (1D theory) Discovery & Assessment

Reviewer profile: literary + technical reviewer for round 3, focused on
pedagogy and accessibility for a mixed statistician/astronomer
audience. Scope: §§4–5 of `cd_sbi_v7.tex` (lines 794–1266) — the
"longest Part of the manuscript and the most theorem-heavy."

## Phase A: Discovery

### Top-line goal(s)

Part III is the **identification theorem layer** of the manuscript: it
proves that, on a sequence of progressively wider one-dimensional
models (location-normal → regular exponential family → discrete via
randomization), the calibration manifold `M` intersected with a
suitable smoothness class is a singleton, and that the unique element
is the closed-form classical optimal CD. In other words, the goal is
to demonstrate that *in cases where the right answer is known
analytically, NF-MLE + (R1) + (R2) recovers it uniquely* — which is
the foundation for trusting the method when no analytic answer exists.

### Central arguments

1. **In the location-normal model, NF-MLE + (R1) + (R2) + `C¹` has
   exactly one minimizer**, the additive pivot `r*(θ;X) = θ − X`
   (Theorem A, §4.1–§4.3). This is the existence proof that the
   framework's promises (frequentist calibration, uniqueness within a
   smoothness class) are non-vacuous in the simplest setting.

2. **The result extends to Lipschitz `r` if (R3) is added**
   (Theorem A*, §4.4). This matters because ReLU networks — the most
   common practical parameterization — are Lipschitz but not `C¹`. (R3)
   replaces the level-set rigidity argument (Lemma 4.2), which uses
   `C¹` essentially, by an architectural hypothesis.

3. **The unique pivot `r*` in the location-normal case is identically
   the classical Schweder–Hjort UMP-unbiased CD** (Proposition 4.5,
   §4.5). This is the manuscript's "reduces to classical theory in
   regular cases" claim, anchoring the framework against the textbook
   answer.

4. **The same machinery lifts to any regular one-parameter exponential
   family with MLR and continuous sufficient statistic `T(X)`**
   (Theorem C, §5.1–§5.2), where the canonical pivot becomes
   `r*(θ;X) = Φ⁻¹(1 − F_θ(T(X)))`. Worked examples (§5.3) demonstrate
   the construction on Student-`t` (via ancillarity) and exponential
   rates.

5. **For discrete sufficient statistics, calibration to `N(0,1)`
   requires auxiliary randomization**: enlarge the input space with
   `U ~ U(0,1)` and use the randomized PIT (Stevens 1950 / Lancaster
   1961), giving `V_θ(X,U) = F_θ(T(X)⁻) + U · p_θ(T(X))` and
   `r*_rand(θ;X,U) = Φ⁻¹(1 − V_θ(X,U))` (§5.7.1).

6. **Discrete uniqueness requires a fourth regularity condition
   (R4)** that orders the `T`-atoms; with (R1)–(R4) the randomized
   pivot is unique (Theorem C*, §5.7.2). The smallest model where (R4)
   is genuinely independent of (R1)–(R3) is `Bin(3,θ)`, demonstrated
   by an explicit alternative tiling that satisfies (R1)–(R3_U) but
   violates (R4).

7. **A deterministic mid-p alternative trades exact calibration for
   `O(1/√n)` Kolmogorov-distance calibration** (§5.7.3), surfacing the
   practical trade-off for applications where carrying an auxiliary
   `U` is undesirable.

### Theorem map (T-A, T-A*, T-C, T-C*) — what each is for and how they relate

| Theorem | Setting | Smoothness | Extra hyp. | Conclusion | Role |
| --- | --- | --- | --- | --- | --- |
| **T-A** | `X ~ N(θ,1)` | `C¹` in X | (R1), (R2) | `M ∩ F_{C¹} = {r*}`, `r*=θ−X` | The base case; the "right answer is recovered uniquely" proof for the cleanest possible model |
| **T-A*** | `X ~ N(θ,1)` | Lipschitz | (R1), (R2), **(R3)** | `r = θ − X` a.e. | The practical extension; ReLU networks are Lipschitz, not `C¹`, so we need this version to apply T-A to a realistic architecture |
| **T-C** | regular 1-parm exponential family with MLR + continuous `T(X)` | `C¹` in `t = T(X)` | (R1), (R2), MLR | `M ∩ F^T_{C¹} = {r* = Φ⁻¹(1−F_θ(T))}` | The horizontal lift; shows T-A is not special to the location-normal model — it covers all the classical regular cases via sufficiency |
| **T-C*** | discrete `T(X)` with MLR | strict-monotone-in-`U` | (R1), (R2), (R3_U), **(R4)** | unique `r*_rand` in the randomized class | The vertical lift; shows the same uniqueness story survives the move from continuous to discrete data, *if* the input space is enlarged by `U` and a new ordering condition (R4) is added |

The four theorems form a 2×2 grid: T-A / T-A* are the `C¹` / Lipschitz
versions of the *same* result in the *same* model; T-C / T-C* are the
horizontal (exponential family) / vertical (discrete) lifts of T-A.
The manuscript needs all four because:

- T-A alone is too restrictive for practice (excludes ReLU nets) — T-A*
  fixes that.
- T-A by itself is "just a Gaussian result" — T-C shows it's actually
  the regular-1-parameter-EF result in disguise.
- T-C inherits T-A's assumption of a continuous sufficient statistic;
  T-C* extends to discrete data, which is needed for any binomial /
  Poisson / count-data application.

The proof of T-A is the main load-bearing argument: T-A*, T-C, and
T-C* all reduce, at the technical core, to "Lemma 4.1 + a suitable
analogue of Lemma 4.2 + Lemma 4.3 with the right pushforward
hypothesis." This is what makes the three-lemma structure of §4.2 the
central pedagogical object of Part III.

## Phase B: Accessibility assessment

### Failure mode 1: Undefined terms

The Part has *some* glossary work (notably the inline "ancillary
statistic" definition at line 1063 and the "Background, brief"
exponential-family paragraph at line 1002), but coverage is uneven.

| Term | First use | Defined? | Adequacy |
| --- | --- | --- | --- |
| *monotone rearrangement* | L-4.1 proof, line 826 | "Standard 1D monotone rearrangement \citep[ch.~2]{Villani2003}" | **Inadequate for astronomers.** Cites Villani 2003 (graduate optimal-transport monograph). No intuition given. An astronomer reads "between two atomless probability measures on R, the unique monotone-increasing measure-preserving map is the composition of CDF and inverse CDF" and has to either know what *atomless* and *measure-preserving* mean or close the document. |
| *pushforward density / measure* | L-4.2, line 850 | Used without definition. | **Inadequate.** A one-line "the pushforward density of a measure under a map is the density of the transformed random variable, computable by change of variables" would suffice. |
| *Lebesgue-a.e.* | T-A*, line 955 | Used inside the Rademacher invocation, no gloss. | **Inadequate.** "Almost everywhere with respect to Lebesgue measure on R^n; equivalently, the set of exceptions has volume zero" — single sentence would resolve. |
| *Rademacher's theorem* | T-A* proof, line 954 | Glossed inline: "a Lipschitz function on R^n is differentiable Lebesgue-a.e." | **Adequate.** Good model for how other terms should be handled. |
| *bi-Lipschitz* | T-A* proof, line 957 | Undefined. | **Inadequate.** "Both `f` and its inverse are Lipschitz; in particular `f` is invertible with non-vanishing derivative bounded above and below" would close the gap. |
| *level-set rigidity* | L-4.2 title, line 836 | Section title; not defined separately. | **Borderline.** The *content* of the lemma explains what it means, but the *name* is jargon. A one-line "we'll call this Lemma 4.2 'level-set rigidity' because it shows that level sets `{X : ψ(X) = v}` can't proliferate without breaking the pushforward density" would help. |
| *exponential family (canonical / natural form)* | §5.1, line 1003 | Defined: "density takes the canonical form `p(X|θ) = h(X)exp(θT(X) − A(θ))`" | **Adequate.** The "Background, brief" paragraph does its job. |
| *sufficient statistic* | §5.1, line 1006 | Defined inline: "X enters the likelihood only through T(X)" | **Adequate.** |
| *ancillary statistic* | §5.3, line 1063 | Defined inline: "distribution does not depend on the parameter" | **Adequate.** |
| *MLR / monotone likelihood ratio* | §5.1, line 1008 | Defined: "for any θ_1 < θ_2, p(X|θ_2)/p(X|θ_1) is a monotone function of T(X)" | **Adequate.** |
| *UMP-unbiased CD / UMPU CD* | §1.2 (line 244) + §4.5 (line 979) | §4.5 says: "the CD whose associated one-sided test family at each level is uniformly most powerful within the class of unbiased tests" | **Borderline for astronomers.** The §4.5 sentence is dense (three nested classical concepts: UMP, unbiased tests, test family at a level). The astronomer has to know "UMP test," "unbiased test," and "test family at level α" before this sentence parses. A one-paragraph aside in §4.5 ("a UMP test for `θ ≤ θ_0` vs `θ > θ_0` is the most powerful test of that null at every alternative; an unbiased test has power ≥ α at every alternative; UMP-unbiased combines these and is the textbook optimality criterion in regular models") would carry an astronomer through. |
| *randomized PIT* | §5.7.1, line 1103 | Defined by formula `V_θ(X,U) = F_θ(T(X)⁻) + U · p_θ(T(X))`. | **Inadequate on intuition.** The formula is given, but the *why* — why this particular construction produces a `U(0,1)` random variable — is given as one sentence: "smear each atom across an interval of an auxiliary uniform variable." A picture or a 2-3 sentence intuition ("when `T(X) = t` has mass `p_θ(t)`, we'd like to spread that mass uniformly over the CDF jump from `F_θ(t⁻)` to `F_θ(t)`; the auxiliary `U` does exactly that") would let an astronomer build the right mental picture. |
| *lex-order monotone rearrangement* | T-C* proof, line 1147 | Used in passing: "in the order (T,U), the unique tiling..." | **Inadequate.** "Lex order" — lexicographic ordering on the product space — is statistician-natural but astronomer-opaque. One line: "order `(T, U)` pairs first by `T`, then by `U`; this is like sorting `(book chapter, page number)` first by chapter, then within each chapter by page." |
| *propriety / strictly proper objective* | invoked at L-4.3, line 880 | Cross-referenced to §3.2; not redefined locally. | **Borderline.** Acceptable for a self-contained Part III re-reader, but a one-clause reminder ("L-4.3 is the conditional version of the strict-propriety argument of §3.2 — KL is zero iff the conditional densities agree") would help. |

### Failure mode 2: Oracle-style passages

This is the dominant failure mode in Part III. Several theorem statements
and proofs land *in medias res* without the reader being told what the
theorem accomplishes or why the proof has the shape it does.

1. **Theorem A statement (§4.1, lines 803–815).** The Part opens with
   "Statement" and immediately hits the formal class definition and
   the equality `M ∩ F_{C¹} = {r*}`. The reader is *not* told first
   *what the theorem is for* — that this is the foundational
   identification result that anchors the rest of the manuscript by
   showing CD-SBI recovers the classical answer when the answer is
   known. The "equivalently, ... the Schweder–Hjort UMP-unbiased CD"
   sentence at the bottom does serve as a payoff, but it comes after
   the heavy formalism, not before it. **A 2–3 sentence opener that
   says "the goal of this theorem is to verify that on the simplest
   model where the right answer is known classically, NF-MLE + (R1) +
   (R2) produces exactly that answer and nothing else" would orient
   the reader before the formal statement.**

2. **The three lemmas (§4.2, lines 817–899).** The lemmas are
   introduced as a list — Lemma 4.1, Lemma 4.2, Lemma 4.3 — with no
   "here's why three lemmas, and here's the role each plays" preamble.
   The single intervening sentence is "The proof rests on three
   lemmas" (line 815), which is informational but not pedagogical. The
   reader is not told that:
   - L-4.1 (monotone transport) provides the *form* (which two
     candidate maps could a `ψ_{θ_0}` be?).
   - L-4.2 (level-set rigidity) ensures *uniqueness within form*
     (`ψ_{θ_0}` has to be monotone, not just satisfy the pushforward
     property).
   - L-4.3 (NF-MLE conditional propriety) supplies the *hypothesis*
     (the calibration property that lets us invoke L-4.1 and L-4.2 in
     the first place).
   **A 4–6 sentence "roadmap of the proof" before the lemmas would
   reframe the three lemmas as components of an argument rather than
   as a checklist.**

3. **The proof of Theorem A (§4.3, lines 900–929).** The proof reads
   "By Lemma 4.3, `ψ_{θ_0}` pushes ... By Lemma 4.2, `ψ_{θ_0}` is
   strictly monotone. By Lemma 4.1, `ψ_{θ_0}(X) ∈ {X − θ_0, θ_0 − X}`."
   This is the most oracle-style passage in Part III: three lemmas
   applied in sequence with no explanation of why this is the order
   or why this constitutes a uniqueness proof. **A "what's happening
   here" sentence before the chain of invocations — "the strategy is
   to pin down `ψ_{θ_0}` in three steps: first, show that it must
   push the source to the target (L-4.3); second, that it must be
   strictly monotone (L-4.2); third, that the only strictly monotone
   map with that pushforward is one of two candidates (L-4.1); then
   pick the candidate that satisfies (R1)" — would convert the
   passage from a recitation to an argument.**

   The "Selecting the direction" paragraph (lines 916–929) was
   improved in round 1 (the measurable-mixture issue was patched) and
   now reads *better*, but it's still terse. The reader is told the
   conclusion before the argument: "(R1) does not rule out a
   measurable mixture..." starts a counterfactual without telling the
   reader *why* this counterfactual matters or how it's resolved. A
   one-sentence frame ("the previous step gives us two candidates
   *for each θ_0*; we now have to ensure that we pick the same
   candidate uniformly in θ_0") would help.

4. **Theorem A* (§4.4, lines 931–976).** §4.4 *does* motivate the
   theorem reasonably well: the opening paragraph (lines 933–936)
   says "Lemma 4.2 uses C¹ regularity essentially: the contradiction
   comes from a `|v − v_*|^(−1/2)` divergence ... which fails for
   Lipschitz V-shapes. For ReLU networks, we need an alternative."
   This is the *right* model for how a theorem should be motivated.

   The introduction of (R3) immediately after, however, is told not
   shown: "The clean approach is to require strict X-monotonicity as
   an extra condition." The reader is not told why (R3) is *added* as
   an architectural assumption rather than *derived* — i.e., why we
   can't prove (R3) from the other hypotheses in the Lipschitz case.
   **One sentence — "in the C¹ case, Lemma 4.2 *derives* strict
   monotonicity from calibration + smoothness; the Lipschitz case
   lacks this derivation because Lipschitz V-shapes are calibrated
   but not monotone, so we must impose monotonicity directly as
   (R3)" — would close the gap.** The "When is (R3) automatic"
   paragraph (lines 970–975) is good and addresses the practical
   concern; pulling that intuition forward into the statement of
   (R3) would help.

5. **Theorem C (§5.2, lines 1033–1060).** The proof says "The
   three-lemma argument of Theorem A then applies" (line 1042) and
   lists how each lemma transfers, via "(general source remark,
   §4.2)" — this is mechanically correct but reads as boilerplate.
   **The reader is not told why the transfer is possible without
   re-proving anything.** The implicit answer is that L-4.2 was
   already proved in source-density-agnostic form (the round-1
   "general source" remark was added precisely for this purpose) and
   L-4.3 depends only on (R2) + KL — so T-C is genuinely a corollary
   of T-A under sufficiency. A 2-3 sentence framing — "T-C is not a
   re-proof of T-A in new coordinates; it is the application of T-A's
   exact machinery in `t = T(X)`-coordinates, with the source
   `N(θ_0, 1)` replaced by the conditional law of `T(X)` given `θ_0`.
   The three lemmas were stated with this generality in mind" — would
   convert the passage from a recapitulation into a payoff.

6. **Theorem C* (§5.7.2, lines 1116–1245).** This is the longest and
   most intricate proof in Part III, and its *purpose* — not its
   conclusion — is the most easily lost. The reader who comes to
   §5.7.2 without round-2 context sees:
   - the four-regularity setup (R1), (R2), (R3_U), (R4), with the
     parenthetical "distinct from (R3) of §4.4" buried mid-paragraph
     (lines 1119–1124);
   - the theorem statement (one line, line 1138);
   - the proof (lines 1141–1160);
   - then a Bin(3,θ) "explicit counterexample" running 1162–1229.

   The counterexample is essential — it's the round-1 fix for the
   broken Bin(1,θ) construction — but its purpose is *not* stated.
   The reader sees the construction, the verification of (R1)–(R3_U),
   the violation of (R4), and at the end the punchline "(R4) selects
   `V*_θ`" (line 1229). **The reader who hasn't internalized round
   1's audit doesn't know that the counterexample's job is to
   demonstrate that (R4) is *not redundant* with (R1)–(R3_U) — i.e.,
   that we couldn't just drop (R4) from the theorem.** A 2-3 sentence
   frame before the counterexample — "the natural worry about
   Theorem C* is whether (R4) is genuinely needed or whether it
   follows from (R1)–(R3_U). The Bin(3,θ) construction below shows it
   is genuinely needed: an alternative tiling `V'_θ` of `[0,1]`
   satisfies all of (R1), (R2), (R3_U), and produces a calibrated
   pivot, yet differs from the canonical `V*_θ` — only (R4) breaks
   the tie" — would convert the passage from "here's a long
   calculation" to "here's the calculation that justifies one of our
   four hypotheses."

### Failure mode 3: Proof-role clarity (the most important failure mode for Part III)

The user's primary concern. Currently, **none of T-A, T-A*, T-C, T-C*
has a "here's what this gives us" framing before the formal
statement.** Each theorem jumps straight into class definitions and
the equality. The pattern is:

- §4.1: "Theorem A (Uniqueness, C¹). *Let X ~ N(θ,1) on Θ = R. Within
  the class ...*"
- §4.4: "Theorem A* (Uniqueness, Lipschitz). *Suppose r: R² → R is
  Lipschitz ...*"
- §5.2: "Theorem C (Lift to exponential families). *Under the setup
  of §5.1, within the class ...*"
- §5.7.2: "Theorem C* (Discrete). *Under (R1)–(R4), the unique pivot
  in the randomized calibration manifold is `r*_rand`.*"

The names — "Uniqueness", "Strengthening to Lipschitz", "Lift to
exponential families", "Discrete uniqueness" — give the *what* but
not the *why*. The reader is left to infer the role from the names
plus the cumulative position in the manuscript. A more pedagogical
pattern would be:

- **Each theorem statement preceded by a "goal" sentence** that says
  what success looks like and why this case had to be proved. For
  T-A: "Goal: verify that on the simplest model where the right
  answer is known classically, the framework + monotonicity + an
  innocuous smoothness assumption produces that answer and nothing
  else." For T-A*: "Goal: extend Theorem A to architectures that
  aren't `C¹` (in particular ReLU networks) by replacing the
  smoothness-derived monotonicity with an architectural one." For
  T-C: "Goal: show T-A wasn't special to Gaussian data — the same
  argument identifies the canonical pivot for any classical
  exponential family that admits a UMPU CD." For T-C*: "Goal: extend
  T-C to count data, where calibration to a continuous target
  requires auxiliary randomization, and where a fourth condition
  (R4) is required."

- **Each proof preceded by a "strategy" sentence** that names the
  load-bearing step. T-A: "Strategy: apply Lemmas 4.1–4.3 in sequence
  to pin `r` down on each `θ_0`-slice, then use (R1) to pick the same
  branch across slices." T-A*: "Strategy: identify what part of
  T-A's proof breaks under Lipschitz regularity (the contradiction
  in Lemma 4.2), patch it by promoting (R3) from a derived property
  to an architectural one, and reuse everything else." T-C:
  "Strategy: work in `t = T(X)`-coordinates and re-run T-A's proof
  using the *general* source-density form of Lemmas 4.2 and 4.3 —
  which the relevant remarks in §4.2 set up exactly for this
  purpose." T-C*: "Strategy: the discrete case has a permutation
  freedom across `T`-atoms that the continuous case lacks; (R4)
  removes that freedom, and the proof reduces the lex-order tiling
  of `[0,1]` to a unique configuration."

These are pedagogical surface fixes that don't require any
mathematical change.

### Failure mode 4: Concept-introduction gaps

The two largest concept-introduction gaps in Part III are:

1. **Why we need the randomized PIT in §5.7.1 (lines 1087–1114).**
   The opening paragraph (lines 1089–1097) does give the *reason*
   ("no function of T(X) alone can have a continuous distribution,
   so N(0,1) is unattainable"), but the *intuition* for the specific
   randomized-PIT formula is one phrase: "smear each atom across an
   interval of an auxiliary uniform variable." A picture or a 3-4
   sentence build-up would help an astronomer who's never seen
   randomized confidence intervals. The right intuition:
   - In the continuous case, `F_θ(T(X)) ~ U(0,1)` by the standard
     PIT.
   - In the discrete case, `F_θ(T(X))` lives on a countable set —
     it can't be `U(0,1)`.
   - But each atom `t` *would* like to live uniformly on the interval
     `[F_θ(t⁻), F_θ(t)]` (a CDF "step" of width `p_θ(t)`).
   - The auxiliary `U` is exactly the device that places the atom
     uniformly within its step: `V_θ(X,U) = F_θ(T⁻) + U · p_θ(T)`.
   - Marginally, the steps tile `[0,1]` with the right widths, so
     `V_θ ~ U(0,1)` exactly.

   The text gives the punchline ("exactly") but not the
   tile-and-fill picture. A figure illustrating the tiling would be
   ideal; if a figure is too heavy, 3-4 sentences walking through the
   construction would suffice.

2. **Why (R4) is the right additional condition (and not some other
   condition).** §5.7.2 says (R4) is an "order-coherence" condition
   on `T(X)`, and the post-counterexample paragraph (lines 1231–1245)
   explains *why (R4) is automatic in the continuous case* — but the
   *meaning* of (R4) at the moment it's introduced (lines 1126–1136)
   is dense. The "order-coherence" name is good; the formal statement
   is technically precise; the parenthetical "the canonical
   `r*_rand = Φ⁻¹(1 − V_θ)` is decreasing in `T(X)` under increasing
   MLR" is informative *but appears after the formal statement,
   buried in the middle of a long itemize.* **Pulling that
   parenthetical out as a 1-2 sentence intuition before the formal
   statement — "the canonical `V*_θ` tiles `[0,1]` in increasing
   `T(X)`-order; (R4) is the condition that says any pivot in the
   class does the same" — would make (R4) feel inevitable rather
   than ad hoc.**

   Related: the parenthetical distinguishing (R3_U) from (R3) of
   §4.4 (lines 1120–1124) is essential but stylistically awkward.
   The reader meets (R3_U) in a sentence that simultaneously
   introduces it and disambiguates it from a previously seen
   condition. Splitting into two short sentences ("(R3_U) is a
   strict-monotone-in-`U` condition needed inside each `T`-atom in
   the discrete setting. It is the discrete-setting analogue of
   (R3) from §4.4, which played the same role for `X` in the
   Lipschitz case.") would reduce cognitive load.

### Failure mode 5: Narrative flow

The §4 → §5 → §5.7 arc is, in principle, a natural generalization
sequence: location-normal → continuous exponential family → discrete
exponential family. But several flow issues blur this:

1. **§5 vs §5.7 imbalance.** §5.1–§5.3 (continuous EF lift, including
   worked examples) runs lines 1000–1086, about 86 lines. §5.7
   (discrete extension) runs lines 1087–1265, about 178 lines —
   more than twice the length of the entire continuous-EF
   discussion, and the longest single subsection in the Part. The
   length is partly justified (T-C* is the most subtle theorem and
   the Bin(3,θ) counterexample carries weight), but the *narrative*
   weight of §5.7 is disproportionate to its conceptual role. To a
   first-pass reader, the message currently is "discrete extension
   is hard and you should worry about it a lot"; the intended
   message is "discrete extension works cleanly via a standard
   classical construction." Some redistribution would help: a
   2-3 line "punchline" framing at the start of §5.7 ("the
   discrete case is handled by a 1950s-vintage construction
   [Stevens, Tocher, Lancaster]; the resulting pivot is unique
   under a fourth regularity condition (R4) that selects the
   correct `T`-ordering. Readers willing to take this on faith
   can skip to §5.7.3 or §6.") would let the reader calibrate the
   weight.

2. **§5.3 worked examples feel disconnected from §5.2.** §5.3
   gives the Student-`t` and exponential-rate examples (lines
   1061–1086) but does not frame them as instances of T-C. The
   reader has to perform the verification themselves: "this is what
   T-C says for the location-`t` model" / "this is what T-C says
   for `Exp(θ)`". One framing sentence at the start of §5.3 ("The
   following examples instantiate T-C in the two regular cases most
   common in practice. In each, the canonical pivot
   `Φ⁻¹(1 − F_θ(T(X)))` is computable in closed form and reduces to
   a textbook pivot.") would bridge.

3. **§5.7.1 → §5.7.2 → §5.7.3 internal flow is good, but §5.7.3
   needs a stronger close.** §5.7.3 (mid-p, lines 1247–1263) ends
   with a citation to Hwang–Yang and no synthesis. The reader
   leaves §5.7 — and therefore Part III — on a technical
   citation footnote rather than a closing summary. **A 2-3 line
   "Part III wrap-up" at the end of §5.7.3 (or as a separate
   §5.8 / closing paragraph) — "Part III has shown that NF-MLE
   + (R1) + (R2) uniquely recovers the classical UMPU CD on every
   regular 1-parameter model, continuous or discrete. Part IV
   takes the same machinery to `d > 1` using triangular
   autoregressive flows" — would close the loop and bridge to
   Part IV.**

4. **The "Schweder–Hjort bridge" (§4.5) currently sits between
   T-A* (§4.4) and the EF lift (§5). It is the right place
   *logically* (the bridge closes T-A's claim), but it reads as a
   short interlude: 21 lines (977–997) between two heavier
   sections. As a pedagogical anchor — "this is what we just
   showed, in classical language" — it underweights itself. **A
   slightly longer §4.5 that explicitly says "we have done two
   things: (i) shown the calibration manifold is a singleton on
   `N(θ, 1)`; (ii) identified that singleton as the textbook
   UMP-unbiased CD. The next section shows both (i) and (ii)
   extend to any regular 1-parameter exponential family" would
   strengthen the transition.**

## Prioritized recommendations

In approximate order of (reader-impact / edit-cost):

### High priority

1. **Add a "goal + strategy" pair of sentences before each of T-A,
   T-A*, T-C, T-C*.** Surface revision, no math change. Directly
   addresses the user's flagged concern. (See Failure mode 3.)

2. **Add a "roadmap of the proof" paragraph at the start of §4.2,
   before the three lemmas.** 4–6 sentences laying out the role of
   each lemma. Greatly clarifies the proof of T-A and, by re-use,
   T-C / T-A* / T-C*. (See Failure mode 2.2.)

3. **Add a "what's happening here" sentence at the start of the
   proof of T-A (§4.3, line 902).** One sentence — naming the
   three-lemma sequence as a *strategy* before invoking it. (See
   Failure mode 2.3.)

4. **Build up the randomized-PIT intuition in §5.7.1.** 3–4
   sentences (or a figure) before the formula for `V_θ`. Astronomers
   in particular will not have seen this construction. (See Failure
   mode 4.1.)

5. **Glossary inserts for *monotone rearrangement*, *pushforward
   density*, *bi-Lipschitz*, *lex order*, *UMP-unbiased CD*.** All
   one-sentence inline glosses, all at first use. (See Failure
   mode 1.)

### Medium priority

6. **Reframe the Bin(3,θ) counterexample in §5.7.2** with a 2-3
   sentence opener that names its *role* (proving (R4) is not
   redundant). (See Failure mode 2.6.)

7. **Pull the "(R4) is automatic in the continuous case" intuition
   (lines 1231–1245) forward into the statement of (R4).** Two
   lines before the formal (R4) statement would make (R4) feel
   inevitable. (See Failure mode 4.2.)

8. **Reframe §5.3 worked examples** as explicit instances of T-C.
   One framing sentence at the start of §5.3. (See Failure mode
   5.2.)

9. **Add a Part III wrap-up paragraph at the end of §5.7.3** that
   summarizes the four theorems and bridges to Part IV. (See
   Failure mode 5.3.)

10. **Expand §4.5 (Schweder–Hjort bridge) by 5–10 lines** to
    explicitly call out "two things we've shown" and motivate the
    EF lift. (See Failure mode 5.4.)

### Low priority

11. **Split the (R3) vs (R3_U) parenthetical** in §5.7.2 (lines
    1120–1124) into two short sentences. (See Failure mode 4.2.)

12. **Soften the "Selecting the direction" paragraph** (§4.3,
    lines 916–929) with a one-line frame before the
    counterfactual. (See Failure mode 2.3.)

13. **Add a brief signpost at the start of §5.7** flagging length
    and offering a skip option. (See Failure mode 5.1.)

14. **Trim or footnote the "Lemma 4.2 / general source remark"
    apparatus** (lines 868–878) — currently invoked silently in
    T-C and T-A-d. Pedagogically, it would be cleaner to *state*
    a "general L-4.2" once and refer to it, rather than have a
    remark that has to be inferred at each invocation.

## Existing strengths to preserve

Several pedagogical patterns in Part III already work well and
should be kept (or used as models for the weaker passages):

- **§4.4's opening (lines 933–936)** is *exactly* the right model
  for a "why this section" framing: it tells the reader what failed
  in §4.3 (Lemma 4.2's contradiction breaks for V-shapes) and what
  the section will do about it. This is the template the other
  theorem statements should follow.

- **The Rademacher gloss in §4.4 ("a Lipschitz function on R^n is
  differentiable Lebesgue-a.e.")** is the right model for inline
  jargon definitions: one parenthetical, in context, no
  disruption.

- **The "Background, brief" paragraph in §5.1 (lines 1002–1012)**
  is good: it defines exponential family, sufficient statistic, and
  MLR inline and cites a textbook for the reader who wants more.

- **The "When is (R3) automatic?" paragraph (lines 970–975)** is
  good practical bridge from theory to architecture, and matches
  the manuscript's stated principle ("monotonicity is
  architectural, not penalty-based").

- **The ancillary-statistic definition in §5.3 (line 1063)** is
  the right model for an inline gloss before first use.

- **The §5.7.2 explicit Bin(3,θ) construction**, even though its
  *purpose* needs framing (see Failure mode 2.6), is well-executed
  on its own terms: the verification of (R1), (R2), (R3_U) is
  explicit and the (R4)-violation calculation is clean. The
  "perfect-square accident" `−3(2θ − 1)^2 ≤ 0` (line 1209) is a
  nice piece of mathematical color that should not be lost in any
  rewrite.

- **The cross-references to round-1 fixes** (e.g., the explicit
  acknowledgement that Bin(1,θ) and Bin(2,θ) don't work, lines
  1162–1175) keep the manuscript honest. Preserve.

- **The "decreasing-MLR case differs only in sign convention"**
  aside in §5.1 (line 1018) is exactly the right level of
  "we've thought about this, the reader can take it on faith"
  signposting. More of this in §5.7 would help.
