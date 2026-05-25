# Round 3 — Part V (Empirical validation) Discovery & Assessment

**Scope:** §7 (architecture, training, diagnostics) and §8 (four
experiments + synthesis), lines 1571–2080 of `cd_sbi_v7.tex`.
**Audience target for round 3:** statisticians + astronomers; the user
has explicitly flagged architecture-design content (which is precisely
§7) as the section most in need of accessibility work.
**Failure modes to look for:** (i) oracle-style architecture
descriptions in §7.1 (UMNN), (ii) astronomer-inaccessible diagnostic
jargon in §7.3, (iii) experiments that read as standalone case studies
rather than evidence for specific theoretical claims.

---

## Phase A: Discovery

### A.1 Top-line goal

Part V has a single integrated goal: **demonstrate that the
population-level theory of Parts III–IV (Theorems A, A-d, the
KR-rearrangement corollary, and the (R2) architectural requirement) is
actually attainable in practice, at practical finite-sample / CPU
budgets, and that the architectural prescriptions distilled from the
theory are the right ones.** The §8 preamble (lines 1703–1709) makes
this stance explicit: the experiments are not "checking that the method
works" (the theory already settled correctness in the limit) but are
*evidence that finite-sample optimization reaches the population
optimum and that architectural prescriptions matter as predicted*.

### A.2 Central arguments (4)

1. **The diagnostic suite has a principled hierarchy.** Coverage is
   the inferentially primary diagnostic — it is the property the
   framework promises practitioners. Diagnostics 1–4 (pivot RMSE,
   marginal PIT, conditional PIT, joint Mahalanobis) are
   *interpretability* checks that localize *where* a calibration
   failure lives. The hierarchy paragraph at lines 1652–1662 was
   inserted/strengthened in round 1.

2. **The UMNN architecture mechanically enforces (R1) and is the right
   building block for the framework.** Parameterizing a function as
   the integral of a strictly positive function gives strict
   monotonicity by construction; this is what NF-MLE needs and what
   autograd-only approaches fail to deliver (§8.4 ablation).

3. **Each of Theorems A, A-d, and the (R2) architectural prescription
   is empirically recovered in a regime where the canonical answer is
   known closed-form.** This is the central evidentiary argument of
   §8: by choosing four regimes where `r*` is analytically known, the
   experiments turn theoretical correctness statements into testable
   recovery claims.

4. **Architectural enforcement of (R2) is not optional.** §8.4's
   autograd-Jacobian ablation produces a loss *below* the
   conditional-entropy lower bound (0.56 < 0.88 = `H(X|θ)`), which by
   Gibbs's inequality is impossible for any valid normalized density.
   This is direct empirical proof that the autograd surrogate is *not
   a density* in the relevant sense — the cautionary tale.

### A.3 Evidence map: experiment → theoretical claim

| §    | Setup                                | Validates                                                             | How                                                                 |
|------|--------------------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------|
| 8.1  | `X ~ N(θ, 1)`                        | **Theorem A** (1D `C¹` uniqueness in `F_{C¹}` → `r* = θ − X`)         | RMSE 0.030; slopes `∂_θ r ≈ 1`, `∂_X r ≈ −1` recovered to 1–2%      |
| 8.2  | `X | θ ~ N(θ, I_2)`                  | Triangular-flow construction (§6.1–6.2) on the diagonal/clean case    | Joint Mahalanobis KS 0.011 — passes the multivariate-specific check |
| 8.3  | `X | θ ~ N(θ, Σ)`, Σ correlated      | **Theorem A-d** + KR corollary: `r^KR = L⁻¹(θ − X)` for Gaussian      | Recovered `E[J_θ]` matches `L⁻¹` to 1–2%; coverage < 1.5%           |
| 8.4  | `T = Σ X_i, X_i ~ Exp(θ)`            | **(R2) sufficiency**: doubly-monotone construction works              | RMSE 0.045; loss matches truth's 0.88 to batch noise                |
| 8.4* | same model, autograd Jacobian        | **(R2) necessity**: non-bijective surrogates fail catastrophically    | Loss 0.56 < 0.88 (Gibbs-impossible); coverage error up to 3.4%      |

§8.5 makes this mapping explicit in a synthesis table (lines 1987–2013).
The mapping *is* in the manuscript; the question is whether the reader
sees it before reading each experiment, or only retrospectively in
§8.5.

### A.4 Implicit structural arguments

- **(§7.3) The per-bin KS noise floor is the right comparator for
  conditional statistics.** The round-1 noise-floor reorganization
  (lines 1664–1691) corrects v6 commentary that compared conditional
  KS values to the marginal floor; the corrected per-bin floor is
  `1.628/√(N/k)`, which is roughly 2× looser than the marginal floor.
  This is a quiet but important technical correction.

- **(§8.3) Pointwise pivot RMSE is the wrong diagnostic in
  multivariate.** The "RMSE 0.27 is not alarming" discussion
  (1881–1907) walks through why: in `d > 1`, recovered Jacobian +
  coverage are the right diagnostics because the calibration manifold
  is a singleton (Theorem A-d) and the pointwise residual is
  finite-network noise around that singleton, not evidence of
  non-uniqueness.

- **(§7.2) NF-MLE has a remarkably simple training recipe.** No
  annealing, no curriculum, no warmup — just Adam + grad clipping +
  3-4k steps. The brevity of §7.2 is itself an argument (the loss
  doesn't need babysitting).

---

## Phase B: Accessibility assessment

### B.1 Undefined / under-explained terms

Cataloguing terms a hybrid statistician/astronomer audience will hit
in Part V. For each, I note whether the manuscript defines, links, or
leaves undefined.

| Term                                 | Where         | Current treatment                                                  | Assessment                                                                                                          |
|--------------------------------------|---------------|--------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| **UMNN**                             | §7.1 def      | Cited (`WehenkelLouppe2019`) + definition box with the integral    | Definition is symbolic ("integral of `σ_+ ∘ MLP`"). The *concept*— "to build a monotone function, integrate a positive one"— is not stated in prose. Critical gap for astronomers; addressed below. |
| **softplus / ELU**                   | §7.1          | Mentioned, not defined                                             | softplus = `log(1 + e^x)`; ELU = piecewise. Astronomers will need a half-sentence ("`σ_+` is just a smooth strictly-positive squashing function — softplus = `log(1+e^x)` here"). |
| **MLP**                              | §7.1          | Used unannotated                                                   | "Multi-layer perceptron" — standard in ML, opaque to a stats/astro reader who hasn't done deep learning. One inline gloss suffices. |
| **Gauss–Legendre / Clenshaw–Curtis** | §7.1          | Named without definition                                           | Numerical-quadrature schemes. One sentence: "fixed-node numerical integration of `σ_+` over `[z_0, z]` — Gauss-Legendre uses ~12 carefully chosen sample points." |
| **bias(c)** / "near-identity init"   | §7.1          | Stated                                                             | The *intent* (start as close to `r(θ,X) ≈ 0` so optimization has room to move) is implied but not stated. A clause helps. |
| **learnable α scalar**               | §7.1          | "init 0.5 / ln 2 ≈ 0.72, deliberately wrong"                        | "Deliberately wrong" is a great phrase but unexplained. *Why* would you initialize at the wrong scale on purpose? (Answer: so the optimizer must actually move, ruling out trivially "lucky" init.) Worth one clause. |
| **Adam / gradient clipping / batch size** | §7.2     | Named, not defined                                                 | Standard in ML. For astronomers: Adam = adaptive-momentum SGD variant; gradient clipping = cap gradient norm to avoid blowup; batch size = samples per gradient step. One footnote covering all three would suffice. |
| **KS test / Kolmogorov limiting distribution** | §7.3 | Used; `1.628` quantile stated; round-2 review noted missing cite   | After round 2, no textbook cite was added (Lehmann-Romano was flagged optional). For round-3 accessibility, a half-sentence ("KS measures the maximum gap between empirical and target CDFs; `√N · KS` converges to the Kolmogorov distribution under the null") would land better than the bare `1.628`. |
| **PIT (marginal / conditional)**     | §7.3          | Used without expansion                                             | "PIT = Probability Integral Transform". The asymmetry between marginal (over the proposal) and conditional (at fixed `θ_0`) PIT is the *whole architectural reason* for the diagnostic hierarchy, and this asymmetry is buried in the formula. The hierarchy paragraph explains the *order* but not what *PIT* itself means. |
| **Joint Mahalanobis / χ²_d**         | §7.3 (Diag 4) | Stated as `‖r‖² ~ χ²_d`                                           | OK for statisticians; astronomers know `χ²_d` from goodness-of-fit. The phrase "joint Mahalanobis" is a hard hit if you've not used the term before — it's `‖r‖²` here, and a parenthetical "(squared Euclidean norm of the pivot — Mahalanobis distance under `N(0, I_d)`)" would help. |
| **SBC / TARP / LF2I**                | §7.3 end      | Cited with one-phrase characterizations                            | The names are acronyms with no expansion in-text. The round-2 review noted these are correctly characterized; for round-3, expanding once ("simulation-based calibration (SBC)") is sufficient. |
| **proposal `ρ` / "support boundary"**| §8.1 commentary | Used                                                             | `ρ` was introduced earlier in the manuscript, but the "training-distribution effect" gloss in §8.1 is good — keep. |
| **conditioning network / cross-coupling** | §8.2-8.3 | Used                                                              | "Conditioning network" = the contextual UMNN; "cross-coupling" = `r_2` depending on `(θ_1, X_1)`. Astronomers will need a one-line gloss for "conditioning network". |
| **recovered Jacobian**               | §8.3          | Used heavily as the multivariate diagnostic                        | Defined implicitly by the formula in the table, but "the Jacobian of `r` with respect to `θ`, evaluated by autograd at 200 random `(θ, X)` points" would help land it. |
| **sufficient statistic `T`**         | §8.4          | Used                                                               | Astronomers may have met this in maximum-likelihood contexts; a parenthetical that `T = Σ X_i` is the sufficient statistic for `θ` in `Exp(θ)` (and that this is why the framework can compress 5 samples to a 1D `T`) ties the choice to §5. |

**Concept-level gap (UMNN).** The most important accessibility issue
in §7.1: the UMNN definition is given symbolically but the *idea* is
not stated in prose. The idea is one sentence: *to build a strictly
monotone function, integrate a strictly positive function; the
positive function is parameterized by an unconstrained neural network,
which is where the "U" comes from.* This sentence belongs at the top
of §7.1.

### B.2 Oracle-style passages — section-by-section

**§7.1 (UMNN definition).** Currently reads as a self-contained
definition box: name, formula, modifications from `WehenkelLouppe2019`,
implementation details (hidden width, init, learnable `α`). The
formula is correct and complete, but the *narrative motivation*
(why-this-shape, not just what-the-shape-is) is missing. A reader who
hasn't seen UMNN before gets the recipe without the rationale. The
discussion of the "Jacobian factor in NF-MLE without recursive
autograd" (line 1591) implicitly assumes the reader remembers why
autograd-recursion is bad (it's in §8.4, but §7.1 doesn't link
forward). The "learnable `α` initialized deliberately wrong" is a
*great* design choice but the reader doesn't know it's a design
choice unless they re-read.

**§7.3 hierarchy paragraph (lines 1652–1662).** The round-1 insertion
explains *the order* of diagnostics and one technical point (marginal
PIT implied by conditional in the population but not finite sample;
joint Mahalanobis not implied by component-wise conditional PIT). What
it does *not* do is connect each diagnostic to *which failure mode it
detects*. A small table would help: e.g., "RMSE detects training
non-convergence; marginal PIT detects gross global miscalibration;
conditional PIT detects local miscalibration; joint Mahalanobis
detects correlated-error / Hermans-type failures; coverage detects the
property that actually matters for inference". The Hermans-type
failure pattern is named (line 1662) but not explained; a one-line
explanation ("two component CDs can each be marginally `N(0,1)` while
their joint is correlated, so component-wise checks miss the
failure") would land it.

**§7.3 noise-floor passage (lines 1664–1691).** Mathematically correct
and the round-1 correction to per-bin floor is essential. As written,
the reader is shown the formulas (`1.628/√N`, `1.628/√(N/k)`) but the
*reason* the floors differ — that a per-`θ_0` conditional KS uses
only the subsample at that `θ_0`, so the relevant `N` is `N/k`, not
`N` — is implicit. One sentence ("conditional statistics use a
*per-bin* sample of size `N/k` because the conditional check is
evaluated only on the `N/k` samples falling in that `θ_0` bin")
between the two bullets would explain *why* the conditional floor is
2× looser. The closing paragraph that explicitly flags the v6 error
(lines 1685–1691) is excellent — keep it.

**§8 preamble (lines 1703–1735).** The "experiments are not validation
in the sense of checking the method works" framing is exactly the
right framing for the reader, and it should be more prominent — it is
currently buried in a single paragraph between the section header and
the mapping table. A dedicated subsection header or a bolded one-line
restatement would help. The mapping table itself (Experiment ↔
Supports) is the *outline of the empirical argument*; that's a
high-value object that should be flagged as such.

**§8.3 "RMSE 0.27 is not the right diagnostic" (lines 1881–1907).**
The argument is correct and important: in multivariate, pointwise RMSE
conflates approximation error with non-canonical alignment. The
walk-through (`r_2` marginally `N(0,1)`, RMSE is 27% of marginal std,
the relevant diagnostic is Jacobian + coverage) is *almost* there but
condensed. The reader is shown the variance calc
(`0.577² + 1.155² − 2 · 0.577 · 1.155 · 0.5 ≈ 1.000`) without being
told what it accomplishes. Spelling out the punchline first — "in
multivariate, even the *truth* `r*` doesn't reduce variance to zero
pointwise; it just makes `r*(θ_0, X) | θ_0` standard normal, which has
unit variance — so any RMSE smaller than 1 is *better than the
constant-zero baseline*, and RMSE = 0.27 is fine" — would land
faster. The "What this validates" paragraph (1898–1907) is good.

**§8.4 ablation argument (lines 1958–1978).** The round-1/2 rewrite is
substantive: the ablation now ties to (i) Gibbs's inequality lower
bound `H(X|θ) = 0.88`, (ii) the ablation reaching 0.56 (below 0.88,
which is impossible for a valid density), (iii) the `−log Z(θ)`
mechanism from §3.5 (when `T → r` folds, the surrogate has
`Z(θ) > 1`, the loss shifts down). This is a complete argument but
dense — three ideas in three sentences. A reader sees: "Gibbs
inequality says loss ≥ entropy → the ablation breaks this bound →
therefore the surrogate is not a density → here's the mechanism".
That's a four-step story; presenting it as such ("First… therefore…
the mechanism is…") rather than as a single paragraph would help.
*The fact that the loss is below the truth value is the smoking gun*
— this could be set off explicitly (e.g., italic emphasis on the
inequality `0.56 < 0.88`).

### B.3 Proof-role clarity / empirical-argument structure

Each experiment in §8 should ideally open with a half-sentence saying
"this experiment is evidence for [specific theorem / claim]". Let me
audit:

- **§8.1** opens with "Setup", not "Validates Theorem A". The
  Theorem-A connection appears only in §8.5's synthesis table. *Add
  one line at the top:* "This experiment validates Theorem A in the
  regime where it applies (1D, location-normal): we expect to recover
  `r*(θ, X) = θ − X` exactly."
- **§8.2** opens with "Setup". The synthesis table maps it to §6.1–6.2
  construction. *Add:* "This experiment is the cleanest test of the
  triangular-flow construction: diagonal `Σ`, so the truth factorizes
  and `r* = θ − X` componentwise."
- **§8.3** has "What this validates" *at the end* (line 1898). That's
  the right content but in the wrong place. *Move (or mirror) at the
  top:* "This experiment validates Theorem A-d for Gaussian models:
  the KR rearrangement is `L⁻¹(θ − X)`, which we expect to recover as
  the unique Jacobian `L⁻¹`."
- **§8.4** has a good subsection title ("validating the non-additive
  case") and the (R2) connection is in the ablation paragraph. *Add
  at the very top:* "This experiment validates two things: (i) the
  doubly-monotone construction handles models outside the additive
  class, and (ii) architectural enforcement of (R2) is a correctness
  requirement, demonstrated by the autograd-Jacobian ablation."

§8.5 synthesis table is good and does the *retrospective* mapping.
The proposed front-loading would let the reader follow the argument
*as it unfolds*, with §8.5 then being a true synthesis rather than the
only place the mapping appears.

### B.4 Concept-introduction gaps

**The diagnostic suite as a concept.** The hierarchy is now stated
explicitly (round 1's contribution) but the *practitioner takeaway* —
"if I have one number to report, it's coverage" — is not quite
foregrounded. A reader leaves §7.3 knowing the suite, but a
practitioner-oriented one-liner ("**In practice: report coverage as
the headline result; use 1–4 to diagnose where any miscalibration
sits**") would close the loop. The hierarchy paragraph at 1652–1662
*almost* says this; tightening the lede sentence would help.

**Per-bin vs marginal floor distinction.** The round-1 correction is
correct but the *reason* the distinction matters — that misapplying
the marginal floor to conditional KS values causes false alarms
("borderline" when the value is actually clean) — is buried in lines
1685–1691. This is one of the most useful concrete takeaways for a
practitioner (you will run KS tests on your conditional PIT and need
to know which floor to use), and the section could foreground it
more.

**Why architecture beats penalty.** The §8.4 ablation is the
empirical complement to the §3.5 / §6.1 architectural argument: you
*cannot* substitute a soft monotonicity penalty for architectural
enforcement, because a non-bijective `r` is not a density at all and
the loss is no longer a calibrated objective. §8.4 says this but the
*generalization* — "the same logic forbids any non-architectural
monotonicity enforcement" — is left implicit. §8.5's prescription
table (lines 2052–2064) lists "architecturally" three times, which is
the right repetition, but doesn't explicitly say "and *not*
penalty-based".

### B.5 Narrative flow

**§7 → §8.** §7 sets up the architecture, training, and diagnostics
without prejudice to the specific experiments; §8 then deploys them
on four model classes. The structural separation is clean and
correct. The forward link from §7 to §8 (e.g., "the diagnostic suite
defined here is applied uniformly to the four experiments below") is
implicit but not stated; one transition sentence at the end of §7.3
would close the loop.

**Within §8.** Each experiment follows the same template (Setup,
Architecture, Results table, brief commentary). The template is
repeatable and easy to scan — good. The issues are:
- the *front-loading* problem in B.3 (what does this experiment
  validate?);
- the experiments feel parallel rather than progressive — they could
  be framed as a "ladder of increasing difficulty" (1D → 2D
  diagonal → 2D correlated → non-additive), which is what the table
  at 1711–1733 implies but the prose doesn't quite say;
- §8.4 is the most interesting experiment (the only failure case, the
  cautionary tale) and is currently the last and longest, which is
  good *if* the reader knows in advance to expect a climax — a
  preamble sentence in §8 ("the fourth experiment includes the
  paper's most important negative result: an architectural-ablation
  that demonstrates (R2) cannot be enforced via autograd") would
  flag this.

**§8.5 synthesis.** The synthesis table (1987–2013) is excellent and
does the retrospective mapping. The summary table (2017–2048) is
useful but partially redundant with the per-experiment tables —
arguably the synthesis table makes the summary table optional, or the
two could be merged. The architectural-prescription list at the end
(2052–2064) is the takeaway for practitioners and lands well; one
small tweak (calling out "not penalty-based" alongside
"architecturally") would sharpen it.

**Open-work caveat (2065–2077).** The honest statement of what §8
does *not* validate (no genuinely multivariate non-additive case, no
high-`d` `X`, no test of alternative losses) is well-handled — keep.
This is the kind of caveat that earns reader trust.

---

## Prioritized recommendations

### High priority (largest accessibility win per word added)

1. **§7.1 — UMNN concept sentence.** Add one prose sentence at the
   top of the definition explaining the *idea*: "A UMNN parameterizes
   a strictly monotone function `g(z; c)` by writing it as the
   integral of a strictly positive function — strict monotonicity
   then follows from positivity of the integrand by the fundamental
   theorem of calculus. The 'unconstrained' refers to the inner
   neural network, which has no monotonicity constraint of its own;
   monotonicity is supplied by the integral structure." This is the
   single highest-value sentence to add in Part V.

2. **§8 (each subsection) — front-load the claim.** Add a
   one-sentence "validates X" lede to each of §8.1, §8.2, §8.3, §8.4
   (see B.3 for proposed wording). This converts each experiment from
   a standalone case study into a stage in an argument.

3. **§7.3 — per-bin vs marginal floor: explain *why*.** Add one
   bridging sentence between the marginal and per-conditional
   bullets explaining that conditional KS uses only the per-`θ_0`
   subsample. The technical content is correct; only the rationale
   needs surfacing.

4. **§8.4 — restructure the ablation argument as a 4-step story.**
   Split the dense paragraph at 1958–1978 into: (i) the ablation
   result (RMSE 1.56, coverage error 3.4%); (ii) the Gibbs inequality
   gives `loss ≥ H(X|θ) = 0.88`; (iii) the ablation reaches 0.56,
   which is *below* the lower bound — impossible for a valid
   density; (iv) the mechanism (folding `T → r` makes `Z(θ) > 1` and
   shifts the loss). The 0.56 < 0.88 inequality is the smoking gun
   and deserves emphasis.

### Medium priority

5. **§7.1 — gloss `σ_+`, MLP, Gauss-Legendre, learnable `α`.** One
   half-sentence each, ideally inline. The "deliberately wrong"
   initialization is a great phrase that needs explanation ("so
   optimization has to find the right scale rather than starting
   accidentally close to the truth").

6. **§7.2 — micro-footnote on Adam / batch size / gradient
   clipping.** A single shared footnote for the three ML-jargon terms
   would render §7.2 self-contained for the astronomer reader.

7. **§7.3 — diagnostic-failure-mode mini-table.** Add a small
   2-column table or bulleted list mapping each diagnostic to the
   failure mode it detects (RMSE → training non-convergence; marginal
   PIT → gross miscalibration; conditional PIT → local
   miscalibration; joint Mahalanobis → Hermans-type correlated-error;
   coverage → the property that matters). Currently the connection is
   implicit.

8. **§7.3 — Hermans-type failure: one-line explanation.** Currently
   named (line 1662) but not explained. Add: "two component CDs can
   each be marginally `N(0,1)` while their joint distribution is
   correlated, so component-wise conditional-PIT checks pass but the
   joint distribution is miscalibrated."

9. **§8 preamble — make the empirical-argument framing more
   prominent.** The "experiments are not 'checking the method works'
   but evidence for specific claims" framing (1703–1709) should be
   set off — possibly as a "**Stance.**" or "**What §8 is for.**"
   labelled paragraph.

10. **§8.3 — front-load the multivariate-RMSE punchline.** Replace
    the variance-calc opener with "the relevant baseline is not zero:
    even the truth `r*` has marginal variance 1 by construction, so
    pointwise RMSE 0.27 is 27% of the marginal std, not 27% of
    'truth'. Pointwise RMSE is not the multivariate diagnostic; the
    Jacobian and coverage are." Then keep the variance calculation
    as the supporting detail.

### Low priority (polish)

11. **§7.3 — practitioner one-liner.** "**In practice: coverage is
    the headline result; the other diagnostics localize where any
    miscalibration sits.**"

12. **§8.5 — "not penalty-based" call-out.** In the prescription list
    (2052–2064), one parenthetical that "architecturally" rules out
    soft-penalty enforcement would tie back to §8.4's lesson
    explicitly.

13. **§7 → §8 transition.** End §7.3 with "These diagnostics will be
    applied uniformly to the four experiments below."

14. **§8 preamble — flag §8.4 as a climax.** "The fourth experiment
    includes the paper's most important negative result, an
    architectural ablation that demonstrates (R2) cannot be enforced
    by autograd."

15. **Expand acronyms on first use:** UMNN, MLP, PIT, KS, SBC, TARP,
    LF2I — most are already cited but not spelled out. A consistent
    expand-once policy would help.

---

## Existing strengths to preserve

- **The §7.3 diagnostic hierarchy paragraph (1652–1662).** The round-1
  insertion is *exactly* the right kind of pedagogical addition —
  giving the reader the conceptual ordering before the details. Do
  not unwind.

- **The §7.3 noise-floor correction (1664–1691).** Honest, explicit
  flag of the v6 error and the corrected calculation. This earns
  reader trust and is exactly the level of transparency the
  manuscript should have throughout.

- **The §8 preamble framing.** "Experiments are not validation in the
  sense of 'checking the method works' — the theory already
  establishes correctness in the population limit" is the right
  meta-framing for the whole section. Keep — just elevate.

- **The §8 → claims mapping table (1711–1733).** A compact, scannable
  outline of the empirical argument. Keep.

- **§8.3's "What this validates" paragraph (1898–1907).** Gold-standard
  example of what each experiment should have at the *top* (and
  arguably already has at the bottom).

- **§8.4 architectural setup (1918–1929).** The doubly-monotone form
  is shown in full with closed-form derivatives. This is the right
  level of detail for the "show your work" architectural section.

- **§8.4 ablation result with Gibbs-inequality argument.** The loss-
  below-truth observation tied to Gibbs is *the* pedagogical
  highlight of Part V. Restructure for clarity (recommendation 4)
  but preserve the substance.

- **§8.5 synthesis table mapping experiments → theoretical claims.**
  Closes the loop on the empirical argument. Keep.

- **§8.5 open-work caveat (2065–2077).** Honest statement of what §8
  does *not* validate. Keep — this is the kind of self-aware caveat
  that distinguishes good empirical work from oversold empirical
  work.

- **The "deliberately wrong" `α` initialization (§7.1).** Distinctive
  design choice, worth keeping; just needs one clause of motivation.

- **Compact, uniform Setup / Architecture / Results / Commentary
  template across the four experiments.** Makes §8 highly scannable.
  Keep.

- **The §10 doubly-monotone (R1)+(R2) prescription at the end of
  §8.5.** The three-item architectural prescription
  (NF-MLE loss; (R1) architecturally; (R2) architecturally) is the
  paper's central practical takeaway. Keep prominently.
