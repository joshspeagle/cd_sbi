# Round 3 — Part VI (Practice & position) Discovery & Assessment

## Phase A: Discovery

### Top-line goal(s)

Part VI translates the framework of Parts I–V into something a
practitioner can actually use: a checklist for training a CD-SBI pivot
on a new model (§9), and a placement of CD-SBI in the SBI methods
landscape so readers know what CD-SBI is for, what it is *not*, and
which existing methods are closest to it conceptually (§10).

### Central arguments

1. **The architectural choice flows from the structure of `r*`.** §9 step
   1 organizes the architecture decision around four scenarios (additive
   truth → additive flow; multivariate with cross-coupling → triangular
   additive; multiplicative θ–X → doubly-monotone; unknown structure →
   autoregressive doubly-monotone). The recipe presumes the reader can
   guess this from domain knowledge.
2. **Both monotonicities must be enforced architecturally, not by
   penalty.** §9 step 2: autograd-only Jacobians fail because the loss
   is unbounded below when `r` folds (§3.5 mechanism; §8.4 ablation).
   Round 1's fix is now in place — the prose explicitly flags "theory
   doing the heavy lifting; single ablation is the smoking gun."
3. **The proposal `ρ` is a sampling device, not a Bayesian object.** §9
   step 3: pick `ρ` with full support; finite-sample edge effects
   appear near boundaries (§8.1).
4. **The training recipe is mundane.** §9 step 4: no annealing, Adam at
   3×10⁻³, batch 256–1024, 3k–10k steps. Deliberately unsurprising —
   the heavy lifting is upstream in the architecture choice.
5. **Coverage is the inferentially primary diagnostic.** §9 step 5:
   marginal-PIT-passes-while-conditional-fails is the diagnostic
   symptom that practitioners should treat as an architecture problem,
   echoing Hermans et al. (2022).
6. **Confidence sets come from a chi-square cutoff, computed by 1D
   root-finding.** §9 step 6: `C_α = {θ : ‖r(θ, X_obs)‖² ≤ χ²_{d,α}}`
   is autoregressive root-finding, no MCMC.
7. **CD-SBI's place in the landscape: SNL loss + LF2I/WALDO output, but
   single-stage.** §10's framing — the §10 prose names LF2I, WALDO, and
   SNL as the "closest cousins," with CD-SBI distinguished by
   (a) targeting a CD rather than a posterior/likelihood/ratio, and
   (b) achieving calibration architecturally rather than via a
   separately-trained critical-values branch (LF2I/WALDO) or
   regularizer/conformal correction (Balanced NRE / CANVI).
8. **The implicit-likelihood-ratio bridge.** §10's closing paragraph
   identifies `r*(θ; X) = θ − X` for the location-normal as the
   *negative* standardized score (round-1 sign fix) and gestures at the
   signed-root log-LR connection in regular exponential families
   (Schweder–Hjort 2016 ch.~5; Cranmer et al. 2015).

### Supporting evidence

**§9 recipe steps:**

| Step | Evidence drawn from | Form of support |
|---|---|---|
| 1. Architecture choice | §6.1 (additive / doubly-monotone forms); §8.1–8.4 (each row matches one experiment except the "unknown structure" default) | A four-row table; the fourth row ("no prior knowledge") is the only row not validated by an experiment, and the round-1 hedge ("informed by the §6.1 form-2 architecture rather than empirically validated for the unknown-structure case") is in place. |
| 2. Architectural monotonicity | §3.5 (KL/Gibbs argument: folded `r` → `Z(θ) > 1` → loss unbounded below); §8.4 ablation (loss 0.56 < 0.88 truth, RMSE 1.56 vs 0.045) | Strong: theoretical mechanism plus empirical confirmation. The "we treat this as a hard rule because the failure mode is structural" framing is round 1's. |
| 3. Proposal `ρ` | §1.3 (population invariance to `ρ`); §8.1 (edge-effects evidence) | Adequate. |
| 4. Training schedule | §7.1–7.2 (UMNN hidden-width 32, batch 512, Adam 3×10⁻³, 3,000–4,000 steps) | Engineering numbers from §8 reproduced in summary form. |
| 5. Validation | §7.3 (five-diagnostic checklist); HermansEtAl2022 citation | The pointer to §7.3 is correct; the round-1 fix replaces "Hermans-style" with the longer "the kind of conditional miscalibration that Hermans et al. (2022) documented." |
| 6. Inversion | §6.2 (triangular Jacobian); §6.4 inversion algorithm | Pivot inversion by 1D root-finding on each coordinate is concrete and immediate. |

**§10 comparison table — every row has a citation, all verified in round 2:**

| Row | Round-2 status |
|---|---|
| NPE/SNPE; NLE/SNL; NRE/SNRE; Balanced NRE; Calibrated NPE; LF2I; WALDO; Box CD; CANVI | All 12/12 citations verified; the Balanced NRE row carries the round-1 "(goal: more reliable posteriors)" parenthetical that resolves the original target-column inconsistency. |
| CD-SBI (this work) | "Confidence distribution / Pointwise, by construction" — the comparative claim that makes the whole section land. |

**§10 commentary (post-table):**

- "Closest cousins": LF2I and WALDO (Neyman-inversion lineage) and
  SNL (same NF-MLE loss). Round 1 expanded the SNL contrast to name
  *both* monotonicity constraints, not just monotone-in-θ.
- Single-stage vs critical-values branch: the architecturally enforced
  population-limit identity `r | θ ~ N(0, I_d)` is the closed-form
  reference that lets CD-SBI dispense with a calibration sample.
- IRT bridge: `r*(θ; X) = θ − X` is the *negative* standardized score;
  in regular exponential families this connects to the signed-root
  log-LR.

---

## Phase B: Accessibility assessment

### 1. Undefined terms for non-specialists

For an astronomer or physical scientist reading Part VI in isolation
(skimming Parts I–V):

- **"Additive flow" / "triangular flow" / "doubly-monotone UMNN"**
  (§9 step 1 table). Each is *named* with a `\S\ref{subsec:6.1}`
  pointer but **not re-explained** in §9. For a reader who came to §9
  before fully internalizing §6.1, the architecture-choice table
  reduces to a four-row recommendation in which the right-column
  entries are opaque labels. The right-column entries cite §6.1
  and §8.x by number; they do not paraphrase the architecture in one
  line. **Fix candidate:** add a one-sentence parenthetical gloss to
  each row, e.g. *"additive flow: r = a(θ) − b(X) with a, b separate
  UMNNs"; "triangular flow: autoregressive, each r_k depends on
  θ_{≤k}, X_{≤k}"; "doubly-monotone: r is a softplus integral in θ
  that is also monotone-non-decreasing in X."*
- **"Autoregressive conditioning"** (§9 step 1, fourth row). Same
  problem: named once, not re-glossed. The reader needs to remember
  §6.1's "k-th output depends only on inputs with index ≤ k."
- **The §10 table's nine method abbreviations** (NPE / SNPE / NLE /
  SNL / NRE / SNRE / Balanced NRE / Calibrated NPE / LF2I / WALDO /
  Box CD / Variational SBI w/ coverage). The acronyms are unpacked
  in §1.1 but not in §10. A reader skimming the §10 table without
  having read §1.1 will not recognize them. The table is dense:
  one-line method labels, one-line targets, one-line calibration
  notes. For a reader who already knows the field this is the
  whole point (a cheat sheet). For a newcomer it is unparseable.
  Round 2 verified all citations; that does not solve the pedagogy
  gap, which is that the table assumes prior familiarity. **Fix
  candidate:** either expand each row with a half-sentence describing
  what the method does (lengthens the table) or move the "field
  cheat sheet" framing into a sentence introducing the table
  ("Readers unfamiliar with these methods should consult §1.1 first;
  this table is a side-by-side comparison").
- **"Neyman construction" / "Neyman inversion" / "critical-values
  branch"** (§10 table + post-table prose). The vocabulary is
  introduced in §1.1 ("learn test statistics and invert via Neyman
  construction") but the *mechanics* — that LF2I/WALDO require an
  auxiliary calibration set to estimate `C_α(θ)` Monte-Carlo-wise —
  are not spelled out in §10. The post-table sentence about
  "critical-values branch" is the only place this is named. An
  astronomer who knows ABC but not LF2I will read "critical-values
  branch" as jargon.
- **"Signed-root log-likelihood ratio"** (§10 final paragraph).
  Cited to Schweder–Hjort 2016 ch.~5 but not defined. For a
  statistician this is a standard term (Barndorff-Nielsen 1986);
  for an astronomer it is opaque. The intended audience for this
  sentence is unclear: if it is the IRT/SBI crowd, the bridge could
  be sharper; if it is the SH/CD crowd, the framing already lands.
- **"Implicit-likelihood-ratio tradition"** (§10 final paragraph).
  This term is *coined* in §10 to denote the Cranmer et al. 2015 line
  of work. It is not standard usage. "Implicit likelihood" is
  standard SBI terminology; "implicit-likelihood-ratio tradition"
  reads as if it should be — but it is the manuscript's own label.
  Either acknowledge that this is the manuscript's name for it, or
  use the more conventional "likelihood-ratio surrogate methods" or
  "CARL line" (per round 2's note about high-energy-physics usage).

### 2. Oracle-style passages

- **§9 step 1 (architecture-choice table).** The format "If truth is
  approximately additive → use additive flow" presumes the reader can
  recognize *whether `r*` is approximately additive* in practice. For
  the location-normal example everyone learns first this is obvious
  (`r* = θ − X` is literally additive). For a black-box simulator
  with no closed-form `r*`, this is a chicken-and-egg problem: the
  whole reason to use CD-SBI is that `r*` is not in closed form, but
  the recipe asks the reader to guess `r*`'s structural form before
  training. The fourth row ("no prior knowledge → doubly-monotone
  autoregressive") is the round-1 hedge; it now reads as the *real*
  recommendation, with the first three rows being shortcuts for
  practitioners who can guess. **The pedagogy gap is that this
  framing is implicit.** A short paragraph before the table —
  something like "If you suspect an additive structure from physical
  reasoning (e.g. an additive-noise observation model with no scale
  parameters), use rows 1–3 as shortcuts; if you do not, default to
  row 4 and let the doubly-monotone form discover the structure" —
  would walk the reader through this decision rather than presenting
  it as oracle output.
- **§9 step 2 (architectural monotonicity).** Round 1's fix is well
  placed: the prose now explains the §3.5 mechanism (folding → loss
  unbounded below) in one sentence and points at the §8.4 empirical
  smoking gun. **This is one of the better-pedagogically-pitched
  passages in Part VI.** Still missing: a one-sentence reminder of
  what "autograd-only Jacobians" *would do* in practice (try to use
  PyTorch / JAX autograd on a generic MLP and read off the determinant
  of the Jacobian as the change-of-variables factor). For a reader
  who has never tried this, the failure mode is too abstract.
- **§9 step 5 (validation).** The Hermans-pattern sentence —
  "marginal PIT passes but conditional fails" — assumes the reader
  knows what that looks like in practice. §7.3 explains the
  diagnostic decomposition (marginalizing out θ preserves uniformity,
  so the conditional PIT is strictly more informative). §9 step 5
  could borrow a half-sentence from §7.3 explaining *why* the
  decomposition matters: marginal PIT can pass while a particular θ₀
  is systematically over- or under-covered.
- **§10 paragraph after the table.** "Closest cousins are LF2I and
  WALDO ... and SNL." The *closeness* is asserted ("which target
  test-statistic CDFs to construct confidence sets via Neyman
  inversion") but for a reader who does not already know LF2I, the
  one-clause description does not establish *similarity* — it
  describes LF2I's mechanism in a vacuum, then asserts that CD-SBI
  is similar. **Fix candidate:** rewrite the sentence to make the
  similarity-and-difference axes explicit. e.g.: "Like LF2I and WALDO,
  CD-SBI delivers confidence sets with frequentist coverage; unlike
  them, the calibration target is enforced by the loss + architecture
  at training time rather than by a separate Monte-Carlo critical-
  values estimation step at inference time." This is one sentence
  longer but tells the reader where the similarity is.
- **§10 IRT paragraph (signed-root, score-function bridge).** Round 1's
  sign-convention fix is in place and the prose now reads
  *defensively* — "with the sign chosen to match the (R1) convention
  that ∂_θ r > 0" — which is technically correct but pedagogically
  reads like a footnote that got promoted. The bridge to Schweder–Hjort
  ch.~5 is asserted ("connects to the signed-root log-likelihood
  ratio") with no explanation of what that connection looks like. A
  reader who does not already know the signed-root LR has nothing to
  hold onto. **Fix candidate:** either drop the signed-root reference
  (the paragraph already works as a score-function bridge alone) or
  add a half-sentence: "the signed-root log-LR, `sgn(θ̂ − θ) ·
  √(−2 log Λ(θ))`, is asymptotically standard normal under θ in
  regular exponential families — the same calibration target
  CD-SBI enforces by construction."

### 3. Proof-role clarity

Not theorem-heavy, but the §10 table is the *de facto* "where does
CD-SBI fit" theorem: the reader looks at the rightmost column
(Calibration) and the targets column and gets the comparative claim
distilled. For that purpose:

- **The table works.** The "Confidence distribution / Pointwise, by
  construction" entry for CD-SBI is sharp and lands the differentiator.
- **The "what does it buy?" question is less clearly answered.** The
  paragraph after the table says "CD-SBI is single-stage" and that
  "the calibration is enforced in the population limit by the
  architecture + loss combination rather than by a hold-out calibration
  sample." This is the operational distinction. What it does *not*
  say plainly is: (a) for problems where LF2I/WALDO work, CD-SBI is
  expected to deliver comparable confidence sets at lower compute
  (no critical-values Monte Carlo at inference); (b) for problems
  where SNL works, CD-SBI gives you frequentist coverage instead of
  posterior-with-prior; (c) the price is the architectural
  constraints (R1 + R2). The "what does the architectural recipe
  buy you in practice" framing is implicit. **Fix candidate:** add
  a short paragraph after the IRT bridge, or replace the IRT bridge
  with such a paragraph, summarizing the practical trade-off in one
  place.

### 4. Concept-introduction gaps

- **§9 starts cold.** "A practitioner's checklist for applying CD-SBI
  to a new model:" is the only preamble. The reader is dropped into
  the architecture-choice table with no recap of *what* the
  architectural class is, why monotonicity matters, or which §s the
  recipe distills. **A 3–5-sentence "tl;dr of Parts II–IV" would help.**
  Something like: "The CD-SBI framework requires three architectural
  commitments: (a) NF-MLE loss (§3); (b) monotone-in-θ pivot, enforced
  by UMNN integration (§4, §6.1); (c) monotone-in-X pivot, enforced
  by additive structure or triangular flow with UMNN heads (§3.5,
  §6.1). The recipe below assumes these commitments and walks through
  the implementation decisions a practitioner faces in a new problem."
  This would also make §9 readable as a standalone reference once
  the manuscript is in print.
- **§10 also starts cold.** No introduction to the table; the reader
  is dropped into the nine-row comparison. A one-sentence preamble
  ("The table below positions CD-SBI against the existing SBI
  methods catalogued in §1.1, with target object and calibration
  mechanism as the comparison axes.") would frame the table.

### 5. Narrative flow

- **§9 → §10 order.** The current order (practitioner checklist
  first, then literature positioning) works for a reader who has
  decided to *use* CD-SBI and wants the implementation recipe before
  reading more historical context. The reverse order (literature
  first, then recipe) would work better for a reader who is still
  deciding whether CD-SBI applies to their problem.
- **For the astronomer audience** (per CLAUDE.md's tone note), the
  literature-first ordering is probably more natural: astronomers
  unfamiliar with the SBI methods landscape want to know "where this
  fits" before reading "how to run it." A statistician audience
  might prefer §9 first.
- **Pragmatic compromise:** keep the current order (§9 → §10) but
  add a one-line forward reference at the top of §9 — "Readers
  unfamiliar with the SBI methods landscape may want to skim §10
  first." This is cheap, preserves the current narrative for
  practitioners, and gives the audience an out.

---

## Prioritized recommendations

1. **(High) Add architectural-recipe preamble to §9.** A 3–5-sentence
   summary of the (a) NF-MLE loss, (b) monotone-in-θ, (c) monotone-in-X
   prescription at the top of §9 — restating what the recipe assumes
   from Parts II–IV. This unlocks §9 as a standalone reference and
   removes the "drop into the table cold" feeling. *Astronomer-
   accessibility priority.*
2. **(High) Add one-sentence glosses to the §9 step 1 architecture
   table.** Each architecture name ("additive flow," "triangular
   flow," "doubly-monotone UMNN") should have a half-sentence
   description in the table cell, not just a §-pointer. The table
   currently rewards readers who have memorized §6.1's prescriptions
   and is opaque to readers who have not. *Highest leverage for
   astronomer accessibility.*
3. **(High) Reframe §9 step 1 to acknowledge the "structure of `r*`
   is unknown" reality.** Add a one-paragraph preamble before the
   architecture table explaining that rows 1–3 are shortcuts for
   practitioners with physical-reasoning evidence about `r*`'s
   form, and row 4 is the default for the general case. The current
   framing reads as if the reader can always identify `r*`'s
   structure, which contradicts the framework's motivation (use
   CD-SBI when `r*` is not in closed form).
4. **(Medium) Add a "what does CD-SBI buy you in practice" paragraph
   to §10.** Either replace the IRT/signed-root sentence (which
   currently reads defensively) with this, or add it after. Concrete
   trade-off statements — "vs LF2I/WALDO: comparable confidence sets
   at lower inference compute; vs SNL: frequentist coverage instead
   of posterior + prior" — would deliver the comparative claim more
   plainly than the table alone.
5. **(Medium) Soften the IRT/signed-root paragraph.** The round-1
   sign-convention fix is correct and necessary but the resulting
   prose reads as a footnote-promoted-to-paragraph. Either drop the
   signed-root reference (the score-function bridge stands alone) or
   add a half-sentence definition of the signed-root LR (per the
   accessibility note above).
6. **(Medium) §10 table preamble.** A one-sentence framing at the top
   of §10 — "this table positions CD-SBI against the SBI methods
   surveyed in §1.1" — helps a reader who arrives at §10 directly
   from the table of contents.
7. **(Low) §9 step 5: borrow a sentence from §7.3.** Explain why
   marginal-PIT-passes-conditional-fails matters in practice — that
   marginal PIT can hide systematic over- or under-coverage at
   particular θ₀.
8. **(Low) Forward reference at top of §9 to §10.** "Readers
   unfamiliar with the SBI methods landscape may want to skim §10
   first." Low-cost, helps the astronomer audience.
9. **(Low) Glossary footnote for "implicit-likelihood-ratio
   tradition."** Either acknowledge that the term is the manuscript's
   label or use the more standard "CARL / likelihood-ratio surrogate
   methods."

---

## Existing strengths to preserve

- **§9 step 2 is well pitched.** The round-1 fix that explicitly
  separates "theory does the heavy lifting" from "single-experiment
  smoking gun" reads honestly and pedagogically. This is the
  template other steps could aim at.
- **§9's six-step structure is the right format.** Numbered checklist,
  one step per architectural commitment / training decision /
  inference decision. Do not let any expansion break this structure.
- **§10's table is dense but correct.** Round 2 verified all twelve
  citations and the round-1 Balanced-NRE target-column fix is in
  place. The table's value is precisely that it is a one-glance
  comparison; expansions should be parenthetical, not row-replacing.
- **The "single-stage vs critical-values branch" distinction lands.**
  This is the most concrete differentiator from LF2I/WALDO and is
  stated cleanly in the post-table paragraph.
- **The CD-SBI "Pointwise, by construction" row is the rhetorical
  payoff.** Preserve the bold weight in the LaTeX; it is what the
  table is for.
- **The §9 step 6 inversion algorithm is concrete.** "1D root-finding
  (autoregressively, in `d > 1`)" is exactly the level of detail a
  practitioner needs without reproducing §6.4.
