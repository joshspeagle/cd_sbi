# Full audit — CPF spec/plan, postmortem conclusions, and research strategy

**Date:** 2026-06-10
**Scope:** everything produced in the postmortem-and-redesign effort: the postmortem conclusions
(both repos), the Conformal Pivot Flow spec + M0–M1a plan, the experimental design, and the
strategic plan-of-attack. **Method:** a self-audit over the full working record, plus four
independent external verification probes (citation audit; TRUST/CP4SBI full-text deep-read;
LF2I-Score novelty closure incl. HEP/econometrics; adversarial red-team of the planned "map paper"
thesis).

**Bottom line:** the *empirical* record and the *citation* base are solid (no hallucinated
references; all 32 load-bearing citations verified). But the audit found **one real technical error
propagated through the committed spec** (the Foygel–Barber conditioning conflation), **one
mischaracterization** (lumping TRUST and CP4SBI as a single "validity machinery" when they condition
on opposite variables), **three bug-grade gaps in the plan** (conformal quantile convention, GATE-1
test power, M1a conditioner ambiguity), and **two strategy-level reversals** (the map paper as
framed would be desk-rejected; LF2I-Score moves to the top of the priority queue on a race risk).
Plus one discovery only the PI can adjudicate (FreB co-authorship). Corrections to the spec/plan are
applied in the same commit series as this document.

---

## 1. The four audit questions, answered

**Q1 — are we clear-eyed about what we want to accomplish?** Partially. The audit surfaced real
goal drift: the program moved from "improve LF2I's power" → "a synthesis method (CPF) whose honest
default is worked-example status" → "a map paper" — while the threads with the strongest claims to
priority (LF2I-Score, which is *built and swept*; the tail-θ₀/rare-object problem, which is the
*science driver*) drifted to the periphery. §5 gives the corrected priority picture; §6 lists the
decisions that need the PI, not drift.

**Q2 — do we understand why the original projects didn't work?** Yes — this part of the record
held up under every probe. The theorem ledger (PKD; MLR/UMP-existence; Dufour; Davies/Drton;
calibration≠sharpness) verified cleanly, including the exact Schweder–Hjort chapter structure and
the verbatim LF2I quote. Four minor citation corrections only (§3.1).

**Q3 — is the plan-of-attack clear?** It was clear but partly mis-founded: the map-paper framing
fails red-team review (§3.4), and the CPF positioning needed the TRUST/CP4SBI split (§3.2). The
corrected plan-of-attack is in §5.

**Q4 — is the experimental design well-justified?** The M0→M1a ladder structure is sound
(closed-form first; machinery isolated from findings; negative controls). Three concrete defects
found and fixed (§4): the conformal quantile convention was ambiguous-to-wrong, GATE 1 thresholds
had no sample sizes, and M1a's conditioner was unpinned (which silently changed what M1a
demonstrates).

---

## 2. Errors found in our own artifacts (the headline of the audit)

### 2.1 The Foygel–Barber conditioning conflation (technical error, was in the committed spec)

The spec (and the working discussion throughout) cited Barber–Candès–Ramdas–Tibshirani as proving
"finite-sample *conditional* coverage is impossible," and used it as the ceiling on **per-θ₀**
coverage. **That theorem is about coverage conditional on the test point X (the data).** Per-θ₀
coverage is a different object, and in SBI it is **not theorem-limited**: the simulator yields the
conditional law at any θ₀ on demand, so exact per-θ₀ finite-sample coverage is achievable
*pointwise* by brute Monte Carlo. The binding constraint on an **amortized** procedure is
**simulation budget / estimation error** (cf. TRUST's per-θ₀ guarantee, which is asymptotic in
budget B) — and the calibration proposal over θ is the *analyst's choice*, so tail-θ₀ coverage is
purchasable with targeted budget. Corrected in spec §3.6 and §6. Consequences:
- the honest ceiling statement becomes "finite-sample local-marginal-under-π ⊕
  asymptotic-**in-budget** per-θ₀ conditional";
- the "tail-validity" claim (audit finding A1) weakens from impossibility-flavored to an **empirical
  budget-allocation trade-off** — which is still real, unquantified in the literature, and the
  defensible core of any future paper on it;
- the planned map paper's "axis-1 bounded by Foygel–Barber" framing is wrong as stated.

### 2.2 TRUST vs CP4SBI: not one machinery (mischaracterization, was in the committed spec)

Full-text deep-read (the R2 reviewer had explicitly flagged this as unverified):
- **TRUST/TRUST++** (TMLR; Cabezas, Soares, Ramos, Stern, Izbicki): conditions on the
  **hypothesized θ** — `H(τ₀|θ) = P(τ(X,θ) ≤ τ₀ | θ)`, calibration pairs regressed *on θ_b* —
  a genuine amortized **Neyman construction**. Mechanism: a **regression tree** (TRUST) /
  random-forest **proximity partition** (TRUST++) with leaf-wise conformal quantiles — *not*
  quantile regression. Guarantees: finite-sample coverage **marginal over the proposal r(θ) within
  each leaf**; per-θ₀ only asymptotic in B. Its estimated CDF *implicitly* yields a nested all-α
  family (a p-value function) — unnamed and unexploited in the paper.
- **CP4SBI** (Phil Trans A): conditions on the **observed x** — `F̂(s(θ;x)|x)` with θ from the
  *posterior*; x-space partitions; guarantee is local **Bayesian** coverage (joint over (θ,X),
  local in x). It does **not** target per-θ₀ frequentist coverage at all.

The same lab built one object on each side of exactly the θ-vs-x axis our spec calls fatal —
CP4SBI is, for our purposes, the *published instantiation of the M0 negative control*. Spec
corrected: **TRUST alone is the validity ancestor**; CP4SBI is a category contrast, not an
ancestor; the "coherent all-α family" increment is **shrunk** (TRUST has it implicitly; CPF's
increment is exploiting/normalizing/sampling it). CPF's surviving daylight, verified against the
full texts and the newest lineage follow-ups: **(i) the learned flow-pivot as the statistic itself
(absent everywhere), (ii) the confidence distribution as a normalized, sampleable object (no paper
in the lineage produces one), (iii) the frozen-sampler firewall.**

### 2.3 Bug-grade gaps in the plan (all fixed in this commit series)

1. **Conformal quantile convention** — the pure-math test said "`q̂_α = ⌈α(n+1)⌉/n`," ambiguous
   about which tail; applied to the score `s = F̂(T)` (large = extreme) that reads as the α-lower
   quantile → ~α coverage instead of 1−α. The classic conformal off-by-(1−α) bug. Fixed: pinned to
   the ⌈(1−α)(n+1)⌉-th smallest score, with a worked numeric example required in the test.
2. **GATE 1 had unpowered thresholds** — "coverage_error_max ≲ 0.03" with no θ₀-grid/n_per_theta,
   "KS ≤ 0.05" with no n (KS criticals scale 1/√n). We had demanded exactly this rigor of the leak
   audit and not of our own gate. Fixed: pre-registered grid, n_per_theta, and n_PIT.
3. **M1a conditioner unpinned** — if M1a uses the oracle `SufficientStatConditioner` (it should),
   it certifies the calibration machinery and the no-`Î` question **only**; the lead finding
   (insufficiency → width cost, not validity break) lives entirely in M1b's insufficient-summary
   arm. Fixed: pinned and stated.

---

## 3. External verification outcomes

### 3.1 Citation audit — CLEAN
All 32 load-bearing citations exist and match their claimed substance; every "suspect" recent arXiv
ID is real. No hallucinations anywhere in the chain. Minor corrections to carry: Grünwald & van
Ommen 2017 is *Bayesian Analysis* (not JRSS-B); misspecification-robust SNL is Kelly–Nott–Frazier–
Warne–Drovandi (no "Ward"); arXiv:2511.15146 is single-author (Ndiaye); credit Vovk 2012 / Lei &
Wasserman 2014 for the basic conditional-coverage impossibility with Barber et al. as the
sharpening.

### 3.2 TRUST/CP4SBI deep-read — see §2.2. Plus the discovery:
**FreB** (arXiv:2508.02602, MLST 2026; "Trustworthy scientific inference with generative models" —
Carzon, Masserano, …, **Speagle**, Izbicki, Lee) appears to have the PI as a co-author. If so, the
entire "competing with the Lee/Izbicki lab from outside" premise of our novelty analysis is
miscalibrated: the PI is *inside* the lineage. This is the single most important open question for
the PI (§6.1).

### 3.3 LF2I-Score novelty — GRAY-ZONE, leaning SLOT-OPEN, with a race clock
- The precise claim survives: *"a new statistic in the LF2I family — the first to use the
  structural-parameter score of an amortized neural likelihood as the Neyman-inverted statistic
  with finite-sample learned critical values."*
- The broad claim does not: **EMM / score-based indirect inference (Gallant–Tauchen 1996;
  Gourieroux–Monfort–Renault 1993)** is conceptual prior art ("score of an estimated flexible
  model + simulated calibration") and must be ceded explicitly, with the auxiliary-vs-structural-
  score distinction spelled out. Also mandatory: SALLY/MadMiner + optimal observables
  (score-as-summary, never score-as-Neyman-statistic — verified, incl. ATLAS NSBI which inverts a
  learned LR, not a score); Sui–Pandey–Wandelt 2025 and **Jiang–Wang–Yang 2026** (learned-score
  Wald/bootstrap UQ — no inversion, no learned c_α(θ)); Dufour Monte Carlo tests.
- **Race risk:** Jiang–Wang–Yang (Mar 2026) is one Rao-inversion follow-up away from the slot.
  The method is already implemented and swept in this repo (`methods/score_cd.py`, the 160-run
  §8.1–8.4 sweep + (μ,σ²)/(μ,Σ)); the missing artifact is the write-up. **Priority implication:
  the LF2I-Score paper is the highest-value lowest-cost item on the board and should move first.**

### 3.4 Map-thesis red-team — NOT VIABLE AS FRAMED; rebuilt core identified
- The triad (validity/sufficiency/power) is **a lens, not a decomposition**: misspecification is a
  missing axis (the field's current frontier); estimation error is non-orthogonal (the SLCP wall is
  an estimation failure that surfaces on whichever axis the budget starves); and by our own
  "insufficiency costs only power" argument, sufficiency is a sub-budget of power. The tail-validity
  refinement is an instance of the classical relevant-subsets/conditionality question
  (Fisher/Buehler) — a real fourth dimension the triad has no slot for.
- Framing novelty is thin: LF2I's three-branch architecture *is* the validity/power/diagnostics
  split; TRUST's abstract *is* the "honest ceiling"; calibration≠information is in Hermans et al.
  (coverage + EIG); **the title concept is taken** ("Coverage is not enough," Alokda–Porciani–
  Eggemeier, arXiv:2605.00980 — in an astronomy application).
- **The rebuilt, ownable core** (red-team's verdict: true, new, useful): (a) the **θ-conditional vs
  X-conditional precision** — correcting a common conflation we ourselves made, perfectly
  instantiated by the TRUST/CP4SBI contrast; (b) the **tail-θ₀ calibration-budget study** —
  quantified coverage-error-per-unit-budget across the family, counter-methods (Mondrian,
  importance-weighted CP, TRUST++) given a fair fight — which is also the **direct bridge to the
  rare-galaxy science driver**; (c) the **calibration≠sufficiency collapse mechanism** (the
  Spearman-0.014 result; information-preservation as the requirement) — already solid in this
  repo's Stage-B/N3 record; with the triad demoted to organizing scaffolding.
- The map paper's would-be empirical base also needs hardening regardless: the neural-copula probe
  numbers (SLCP 0.115/0.426; Cauchy 0.022; d_θ=10) are single-seed prototypes by the notes' own
  admission and must be replicated before appearing in any paper.

---

## 4. Corrections applied in this commit series

**Spec** (`2026-06-07-conformal-pivot-flow-design.md`): §3.6 + §6 Foygel–Barber rewrite
(θ-vs-X conditioning; budget-limited not theorem-limited; ceiling restated); §2 header + prior-art
table split TRUST (validity ancestor; trees/proximity, not QR; finite-sample r(θ)-marginal-within-
leaf; implicit nested all-α family) from CP4SBI (x-conditional Bayesian local calibration — the
published instantiation of our negative control); §4.3 ancestry corrected; coherence increment
shrunk accordingly.

**Plan** (`2026-06-07-cpf-m0-m1a.md`): Step 4 conformal quantile convention pinned (exact order
statistic + worked example); Step 9 GATE 1 powers pinned (θ₀-grid, n_per_theta, n_PIT); M1a
conditioner pinned to the oracle with an explicit statement of what M1a does and does not
demonstrate.

**Deferred (need PI input or new work):** repositioning CPF/M1b baselines in light of FreB (§6.1);
the prototype-hardening experiment list (§6.4); the LF2I-Score write-up (§5).

---

## 5. The corrected strategic picture (priority-ordered)

1. **LF2I-Score write-up — first.** Built, swept, novelty-verified (gray-zone leaning open with
   precise scoping + mandatory EMM/SALLY citations), and on a race clock. Weeks of writing, not
   months of building. Re-run the arXiv check at submission.
2. **The rebuilt empirical paper** — tail-θ₀ calibration-budget study + calibration≠sufficiency
   mechanism + the θ-vs-X conditioning clarification, triad as scaffolding only. Connects directly
   to the rare-object science. Requires the prototype-hardening experiments (§6.4) and the budget-
   study design (new).
3. **CPF** — survives with reshaped daylight (flow-pivot statistic; sampleable CD; firewall;
   credit TRUST). M0/M1a remain valid as machinery certification. Whether it proceeds as a
   standalone method, a worked example, or a contribution coordinated within the FreB collaboration
   is gated on §6.1 — do not start M1b (the external-baseline builds) before that decision.
4. **Bet B / non-identifiability honesty** — partially subsumed: the tail-budget study covers the
   "weakest where it matters" theme; the Dufour/unbounded-set frontier remains open and unclaimed.

## 6. Decisions that need the PI (not drift)

> **RESOLVED (PI, 2026-06-10).** FreB co-authorship confirmed; the PI is in normal coordination
> with the Lee group and **leads this work independently**. Consequences: the "novelty threat"
> framing is fully retired (coordination is routine, not a strategic fork); LF2I-Score is PI-led
> with group interest (draft v0.1 at `docs/papers/lf2i-score-draft.md`); TRUST++ baselines will be
> implemented **independently** (reference code exists but is not used — M1b/M2 effort estimates
> stand as budgeted); the tail-budget study is a research option judged on merit, not a
> coordination item; the cd_sbi_v7 §10 "developed separately" phrasing is a one-line edit at
> publication time. Net: all five items below dissolve into ordinary execution. **Process note:**
> the audit itself over-weighted these — converting work into coordination questions is a form of
> the very drift §1/Q1 criticizes. The original list is retained below for the record.

1. **FreB.** Is the co-authorship real, and how does that collaboration's roadmap relate to CPF
   and to the rebuilt paper? This single answer re-prices every novelty/positioning judgment in
   this audit.
2. **LF2I-Score authorship/venue** and whether to coordinate it with the Lee-group lineage
   (given FreB) or keep it separate.
3. **CPF's GATE-2 question, re-posed**: standalone / worked example / collaboration contribution.
4. **The hardening list**: approve replicating the single-seed probe results (SLCP wall with NSF
   and MAF; Cauchy loc-scale; d_θ=10) at proper seeds/budgets before any paper uses them.
5. **The conjugate-sandbox lesson, applied forward**: the tail-budget study should be designed on
   targets where the tail behavior is *checkable* (closed-form tails first, then the d=5 target's
   documented extreme-θ₀ regime) — same closed-form-first discipline as M0.

## 7. Process lessons (what this audit says about how we worked)

- **Single-agent characterizations of load-bearing papers are insufficient.** The TRUST
  characterization drove a full spec repositioning from one abstract-level read; the deep-read
  changed it materially. Rule: any paper that moves a spec gets a full-text read before the spec
  lands.
- **Impossibility theorems must be checked for their exact conditioning variable / setting** before
  being used as ceilings. The Foygel–Barber slip survived two spec review rounds because every
  reviewer inherited the framing from the prompt. Independent probes must be given the *claim*,
  not our gloss of it.
- **Audit the thesis, not just the method.** The map-paper framing accumulated for days without
  the adversarial treatment the method got within hours.
- **Track test power everywhere a threshold appears.** We caught it in the leak audit and missed
  it in GATE 1 in the same document.
- What worked and should be kept: closed-form-first milestones; negative controls with
  pre-registered detection; the claims-ledger-by-verification-depth habit; citation audits before
  writing (zero hallucinations found, but four small corrections that would have embarrassed).
