# Round 2 — Part III (1D theory) Lit-Review

## Summary

- **Bibkeys verified:** 2 (`Lancaster1961`, `HwangYang2001`).
- **Verdicts:** both verified, both correctly attributed in the
  sense that the cited papers exist with the bibliographic metadata
  in `cd_sbi.bib` and the cited papers do support the manuscript's
  claims. Two soft flags:
  - `Lancaster1961` is cited for the randomized-PIT construction
    `V = F(t^-) + U · p(t)`, but priority for the *precise* formula
    sits with Stevens 1950 (Biometrika 37:117–129) and Tocher 1950
    (Biometrika 37:130–144). Lancaster's own contribution in this
    paper is the mid-p value; the randomized PIT is reviewed rather
    than introduced.
  - `HwangYang2001` is cited as the "optimality theory of mid-p in
    the closely related contingency-table setting" — this is
    accurate, and the manuscript's hedge ("closely related") is
    appropriate: H&Y prove optimality for 2×2 tables, not for the
    general `Bin(n, θ)` setting that §5.7.3 invokes.
- **Citation gaps:** 3 candidates identified (one mild, two
  optional). See "Citation gaps" below.

## Per-citation findings

### Lancaster1961

**Status:** ✓ (with a soft attribution flag).
**Verified:**
- JSTOR record: https://www.jstor.org/stable/2282247
- Semantic Scholar: https://www.semanticscholar.org/paper/659ba737ff635725e4116ce9ed84035d166c37f7
- Cross-referenced in Hwang & Yang 2001 (their reference list, p. 825).

**Attribution check:**
Lancaster 1961 is JASA 56(294):223–234 — metadata in `cd_sbi.bib` is
correct.

The manuscript at §5.7.1 cites Lancaster for "auxiliary randomization:
smear each atom across an interval of an auxiliary uniform variable."
Lancaster does describe and discuss the construction, but the priority
in the secondary literature consistently goes to:
- **Stevens 1950**, "Fiducial limits of the parameter of a discontinuous
  distribution," Biometrika 37:117–129 — the artificial-data construction
  `Z = Y + V` with `V ~ U(0, 1)` independent (mathematically identical
  to `V_θ(X, U) = F_θ(T(X)^-) + U · p_θ(T(X))`).
- **Tocher 1950**, "Extension of the Neyman–Pearson theory of tests to
  discontinuous variates," Biometrika 37:130–144 — parallel
  testing-theoretic development.

The Hannig–Iyer–Lai–Lee 2016 *JASA* GFI review credits Stevens 1950 for
the randomized-confidence construction. Hwang & Yang 2001 (the
companion citation here) credit **Lancaster 1961** specifically for the
**mid-p value**, not for the randomized PIT — they explicitly say "the
mid p-value was proposed first by Lancaster (1961)." This split
attribution (Stevens/Tocher for randomized PIT; Lancaster for mid-p)
is the standard convention.

**Notes:**
- Recommended remediation (optional, low priority): either add a
  Stevens 1950 citation alongside Lancaster in §5.7.1
  (`\citep{Stevens1950, Lancaster1961}`), or add a parenthetical
  footnote noting Stevens' priority for the precise formula. Lancaster
  remains a legitimate secondary reference and would still be cited
  for the mid-p at §5.7.3.
- Paper note written at `references/Lancaster1961.md`.

### HwangYang2001

**Status:** ✓
**Verified:**
- Statistica Sinica record: https://www3.stat.sinica.edu.tw/statistica/j11n3/j11n313/j11n313.htm
- Open-access PDF: https://www3.stat.sinica.edu.tw/statistica/oldpdf/A11n313.pdf
- Bibliographic metadata in `cd_sbi.bib` exactly matches the
  Statistica Sinica record (vol 11, no 3, pp 807–826, year 2001;
  authors J. T. G. Hwang and M.-C. Yang).

**Attribution check:**
H&Y 2001 prove that, in the 2×2 contingency-table setting (one-sided
case, balanced binomial), the **expected p-value** under a
decision-theoretic / Neyman–Pearson framework coincides exactly with
the mid-p value. This is precisely the "optimality theory of mid-p"
that §5.7.3 invokes.

Important scope clarification: H&Y's optimality is established for
2×2 tables, not for arbitrary `Bin(n, θ)`. The §5.7.3 phrasing —
"See \citet{HwangYang2001} for the optimality theory of mid-p in the
**closely related** contingency-table setting" — correctly hedges this.

The `(1/2) sup_t p_θ(t)` Kolmogorov-distance bound in §5.7.3 is **not**
from H&Y 2001. H&Y use risk-function and type-I-error comparisons,
not Kolmogorov-distance bounds. The manuscript's bound is a direct
calculation (the manuscript even gives the achieving point), so no
external citation is needed. The asymptotic refinement
`sup_t p_θ(t) = Θ(1/√n)` for `Bin(n, θ)` is the local CLT.

**Notes:**
- No remediation needed.
- Paper note written at `references/HwangYang2001.md`.

## Citation gaps

The following claims in Part III could carry a citation but currently
don't. Severity rating: **mild** = nice to have; **optional** =
defensible either way.

1. **§4.2 Lemma 4.1 (monotone-rearrangement):** "Standard 1D monotone
   rearrangement: between two atomless probability measures on R, the
   unique monotone-increasing measure-preserving map is the
   composition of CDF and inverse CDF." Classical result with
   well-established priority chain:
   - Hardy, Littlewood, Pólya (1934), *Inequalities*, Cambridge UP —
     foundational rearrangement inequalities (Theorem 368, §10.2).
   - Modern presentations: Villani (2003), *Topics in Optimal
     Transport*, AMS, ch. 1, or Santambrogio (2015), *Optimal
     Transport for Applied Mathematicians*, ch. 2 — both treat 1D
     monotone rearrangement explicitly.
   - **Recommendation:** mild. A one-line `\citep{Villani2003}` or
     `\citep{Santambrogio2015}` would help an SBI-audience reader
     who isn't already steeped in optimal-transport literature.
     Hardy-Littlewood-Pólya is the historical reference but is
     probably too remote for the working audience.
   - **Severity:** mild.

2. **§5.1 "regular one-parameter exponential family with MLR" setup:**
   Cited uncited as a "standard setup." The natural references are:
   - Lehmann & Romano (2005), *Testing Statistical Hypotheses*, 3rd
     ed., Springer — chapters 3 (MLR) and 4 (one-parameter exponential
     families). The canonical graduate-level reference.
   - Casella & Berger (2002), *Statistical Inference*, 2nd ed.,
     Duxbury — §3.4 (exponential families), §8.3 (MLR + UMP).
   - **Recommendation:** mild. The current §5.1 paragraph defines
     MLR and the canonical form inline, so the setup is
     self-contained. But a `\citep{LehmannRomano2005}` at the end of
     the "Background, brief" paragraph would be a courtesy citation
     for readers wanting the full classical treatment.
   - **Severity:** mild.

3. **§4.5 UMP-unbiased CD definition:** The opening sentence "A
   UMP-unbiased CD is, in the classical sense, the CD whose
   associated one-sided test family at each level is uniformly most
   powerful within the class of unbiased tests" is followed by a
   citation to Schweder & Hjort 2016 (ch. 5) for the specific
   location-normal result. The general UMP-unbiased *testing*
   concept is from Lehmann (1947) / Lehmann–Romano. Currently fine
   as written — Schweder & Hjort is the right reference because they
   are the ones who explicitly lift UMP-unbiasedness from tests to
   CDs. **No action needed.**
   - **Severity:** optional / no-op.

## Historical-context additions

Two small notes worth surfacing for future revisions, but not
mandatory:

1. **Stevens 1950 / Tocher 1950 priority for the randomized PIT.**
   The CD-SBI manuscript's §5.7.1 "the standard fix [Lancaster1961]
   is auxiliary randomization" reads as if Lancaster originated the
   construction. A more historically accurate (and barely longer)
   version would be: "The standard fix \citep{Stevens1950,
   Tocher1950, Lancaster1961} is auxiliary randomization..." This
   gives the priority chain — Stevens & Tocher in 1950 introduced
   randomized confidence intervals / tests for discrete distributions
   in the same Biometrika volume; Lancaster reviewed both and added
   the deterministic mid-p alternative. This is a one-line edit if
   desired.

2. **Mid-p has a deeper bibliography than §5.7.3 currently surfaces.**
   The §5.7.3 paragraph cites only Hwang & Yang 2001. A reader who
   wants the broader picture would benefit from one or two of:
   Routledge (1994), Berry & Armitage (1995), Hirji et al. (1991),
   or the Heller (2021) "Meta-Analysis of Mid-p-Values" paper that
   establishes a convex-order bound on the mid-p null distribution
   (the latter is the closest existing paper to the §5.7.3
   Kolmogorov-distance bound). None of these are strictly needed for
   the manuscript's argument; flagged here for completeness.

(Neither addition is escalated — both are "nice to have" rather than
"must fix.")
