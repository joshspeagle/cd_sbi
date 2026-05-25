# Round 2 — Part I (Framework) Lit-Review

**Scope:** §1 (Motivation: SBI and confidence distributions) and §2 (The pivot
and the calibration manifold) of `cd_sbi_v7.tex` (lines ~139–432).

**Owned bibkeys (per task assignment):**
- Primary cites in Part I: `HermansEtAl2022`, `Fisher1930`, `Cox1958`,
  `Efron1993`, `Efron1998`, `SchwederHjort2016`.
- Background `\nocite` entries: `Fraser2011`, `SinghEtAl2007`, `XieSingh2013`,
  `CranmerEtAl2020`, `Lyons2013`, `SriperumbudurEtAl2011`.

## Summary
- Bibkeys verified: 12 (6 primary + 6 background).
- Verdicts: 11 correct attribution, 1 minor wording issue (`HermansEtAl2022`),
  0 major issues.
- Citation gaps identified: 3 (each minor — promotions of `\nocite` entries
  to inline, plus an additional SBI-review pointer).

## Per-citation findings

### HermansEtAl2022
**Status:** ⚠ minor wording issue
**Verified:** Joeri Hermans, Arnaud Delaunoy, François Rozet, Antoine Wehenkel,
Volodimir Begy, Gilles Louppe — TMLR 2022 — arXiv:2110.06581. Verified
against the OpenReview forum page (`LHAbHkt6Aq`) and the arXiv abstract.
**Attribution check:** The §1.1 sentence says the paper "documented that all
four major SBI families systematically produce overconfident posteriors".
The Hermans et al. benchmark actually covers **three** families:
(S)NPE, (S)NRE, and ABC variants. NLE/SNL is *not* explicitly part of their
benchmark (though the wider community has since reproduced the overconfidence
finding for NLE too). The "four major SBI families" phrasing slightly
overstates the paper's scope.
**Notes / suggested fix:** soften "all four major SBI families" to "all
three families benchmarked (NPE, NRE, and ABC variants)" or "the main
classes of neural SBI estimators benchmarked there". Either fix is one
sentence; the headline overconfidence finding itself is correctly attributed.

### Fisher1930
**Status:** ✓ correct
**Verified:** R. A. Fisher — *Proceedings of the Cambridge Philosophical
Society*, vol. 26, no. 4, pp. 528–535, October 1930. DOI 10.1017/S0305004100016297.
Confirmed via cambridge.org.
**Attribution check:** Manuscript correctly cites Fisher 1930 as the
historical origin of fiducial inference, the proto-CD. Quotation is
appropriately weighted ("traces back through Fisher's fiducial inference") —
does not over-claim that the modern CD definition is Fisher's.
**Notes / suggested fix:** Note that the journal was renamed to *Mathematical
Proceedings of the Cambridge Philosophical Society* in 1975; the bib entry
uses the pre-1975 name, which is correct for a 1930 citation. Some modern
style guides retroactively apply the new name — either is defensible; the
current choice matches typical historical practice. The bib entry could
optionally add an issue number (4) for completeness.

### Cox1958
**Status:** ✓ correct
**Verified:** D. R. Cox — *Annals of Mathematical Statistics*, vol. 29,
no. 2, pp. 357–372, June 1958. DOI 10.1214/aoms/1177706618. Confirmed via
Project Euclid.
**Attribution check:** §1.2 cites Cox 1958 as part of the historical
lineage of CDs alongside Fisher 1930 — this is the standard pairing.
Strictly speaking Cox 1958 is about conditional inference / ancillary
statistics, not CDs per se; but the citation context ("traces back
through Fisher's fiducial inference, Cox, and Efron's papers") is
appropriately broad and does not over-claim. The §5.3 use (ancillary
statistic definition) is technically load-bearing and correct.
**Notes / suggested fix:** None. The pairing with Fisher 1930 in §1.2 is
the canonical "Fisher–Cox conditional-inference school" narrative.

### Efron1993
**Status:** ✓ correct
**Verified:** B. Efron — *Biometrika*, vol. 80, no. 1, pp. 3–26, March
1993. JSTOR 2336754, DOI 10.1093/biomet/80.1.3.
**Attribution check:** Efron 1993 introduces the "implied confidence
distribution" obtained by inverting a one-parameter family of CIs — this
is precisely the construction §1.2 paraphrases when it identifies the
α-quantile of `H(·; X)` with the upper endpoint of a one-sided
α-confidence interval. Attribution is correct.
**Notes / suggested fix:** None.

### Efron1998
**Status:** ✓ correct
**Verified:** B. Efron — *Statistical Science*, vol. 13, no. 2, pp. 95–122,
May 1998 (1996 R. A. Fisher Lecture). DOI 10.1214/ss/1028905930. Confirmed
via Project Euclid.
**Attribution check:** Cited in §1.2 as a historical retrospective on
Fisher's programme and the modern revival of fiducial-style thinking.
Standard usage. Note that Efron 1998 famously calls fiducial inference
"Fisher's biggest blunder" but also predicts its return — both points
align with how §1.2 uses the reference.
**Notes / suggested fix:** None.

### SchwederHjort2016
**Status:** ✓ correct
**Verified:** T. Schweder & N. L. Hjort — Cambridge University Press, 2016
(Cambridge Series in Statistical and Probabilistic Mathematics, vol. 41).
ISBN 9780521861601, DOI 10.1017/CBO9781139046671. Confirmed via cambridge.org
and the publisher's series page.
**Attribution check:** §1.2 cites Def. 3.1 of Schweder–Hjort directly — this
is correct (the two-axiom CD definition appears as Definition 3.1 in
Chapter 3 of the book). The manuscript's calibration definition matches
Def. 3.1 up to a change of base measure (product Gaussian rather than
product uniform), which is equivalent under the coordinate-wise Φ map.
The §4.5 and §10 invocations are also appropriate (the UMP/UMPU optimality
theorem is in Chapter 5 of the textbook).
**Notes / suggested fix:** None.

### Fraser2011 (background `\nocite`)
**Status:** ✓ correct
**Verified:** D. A. S. Fraser — *Statistical Science*, vol. 26, no. 3,
pp. 299–316 (2011, with discussion). arXiv:1112.5582, DOI 10.1214/11-STS352.
**Attribution check:** Correctly placed in the v6 reference list as
background. Fraser 2011 is the standard modern statement of the "Bayes
posterior is not generically confidence" position — directly relevant to
§1.2's bullet contrasting CDs with Bayesian posteriors.
**Notes / suggested fix:** See "Citation gaps" below — a candidate for
promotion to inline citation in §1.2.

### SinghEtAl2007 (background `\nocite`)
**Status:** ✓ correct
**Verified:** K. Singh, M. Xie, W. E. Strawderman — IMS Lecture Notes
Monograph Series, vol. 54, pp. 132–150 (2007). arXiv:0708.0976, DOI
10.1214/074921707000000102.
**Attribution check:** Correctly placed as background CD literature.
This paper is the modern proto-statement of the CD definition (immediate
precursor to Schweder–Hjort 2016).
**Notes / suggested fix:** See "Citation gaps" — candidate for promotion
to inline in §1.2 alongside Schweder–Hjort.

### XieSingh2013 (background `\nocite`)
**Status:** ✓ correct
**Verified:** M. Xie & K. Singh — *International Statistical Review*,
vol. 81, no. 1, pp. 3–39 (2013). DOI 10.1111/insr.12000.
**Attribution check:** Correctly placed as the canonical CD review article.
**Notes / suggested fix:** Same as SinghEtAl2007 — see "Citation gaps".

### CranmerEtAl2020 (background `\nocite`)
**Status:** ✓ correct
**Verified:** K. Cranmer, J. Brehmer, G. Louppe — *PNAS*, vol. 117, no. 48,
pp. 30055–30062, 2020. arXiv:1911.01429, DOI 10.1073/pnas.1912789117.
**Attribution check:** Correctly placed as background general-SBI literature.
**Notes / suggested fix:** See "Citation gaps" — candidate for promotion
to inline in §1.1's first paragraph, since the manuscript opens with a
definition of SBI and a PNAS review pointer would serve readers from
outside ML.

### Lyons2013 (background `\nocite`)
**Status:** ✓ correct
**Verified:** R. Lyons — *Annals of Probability*, vol. 41, no. 5,
pp. 3284–3305, 2013. DOI 10.1214/12-AOP803. arXiv:1106.5758.
**Attribution check:** Correctly placed as background for the §3.7
energy-distance / distance-covariance discussion (not Part I content;
flagged in inventory as Part II).
**Notes / suggested fix:** None for Part I; cross-reference Part II agent
for the §3.7 attribution check.

### SriperumbudurEtAl2011 (background `\nocite`)
**Status:** ✓ correct
**Verified:** B. K. Sriperumbudur, K. Fukumizu, G. R. G. Lanckriet —
*Journal of Machine Learning Research*, vol. 12, pp. 2389–2410, 2011.
arXiv:1003.0887.
**Attribution check:** Correctly placed as background for the §3.7
characteristic-kernel / HSIC discussion (not Part I content; flagged in
inventory as Part II).
**Notes / suggested fix:** None for Part I; cross-reference Part II agent.

## Citation gaps

Three minor gaps in Part I. All are *suggestions*, not required fixes —
the existing citations are technically sufficient, and these would
strengthen the framing for non-specialist readers.

- **§1.1 first paragraph (line ~150):** The opening definition of SBI
  ("modern simulation-based inference (SBI) trains a deep-network
  surrogate...") does not cite a general SBI review. **Suggestion:**
  add an inline `\citep{CranmerEtAl2020}` here. The Cranmer–Brehmer–Louppe
  PNAS review is already in the bib via `\nocite` — promoting it to inline
  at the SBI definition is the natural fix.

- **§1.2 "What a CD is" paragraph (line ~201):** The lineage citation chain
  Fisher → Cox → Efron → Schweder–Hjort skips Singh–Xie–Strawderman 2007
  and Xie–Singh 2013, the two papers that actually re-introduced and
  reviewed the CD concept between Efron 1998 and the Schweder–Hjort 2016
  textbook. **Suggestion:** insert `\citet{SinghEtAl2007}` (or a combined
  `\citep{SinghEtAl2007, XieSingh2013}`) between Efron 1998 and
  Schweder–Hjort 2016 in the lineage sentence. Both are already in the
  bib via `\nocite`. This is the standard CD-literature citation chain
  and would match the way Schweder–Hjort 2016 itself frames the recent
  history.

- **§1.2 Bayesian-vs-CD bullet (lines ~221–235):** The claim that a
  Bayesian posterior generally has no pointwise frequentist coverage
  guarantee is asserted but not cited. **Suggestion:** add
  `\citep{Fraser2011}` at the end of the first bullet. This is the
  canonical modern citation for that distinction and is already in the
  bib via `\nocite`.

No gaps requiring new bibliography entries — all suggested promotions
use entries already present in `cd_sbi.bib`.

## Historical-context additions (optional, for integrator)

These are pieces of background that the integrator could optionally weave
into §1.2 prose. None are required for correctness; they would add
texture for astronomer-aimed readers.

- **The "fiducial winter" framing.** Fisher's fiducial inference was
  attacked in the 1950s–60s (Lindley, Stein) and lost mainstream
  acceptance; the modern CD revival is sometimes framed as the
  vindication of the *calibration* core of the fiducial idea after
  shedding the "fiducial probability" interpretation. Efron 1998
  contains the classic "biggest blunder" remark and its prediction of
  return. A single sentence in §1.2 acknowledging this gap between the
  1930s and the modern revival would help readers from outside
  statistics.

- **CD vs structural inference.** There is a parallel programme of
  *structural inference* (Fraser, Hannig) that arrives at fiducial-style
  distributions via group-theoretic constructions. Not relevant to
  CD-SBI's core message and out of scope, but worth knowing about if a
  referee asks.

- **What §1.2's "exact coverage" condition actually buys.** Schweder &
  Hjort's Def. 3.1 imposes uniform PIT under the truth but does not
  by itself imply *optimal* (UMP/UMPU) intervals — optimality is an
  *additional* requirement that requires (R1)-style monotonicity in θ
  plus regularity (Chapter 5 of the textbook). The manuscript's §1.2
  correctly separates "calibration" (Property (ii) of Schweder–Hjort
  Def. 3.1) from "monotonicity" (R1, §2), which is consistent with the
  textbook's logical structure — worth noting that this matches the
  textbook's own division of the two roles.
