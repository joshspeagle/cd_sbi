# Round 2 — Part VI (Practice & position) Lit-Review

## Summary

- Bibkeys verified: 12 / 12.
- Verdicts: 11 ✓, 1 ⚠ (PatelEtAl2023 — arXiv 2023 vs ICML 2024
  conference version: BibTeX could be upgraded to `@inproceedings`).
- No attribution errors of substance found. The §10 comparison-table
  rows accurately characterize each method's target and calibration
  mechanism. The round-1 Part VI fix of the Balanced NRE row from
  "Posterior" to "Likelihood ratio (goal: more reliable posteriors)"
  is correct against the source (paper trains a ratio estimator; the
  downstream posterior is what becomes more conservative).
- The §10 closing paragraph's score-function bridge to the implicit-LR
  tradition (Cranmer et al. 2015) is accurate; the sign convention
  matches (R1).
- Minor citation-style notes are surfaced below; none require Part VI
  prose changes.

## Per-citation findings

### PapamakariosMurray2016
**Status:** ✓
**Verified:** [arXiv:1605.06376](https://arxiv.org/abs/1605.06376); NeurIPS 2016 proceedings.
**Attribution check:** Foundational NPE / "fast ε-free" paper, often relabeled SNPE-A in the post-Lueckmann taxonomy. The manuscript's §1.1 (NPE/SNPE row), §3.7 (Class-5 on posterior), and §10 (target = Bayesian posterior, calibration = none) are consistent with the paper's actual contribution.
**Notes:** Authors Papamakarios + Murray (two-author). Manuscript correctly does not conflate with SNPE-B (Lueckmann et al. 2017).

### GreenbergEtAl2019
**Status:** ✓
**Verified:** [arXiv:1905.07488](https://arxiv.org/abs/1905.07488); ICML 2019, [PMLR v97](http://proceedings.mlr.press/v97/greenberg19a.html), pp. 2404–2414.
**Attribution check:** APT = SNPE-C in the standard naming. §1.1 / §3.7 / §10 are consistent. Three authors confirmed: Greenberg, Nonnenmacher, Macke.
**Notes:** None.

### PapamakariosEtAl2019
**Status:** ✓
**Verified:** [arXiv:1805.07226](https://arxiv.org/abs/1805.07226); AISTATS 2019, PMLR v89.
**Attribution check:** SNL paper by Papamakarios, Sterratt, Murray. The manuscript's §3.4 "NF-MLE is the SNL loss" identification is correct: SNL's training objective is conditional log-density on `X | θ`, identical to NF-MLE modulo the architectural class.
**Notes:** None.

### HermansEtAl2020
**Status:** ✓
**Verified:** [arXiv:1903.04057](https://arxiv.org/abs/1903.04057); ICML 2020 (v5 explicitly notes camera-ready). Three authors: Hermans, Begy, Louppe.
**Attribution check:** AALR / NRE foundational paper. §10 row target = likelihood ratio, calibration = none built-in — accurate.
**Notes:** None.

### MillerEtAl2022
**Status:** ✓
**Verified:** [arXiv:2210.06170](https://arxiv.org/abs/2210.06170); NeurIPS 2022. Three authors confirmed: Miller, Weniger, Forré (with accent: Patrick Forré, UvA).
**Attribution check:** Contrastive NRE. Manuscript's §1.1 and §10 group it with HermansEtAl2020 under "NRE / SNRE" — appropriate granularity.
**Notes:** None.

### DelaunoyEtAl2022
**Status:** ✓
**Verified:** [arXiv:2208.13624](https://arxiv.org/abs/2208.13624); [NeurIPS 2022 proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7e6288bfb68182db7d6e328b0aefa89a-Abstract-Conference.html). Five authors confirmed.
**Attribution check:** Round-1 fix from "Posterior" to "Likelihood ratio (goal: more reliable posteriors)" is correct. Paper extends NRE (ratio estimator), but the stated *goal* is more reliable (conservative) posterior approximations. The §10 row's parenthetical captures this distinction precisely.
**Notes:** None.

### FalkiewiczEtAl2023
**Status:** ✓
**Verified:** [arXiv:2310.13402](https://arxiv.org/abs/2310.13402); [NeurIPS 2023 proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/03a9a9c1e15850439653bb971a4ad4b3-Abstract-Conference.html). Seven authors confirmed: Falkiewicz, Takeishi, Shekhzadeh, Wehenkel, Delaunoy, Louppe, Kalousis.
**Attribution check:** Calibrated NPE; differentiable coverage relaxation. §10 row "Differentiable coverage" is accurate.
**Notes:** None.

### DalmassoEtAl2024
**Status:** ✓
**Verified:** [arXiv:2107.03920](https://arxiv.org/abs/2107.03920); Electronic Journal of Statistics 18(2):5045–5090 (2024); DOI [10.1214/24-EJS2307](https://doi.org/10.1214/24-EJS2307). Five authors: Dalmasso, Masserano, Zhao, Izbicki, Lee.
**Attribution check:** The consolidated EJS 2024 paper is the right reference for the unified LF2I framework. ACORE / BFF are specific test-statistic variants bundled under LF2I. The §10 row "LF2I (ACORE, BFF) / Confidence sets / Neyman construction on learned test stat" is accurate.
**Notes:** The bibkey is named with year 2024 (matching the EJS publication) but the BibTeX `note` field correctly records the arXiv version. Bib file should add the DOI: `doi = {10.1214/24-EJS2307}`. (Citation-quality improvement, not an error.)

### MasseranoEtAl2023
**Status:** ✓
**Verified:** [arXiv:2205.15680](https://arxiv.org/abs/2205.15680); AISTATS 2023, [PMLR v206](https://proceedings.mlr.press/v206/masserano23a.html), pp. 2960–2974. Five authors: Masserano, Dorigo, Izbicki, Kuusela, Lee.
**Attribution check:** WALDO. §10 row "Confidence sets / Neyman inversion on Wald-style stat" — accurate. The published title is shorter than the v1 arXiv title; the BibTeX uses the published title, which is correct.
**Notes:** Bib `note = {arXiv:2205.15680}` is the right cross-reference.

### BortolatoVentura2025
**Status:** ✓
**Verified:** [arXiv:2502.11072](https://arxiv.org/abs/2502.11072); first posted 16 February 2025 (v1), revised through January 2026. Two authors: Bortolato, Ventura (Padova).
**Attribution check:** Box CD. §10 row "Confidence sets / Depth-based, hyper-rectangles" — accurate to the paper's stated method.
**Notes:** No peer-reviewed venue yet; arXiv-only is appropriate.

### PatelEtAl2023
**Status:** ⚠ (year/venue update available)
**Verified:** [arXiv:2305.14275](https://arxiv.org/abs/2305.14275) (preprint May 2023); CANVI was published at [ICML 2024, PMLR v235](https://proceedings.mlr.press/v235/patel24a.html). Five authors: Patel, McNamara, Loper, Regier, Tewari.
**Attribution check:** Manuscript's §10 row characterization "Conformal post-hoc" is correct — CANVI literally stands for *Conformalized* Amortized Neural Variational Inference. The calibration mechanism is split conformal prediction on candidate variational approximators, with selection by predictive efficiency.
**Notes:** The bibkey `PatelEtAl2023` matches the arXiv year; if Round 2 wants to upgrade to the published version, it should become:
```
@inproceedings{PatelEtAl2023,
  author    = {Patel, Y. and McNamara, D. and Loper, J. and Regier, J. and Tewari, A.},
  title     = {Variational inference with coverage guarantees in simulation-based inference},
  booktitle = {International Conference on Machine Learning ({ICML})},
  volume    = {235},
  year      = {2024},
  note      = {arXiv:2305.14275}
}
```
Whether to rename the bibkey to `PatelEtAl2024` is a stylistic call; keeping `2023` is acceptable if you want the arXiv year. Minor; not blocking.

### CranmerEtAl2015
**Status:** ✓
**Verified:** [arXiv:1506.02169](https://arxiv.org/abs/1506.02169) (submitted June 2015, revised March 2016). Three authors: Cranmer, Pavez, Louppe. No formal peer-reviewed venue; the arXiv version is the standard citation.
**Attribution check:** The §10 closing paragraph correctly attributes the implicit-LR tradition to this paper and draws an accurate bridge to the location-normal score function (with the sign convention matching (R1)).
**Notes:** None.

## §10 comparison-table row check

| Method | Target column | Calibration column | Citation | Verdict |
|---|---|---|---|---|
| NPE / SNPE | Bayesian posterior | None built-in | PapamakariosMurray2016; GreenbergEtAl2019 | ✓ |
| NLE / SNL | Likelihood | None built-in | PapamakariosEtAl2019 | ✓ |
| NRE / SNRE | Likelihood ratio | None built-in | HermansEtAl2020; MillerEtAl2022 | ✓ |
| Balanced NRE | Likelihood ratio (goal: more reliable posteriors) | Conservative regularizer | DelaunoyEtAl2022 | ✓ |
| Calibrated NPE | Posterior | Differentiable coverage | FalkiewiczEtAl2023 | ✓ |
| LF2I (ACORE, BFF) | Confidence sets | Neyman construction on learned test stat | DalmassoEtAl2024 | ✓ |
| WALDO | Confidence sets | Neyman inversion on Wald-style stat | MasseranoEtAl2023 | ✓ |
| Box CD | Confidence sets | Depth-based, hyper-rectangles | BortolatoVentura2025 | ✓ |
| Variational SBI w/ coverage | Posterior | Conformal post-hoc | PatelEtAl2023 | ✓ |
| **CD-SBI (this work)** | **Confidence distribution** | **Pointwise, by construction** | — | ✓ |

All nine non-CD-SBI rows pass. No misattributions.

## Citation gaps

None that materially affect Part VI. Two optional additions:

1. **NRE-B / Durkan et al. 2020** ("On contrastive learning for
   likelihood-free inference", ICML 2020) is the intermediate step
   between AALR (NRE-A; HermansEtAl2020) and Contrastive NRE
   (NRE-C; MillerEtAl2022). Currently neither §1.1 nor §10 cites it.
   *Not blocking* — the §10 NRE/SNRE row already bundles two
   citations and a third would not change the table's message — but
   could be added for completeness if Round 2 wants finer-grained
   NRE-family attribution.

2. **Dalmasso et al. 2020 ACORE** (ICML 2020, arXiv:2002.10399) is
   the original ACORE paper, separate from the 2024 EJS consolidated
   LF2I paper. The current §10 LF2I row label parenthesizes "(ACORE,
   BFF)" but only cites the 2024 paper, which is acceptable because
   the EJS paper does cover both. If finer attribution is desired,
   ACORE could be cited separately.

Both are minor and not required to land Round 2.

## Historical-context additions

- **The score-function / signed-root-LR bridge.** The §10 closing
  paragraph already draws this bridge. A useful additional anchor
  (one sentence) would be a pointer to the Schweder–Hjort discussion
  of CDs derived from signed-root LR in regular exponential families
  (already cited as `\citep[ch.~5]{SchwederHjort2016}`). The current
  text is accurate; the only improvement would be a single sentence
  acknowledging that in regular exponential families this connects to
  Cornish–Fisher / saddlepoint approximations of the LR — but that is
  scope creep, not a gap.

- **Marginal vs pointwise coverage distinction.** Worth flagging in
  §10 — or in the §1.1 retrofit-methods list — that CANVI
  (PatelEtAl2023) guarantees *marginal* coverage (averaged over the
  proposal) via conformal exchangeability, in contrast to the
  *pointwise* coverage CD-SBI targets. The current table conveys
  this implicitly via the "Posterior" vs "Confidence distribution"
  target labels, but an explicit footnote could prevent confusion.
  Optional.

- **CARL terminology.** The CranmerEtAl2015 framework is sometimes
  called "CARL" (Calibrated Approximate Ratio Likelihood) in the
  high-energy physics community. The manuscript does not use the
  CARL label, which is fine; if Round 3 wants to add a parenthetical
  for HEP readers, that would be a stylistic improvement, not a
  factual fix.

## Cross-Part flags (heads-up for other Round 2 agents)

- `DalmassoEtAl2024` is cited in §1.1 (Part I), §3.7 (Part II), §7.3
  (Part V), §10 (Part VI). All Part VI uses are consistent with the
  framework-level (not ACORE-specific) characterization. Part II and
  Part V agents should confirm their §3.7 and §7.3 uses are also
  framework-level.
- `HermansEtAl2022` (not owned by Part VI) is cited in §9 step 5;
  this is a cross-link Part I and Part VI both touch. The §9 prose
  ("the kind of conditional miscalibration that Hermans et al.
  documented") is consistent with how Part I (§1.1) frames the
  trust-crisis paper. No conflict.
