# Round 2 — Part V (Empirical validation) Lit-Review

**Scope:** §7 (architecture, training, diagnostics) and §8 (the four
experiments). Manuscript lines ~1559–2064 of `cd_sbi_v7.tex`.
**Bibkeys owned:** `TaltsEtAl2018`, `LemosEtAl2023`.
**Cross-Part:** `DalmassoEtAl2024` (LF2I) is also cited at §7.3 but is
owned by Part II (see citation inventory). `WehenkelLouppe2019` (UMNN)
is cited at §7.1 but is owned by Part II as well.

## Summary

Both Part V citations (`TaltsEtAl2018`, `LemosEtAl2023`) have correct
metadata and are characterized accurately at §7.3 — SBC and TARP are
indeed related-but-distinct diagnostic frameworks, and the manuscript's
positioning (closer to LF2I than TARP because CD-SBI's coverage check is
local/pointwise rather than marginal) is correct. SBC and TARP are both
fundamentally Bayesian-posterior diagnostics, but the rank/coverage
machinery generalizes to CDs with the caveat that for CDs the natural
statement is pointwise-in-`θ₀`, which is what CD-SBI's Diagnostic 3
(conditional PIT) tests. One minor citation gap is flagged: §7.3
invokes the Kolmogorov limiting distribution and gives the numerical
quantile `1.628`, but the original Kolmogorov 1933 / Smirnov 1944 papers
or a standard textbook (Lehmann–Romano, *Testing Statistical
Hypotheses*) are not cited; this is a textbook fact and citing it would
be unusual, but adding a Lehmann–Romano cite would be consistent with
how the manuscript cites other textbook facts (e.g., Schweder–Hjort,
Gneiting–Raftery). No factual corrections required; one stylistic
suggestion in §7.3.

## Per-citation findings

### `TaltsEtAl2018` — Talts, Betancourt, Simpson, Vehtari, Gelman (2018)

- **Metadata verified.** Authors, title, year, arXiv 1804.06788 all
  correct. Note: never formally published in a refereed venue (still an
  arXiv preprint as of 2026, with a v2 revision in October 2020). The
  BibTeX entry's `journal = "arXiv preprint arXiv:1804.06788"` field
  is fine but could be cleaned to `@misc` style if the .bib gets
  normalized — stylistic, not a fix.
- **Attribution verified.** §7.3 calls SBC a "related diagnostic
  framework" — accurate. SBC's rank-uniformity check is
  semantically close to CD-SBI's Diagnostic 2 (marginal PIT).
- **Bayesian vs CD generalization (specific scrutiny item).** SBC's
  rank-uniformity identity is derived from the **Bayesian joint**
  `π(θ) p(X | θ)`: the rank of `θ₀` among samples from a sampler
  targeting `π̂(θ | X)` is uniform marginally iff `π̂ = π`. The
  rank-statistic machinery is in principle agnostic to whether `θ̂` is
  Bayesian-posterior or CD-sourced (both are probability measures on
  the parameter space), but Talts et al. **do not** discuss CDs and
  the identity as stated is Bayesian. The CD analog is pointwise in
  `θ₀` (because `r(θ₀; X) | θ₀ ~ N(0, I_d)` by construction), which
  is precisely Diagnostic 3 (conditional PIT) in the manuscript. So
  the §7.3 framing — that SBC is *related* and not equivalent — is
  correct. The manuscript could (optionally) add one sentence noting
  that SBC's rank machinery is the **marginal** analog of Diagnostic
  2, while Diagnostic 3 is the **conditional** analog, but this is a
  presentation choice; no factual issue.

### `LemosEtAl2023` — Lemos, Coogan, Hezaveh, Perreault-Levasseur (2023)

- **Metadata verified.** Authors, year, ICML 2023 venue, arXiv
  2302.03026 all correct. Paper appears in PMLR vol. 202 (the ICML
  2023 proceedings).
- **Attribution verified.** §7.3 calls TARP a "related diagnostic
  framework" alongside SBC and LF2I, then states CD-SBI's coverage
  diagnostic is "closest in spirit to the LF2I local-coverage check."
  This is correct: TARP is a **marginal** coverage test (expectation
  over the proposal / prior); LF2I and CD-SBI's Diagnostic 5 are
  **local** (pointwise in `θ₀`).
- **"More directly relevant?" (specific scrutiny item).** No.
  TARP's main contribution is a necessary-and-sufficient sample-only
  coverage check for **general posterior estimators**, and it does
  improve on SBC in the sense that SBC's rank uniformity is only
  necessary (a sampler can have uniform ranks but miscalibrated
  posterior shape). However, TARP remains a **marginal** test in the
  CD-SBI sense: the expected empirical coverage is averaged over draws
  of `θ₀`. CD-SBI promises pointwise frequentist coverage at every
  `θ₀`, which is strictly stronger than TARP's marginal coverage; so
  the closer comparator is LF2I, exactly as the manuscript says. The
  brief mention is accurate.
- **Optional enrichment (not required).** If §7.3 were to expand the
  positioning sentence, the natural additions would be: (i) TARP is
  necessary-and-sufficient where SBC is only necessary; (ii) TARP is
  sample-only (no density evaluation), which CD-SBI's pivot-based
  `C_α` also is by construction; (iii) TARP averages over the
  proposal, CD-SBI/LF2I check pointwise. None of these are needed for
  the current §7.3 sentence to be correct.

## Citation gaps

### Gap 1 — Kolmogorov / Smirnov / KS-test reference (§7.3)

§7.3 lines ~1650–1668 state:

> Under the null, `\sqrt{N} \cdot KS` has the Kolmogorov limiting
> distribution with `Pr(\sqrt{N}\cdot KS > 1.628) = 0.01`.

This invokes Kolmogorov's 1933 limiting distribution result without a
citation. The standard references would be:

- Kolmogorov (1933), *Sulla determinazione empirica di una legge di
  distribuzione*, Giornale dell'Istituto Italiano degli Attuari, 4,
  83–91. (Original limiting distribution.)
- Smirnov (1944), *Approximate laws of distribution of random variables
  from empirical data*, Uspekhi Mat. Nauk, 10, 179–206. (Critical
  values.)
- Or, more practically and more in-line with the manuscript's existing
  textbook-citation habits: **Lehmann & Romano (2005), *Testing
  Statistical Hypotheses* (3rd ed., Springer), §14.2** for the KS
  limit distribution treatment.

**Recommendation:** Optional. The Kolmogorov limit is genuinely
textbook material; the manuscript does not cite a textbook for, e.g.,
the χ² distribution either. But the manuscript **does** cite
`GneitingRaftery2007` for strict propriety (also a textbook-level
definition) and `SchwederHjort2016` for CD background, so the bar for
"this is too basic to cite" is not set very high. Adding
**Lehmann–Romano (2005)** as a `\citep` at the start of the
"Statistical noise floor for KS" paragraph would be consistent with the
manuscript's existing citation density and would document the source
of the `1.628` quantile. Low priority; flag rather than mandate.

### Gap 2 — "edge effects near the support boundary" (§8.1)

§8.1 lines ~1762–1769 state:

> The conditional PIT degrades near `θ₀ = ±7` because the trained `r`
> is a poorer approximation there: those points are at the support
> boundary of the training proposal `ρ`, where the network sees fewer
> samples and the universal-approximation error is largest.

This is presented as a self-evident finite-sample / training-proposal
effect, not as a known result requiring citation. The explanation is
correct on its face (it is a property of how the trained network
allocates capacity over the proposal support). No standard SBI-paper
cite is obviously called for; the closest related literature would be
the general boundary-effect / extrapolation-failure discussion in NF
literature (e.g., `PapamakariosEtAl2021`'s NF review), but the
manuscript doesn't lean on that framing. **No action recommended.**

### Gap 3 — KS test interpretation generally (§8.1–8.4)

Each experiment table uses KS statistics and reports `p`-values
("`p > 0.1`", "`p = 0.23`"). These are standard KS interpretations and
do not need new citations beyond the same Kolmogorov reference flagged
in Gap 1 — adding one Lehmann–Romano cite at §7.3 would suffice as the
implicit reference for all four experiments. **Subsumed by Gap 1.**

## Historical-context additions

None proposed. SBC (Talts et al.) and TARP (Lemos et al.) are the two
correct contemporary SBI-calibration-diagnostic citations. Earlier
predecessors (e.g., Cook–Gelman–Rubin 2006, *Validation of software for
Bayesian models using posterior quantiles*, JCGS — the original
posterior-quantile uniformity check that Talts et al. generalize) could
be mentioned for completeness, but it is reasonable to point the
reader at Talts as the canonical modern reference. **No action
recommended** unless the manuscript decides to extend §7.3 into a more
expansive comparison.

## Cross-references

- `DalmassoEtAl2024` (LF2I) is cited at §7.3 in the same sentence as
  TARP/SBC. The Part II agent owns this bibkey. Cross-check that the
  Part II agent confirms: the LF2I conditional-coverage check is local
  in `θ₀`, matching the manuscript's "closest in spirit" framing.
- `WehenkelLouppe2019` (UMNN) is cited at §7.1 (architecture). Part II
  agent owns. Cross-check that the §7.1 UMNN definition is consistent
  with the original paper's UMNN definition.

## Files written

- `references/TaltsEtAl2018.md`
- `references/LemosEtAl2023.md`
- `reviews/round2/part5_empirical.md` (this file)
