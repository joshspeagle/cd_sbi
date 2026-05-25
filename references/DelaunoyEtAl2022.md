# Delaunoy, Hermans, Rozet, Wehenkel, Louppe, 2022 — Balanced Neural Ratio Estimation

**Authors:** Arnaud Delaunoy, Joeri Hermans, François Rozet, Antoine Wehenkel, Gilles Louppe
**Year:** 2022
**Venue:** Advances in Neural Information Processing Systems 35 (NeurIPS 2022)
**arXiv:** 2208.13624
**DOI:** —

## One-paragraph summary
Introduces Balanced Neural Ratio Estimation (BNRE), a modification of
NRE that adds a "balancing" regularizer to the binary classification
loss to enforce that the average classifier output is `1/2` on
calibration samples. The motivation is the Hermans et al. 2022
trust-crisis pattern: standard NRE produces overconfident posteriors.
BNRE encourages the *posterior approximations* obtained from the
learned ratio to be more conservative — i.e., wider credible regions
that more reliably cover the truth — while retaining the same
Bayes-optimal solution as NRE.

## Why CD-SBI cites it
- §1.1 lists BNRE among the "retrofit" calibration methods
  ("conservative regularization").
- §3.7 puts BNRE on the same shelf as Calibrated NPE — adding
  regularization terms to address conservativeness post hoc.
- §10 comparison-table row: target = "Likelihood ratio (goal: more
  reliable posteriors)", calibration = "Conservative regularizer".

## Specific anchors
- The "balanced" condition: the classifier's average output on the
  marginal distribution should equal 1/2.
- The trained object is still a likelihood-ratio estimator; the
  conservativeness applies to the *downstream* posterior obtained
  from the ratio via Bayes's rule.

## Notes
- **Attribution-check OK.** Round-1 Part VI review had already
  corrected "Posterior" → "Likelihood ratio (goal: more reliable
  posteriors)" in the §10 table; the abstract explicitly says BNRE
  produces "posterior approximations that tend to be more conservative".
  The trained object is the ratio (since BNRE extends NRE); the
  downstream object that becomes conservative is the posterior. The
  parenthetical in the §10 row captures this distinction.
- "More reliable posteriors" is the paper's stated goal but should
  not be read as the *training target*.
