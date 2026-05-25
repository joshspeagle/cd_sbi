# Cranmer, Pavez, Louppe, 2015 — Approximating Likelihood Ratios with Calibrated Discriminative Classifiers

**Authors:** Kyle Cranmer, Juan Pavez, Gilles Louppe
**Year:** 2015
**Venue:** arXiv preprint (stat.AP), 1506.02169
**arXiv:** 1506.02169
**DOI:** —

## One-paragraph summary
The originating paper of the implicit-likelihood-ratio tradition in
SBI. Shows that a binary classifier trained to discriminate samples
from `P_θ_1` versus `P_θ_0` recovers (after calibration) the
likelihood ratio `p(X | θ_1) / p(X | θ_0)`, by the standard density-
ratio identity. The ratio can then be used in classical
hypothesis-testing / confidence-set construction even when the
individual likelihoods are intractable. Predates and motivates the
modern NRE family (Hermans et al. 2020, Miller et al. 2022). Within
high-energy physics (the original target domain), the framework is
known as CARL / "calibrated classifier-based likelihood ratio".

## Why CD-SBI cites it
- §10 prose (last paragraph): "The framework also relates to the
  implicit-likelihood-ratio tradition `\citep{CranmerEtAl2015}`."
- Provides the historical hinge from binary classification to LR
  estimation that the §3.7 / §10 NRE discussions rely on.

## Specific anchors
- Density-ratio trick: a classifier's optimal logit equals the log
  ratio of conditional densities up to the class-prior offset.
- Calibration of the classifier output as an additional post-training
  step to convert the classifier into a usable LR estimator.

## Notes
- **Attribution-check OK.** Three authors confirmed: Cranmer, Pavez,
  Louppe. arXiv 1506.02169 confirmed (submitted June 2015, revised
  March 2016). No formal venue; this is the "stat.AP" arXiv version
  that has become the standard citation. The manuscript correctly
  cites it as `CranmerEtAl2015`.
- The connection drawn in §10 — "for the location-normal model the
  score `∂_θ log p(X | θ) = X − θ`...up to sign, `r*` is the
  standardized score" — is correct and provides a useful
  bridge to the LR tradition. Worth verifying that the
  signed-root-LR connection in regular exponential families (cited
  in §10 via `[ch. 5]{SchwederHjort2016}`) is indeed in that
  chapter; this is the historical-context anchor for the §10 closing
  paragraph.
