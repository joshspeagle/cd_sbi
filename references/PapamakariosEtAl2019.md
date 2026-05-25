# Papamakarios, Sterratt, Murray, 2019 — Sequential Neural Likelihood

**Authors:** George Papamakarios, David C. Sterratt, Iain Murray
**Year:** 2019
**Venue:** International Conference on Artificial Intelligence and Statistics (AISTATS 2019); PMLR vol. 89
**arXiv:** 1805.07226
**DOI:** —

## One-paragraph summary
Introduces Sequential Neural Likelihood (SNL): train a normalizing-flow
density estimator on `X | θ` from simulated pairs, then run MCMC on the
learned surrogate likelihood to obtain a posterior. The training
objective is per-sample maximum likelihood — the same NF-MLE loss CD-SBI
uses, applied to the (X, θ) conditional rather than to a calibrated
pivot. Sequential refinement of the proposal toward the current
posterior estimate keeps the method tractable when the prior is
diffuse.

## Why CD-SBI cites it
- §1.1 NLE/SNL row in the dominant-families list.
- §3.4 "The Bayesian connection: NF-MLE is the SNL loss" — explicitly
  identifies NF-MLE and SNL as the same loss applied to different
  architectures.
- §3.7 ("Position in the SBI literature") classifies SNL as a Class-5
  log-density loss on the likelihood; the difference from CD-SBI is
  the architectural prescription (monotone in θ + monotone in X) and
  the chi-square pivot inversion rather than MCMC.
- §10 comparison-table row for "NLE / SNL", target = "Likelihood",
  calibration = "None built-in".

## Specific anchors
- The SNL objective is the conditional log-density
  `E_(θ, X)[ -log q_φ(X | θ) ]`, identical to NF-MLE up to the
  architectural class constraint.
- The "MCMC on learned surrogate" inference step is what CD-SBI
  replaces with direct chi-square pivot inversion (§6.4).

## Notes
- **Attribution-check OK.** The §10 row, §3.4 identification of the
  loss, and §3.7 positioning are all consistent. SNL is correctly
  described as targeting the likelihood with no built-in coverage.
- This is the closest "cousin" method in the literature; the
  manuscript's §10 prose ("SNL: same NF-MLE loss, but with a generic
  flow architecture rather than the monotone-in-θ + monotone-in-X
  prescription") accurately characterizes the relationship.
