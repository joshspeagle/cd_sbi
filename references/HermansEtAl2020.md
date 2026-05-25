# Hermans, Begy, Louppe, 2020 — Likelihood-Free MCMC with Amortized Approximate Ratio Estimators

**Authors:** Joeri Hermans, Volodimir Begy, Gilles Louppe
**Year:** 2020 (arXiv preprint 2019)
**Venue:** International Conference on Machine Learning (ICML 2020); PMLR vol. 119
**arXiv:** 1903.04057
**DOI:** —

## One-paragraph summary
Introduces Amortized Approximate Ratio Estimators (AALR), the foundational
paper of the Neural Ratio Estimation (NRE) family for SBI. A binary
classifier is trained to discriminate joint `(θ, X) ~ p(θ, X)` from
marginal `(θ, X) ~ p(θ)p(X)` simulations; by the standard density-ratio
identity, the optimal classifier's logit recovers the likelihood ratio,
which can then be used to drive MCMC for posterior inference. The
amortization makes the trained classifier reusable across different
observations.

## Why CD-SBI cites it
- §1.1 NRE/SNRE row in the dominant-families list (paired with Miller
  et al. 2022).
- §3.7 ("Position in the SBI literature") classifies NRE as a
  Class-5-flavored loss on a different object (the ratio).
- §10 comparison-table row for "NRE / SNRE", target = "Likelihood
  ratio", calibration = "None built-in".

## Specific anchors
- The binary cross-entropy classifier objective is the canonical
  Class-5 (per-sample log-density) loss for the density-ratio object,
  by the standard equivalence between classification and ratio
  estimation.
- MCMC on the learned ratio is the inference step.

## Notes
- **Attribution-check OK.** The §10 row matches the paper's target: the
  learned quantity is the likelihood ratio; calibration is not built
  into the training objective.
- Hermans is also the senior author of the "trust-crisis"
  (HermansEtAl2022) paper that documents the overconfidence pattern;
  Balanced NRE (Delaunoy et al. 2022) is a direct response.
