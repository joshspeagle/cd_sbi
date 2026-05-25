# Miller, Weniger, Forré, 2022 — Contrastive Neural Ratio Estimation

**Authors:** Benjamin Kurt Miller, Christoph Weniger, Patrick Forré
**Year:** 2022
**Venue:** Advances in Neural Information Processing Systems 35 (NeurIPS 2022)
**arXiv:** 2210.06170
**DOI:** —

## One-paragraph summary
Generalizes the binary-classification training of NRE to a multi-class
contrastive setup: given a joint sample `(θ_0, X)` and `K-1` independent
"negative" samples `θ_k ~ p(θ)`, the classifier learns to identify the
joint pair among `K` candidates, recovering an estimate of the
likelihood ratio that is theoretically equivalent to NRE in the
binary limit but yields lower-variance gradients and better empirical
performance. Often referred to as NRE-C in the SBI toolkits.

## Why CD-SBI cites it
- §1.1 NRE/SNRE row in the dominant-families list (paired with Hermans
  et al. 2020).
- §10 comparison-table row for "NRE / SNRE", target = "Likelihood
  ratio", calibration = "None built-in".

## Specific anchors
- Multi-class contrastive objective generalizes the AALR binary loss.
- Provides the modern (post-2020) reference point for NRE variants.

## Notes
- **Attribution-check OK.** Authors confirmed as Miller, Weniger,
  Forré (Patrick Forré at University of Amsterdam). NeurIPS 2022
  acceptance confirmed via arXiv page and OpenReview.
- Sometimes called NRE-C; the original AALR (HermansEtAl2020) is
  then NRE-A and Durkan et al. 2020 ("On contrastive learning for
  likelihood-free inference") is NRE-B. The manuscript's §10 row does
  not distinguish, which is fine for the comparison table's level of
  granularity.
