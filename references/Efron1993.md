# Efron, 1993 — Bayes and likelihood calculations from confidence intervals

**Authors:** B. Efron
**Year:** 1993
**Venue:** Biometrika, vol. 80, no. 1, pp. 3–26
**arXiv:** n/a
**DOI:** 10.1093/biomet/80.1.3

## One-paragraph summary
Efron shows that a one-parameter family of confidence intervals, viewed as a function from confidence level to interval endpoints, implicitly defines a distribution over the parameter — the *implied confidence distribution* — and that the corresponding implied likelihood and implied prior can be read off this distribution. The paper develops the construction for one-parameter exponential families with particular attention to accurate-confidence (BC, BCa, ABC) intervals and shows how this yields a *frequentist* posterior-like object whose intervals match the original CIs by construction. This is one of the key papers that re-legitimized the distribution-of-a-parameter idea on frequentist (rather than fiducial) footing.

## Why CD-SBI cites it
Historical lineage in §1.2 (claim ID **D-CD historical**). Specifically: Efron 1993 is the paper that frames a confidence distribution as the inversion of a one-parameter family of confidence intervals, which is essentially the construction the CD-SBI manuscript uses in §1.2 when it writes `H(θ; X)` and identifies its α-quantile with the upper endpoint of an α-confidence interval.

## Specific anchors
- §2 and §3 of Efron 1993: the implied-CD-from-CIs construction.
- The construction is exactly the definitional viewpoint adopted in CD-SBI §1.2's "What a confidence distribution is" paragraph.

## Notes
- Efron's framing here is what later became the Singh–Xie–Strawderman / Schweder–Hjort CD definition. The CD-SBI §1.2 lineage Fisher → Cox → Efron → Schweder–Hjort is historically accurate.
- The paper does not use the modern term "confidence distribution" consistently; it talks about "implied confidence distributions" and "implied likelihoods". Schweder & Hjort later canonicalized the terminology.
