# Schweder & Hjort, 2016 — Confidence, Likelihood, Probability: Statistical Inference with Confidence Distributions

**Authors:** T. Schweder, N. L. Hjort
**Year:** 2016
**Venue:** Cambridge University Press (Cambridge Series in Statistical and Probabilistic Mathematics, vol. 41)
**arXiv:** n/a
**DOI:** 10.1017/CBO9781139046671
**ISBN:** 9780521861601

## One-paragraph summary
The defining modern textbook on confidence distributions. Develops the CD framework as a fully frequentist, distribution-valued summary of evidence for a parameter, without invoking fiducial probability or any Bayesian prior. Treats CDs in one-parameter and multi-parameter settings, the relation between CDs and likelihoods (the "confidence likelihood" construction), Neyman–Pearson-style optimality theory for CDs (including UMP and UMPU confidence distributions, the Schweder–Hjort optimal CDs), bootstrap CDs, exponential-family CDs, and applications in meta-analysis and structural equation modelling. The book consolidates ideas from Fisher 1930, Cox, Efron 1993/1998, Singh–Xie–Strawderman 2007, and Xie–Singh 2013 into a single canonical treatment.

## Why CD-SBI cites it
- §1.2: the *defining* reference for the CD framework — the manuscript directly quotes "Def. 3.1" of Schweder–Hjort for the two CD axioms (CDF in θ for each X, and uniform-PIT calibration under the truth). Claim ID **D-CD**.
- §4.5: cited as the source of the UMP/UMPU optimality theory for CDs that Theorem A then *recovers from* via NF-MLE in regular one-parameter families. Claim ID **P-4.5**.
- §10 (positioning vs other SBI methods): cited as the classical-statistics counterpart of an entry in the SBI–CD comparison table. Claim ID **C-10-IRT**.

## Specific anchors
- **Definition 3.1** (Chapter 3, "Confidence distributions"): the axiomatic definition with the calibration property `H(θ_0; X) ~ U(0,1)` under `X ~ P_{θ_0}`. This is precisely what CD-SBI §1.2 cites.
- **Chapter 5** ("Optimality of confidence distributions"): the UMP/UMPU optimality theorem in regular one-parameter exponential families. This is the result Theorem A in §4 recovers.
- Chapter 10 ("Likelihoods and confidence likelihoods"): the construction connecting CDs to likelihood-based summaries.

## Notes
- This is the single most important reference for CD-SBI; the framework is essentially asking "how do we estimate the Schweder–Hjort optimal CD when the likelihood is implicit?".
- The CD-SBI manuscript's calibration definition matches Def. 3.1 of Schweder–Hjort up to the choice of `Φ_d` (product Gaussian) rather than `U(0,1)^d` for the base measure — these are equivalent up to coordinate-wise Φ-transformation.
