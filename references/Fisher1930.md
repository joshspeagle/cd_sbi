# Fisher, 1930 — Inverse probability

**Authors:** R. A. Fisher
**Year:** 1930
**Venue:** Proceedings of the Cambridge Philosophical Society, vol. 26, no. 4, pp. 528–535
  (the journal was renamed *Mathematical Proceedings of the Cambridge Philosophical Society* in 1975; pre-1975 citations conventionally use the older title)
**arXiv:** n/a
**DOI:** 10.1017/S0305004100016297

## One-paragraph summary
Fisher's foundational paper introducing the *fiducial argument*. He argues against the routine Bayesian use of "inverse probability" with uniform / Bayes–Laplace priors and proposes that, in continuous one-parameter problems with a sufficient statistic, one can invert the sampling distribution of the statistic to obtain a *fiducial distribution* over the parameter — a distribution-valued summary of evidence that requires no prior. The location-normal mean is the worked example, and the construction gives the textbook Student-style intervals as quantile intervals of the fiducial distribution.

## Why CD-SBI cites it
Historical anchor in §1.2 for the lineage of confidence distributions. Claim ID **D-CD (historical)**: the modern CD theory traces back to Fisher's fiducial argument as the first proposal that a parameter could have a data-driven distribution constructed without a prior.

## Specific anchors
- The construction itself, around pp. 530–532: invert the sampling distribution of a pivot to a parameter-distribution.
- The location-normal example is the cleanest case; the manuscript's §1.2 explicit `Φ(θ − X)` CD is exactly Fisher's fiducial distribution for the normal mean (modulo sign convention).

## Notes
- Fiducial inference is famously contested — Fisher's argument breaks down outside the regular one-parameter case, which is precisely why the modern CD literature (Schweder–Hjort, Xie–Singh) restated the calibration property as the defining requirement and stepped away from the prior-free "fiducial probability" interpretation.
- The CD-SBI manuscript correctly cites Fisher as historical lineage, not as a technical foundation.
- Bibkey journal name: the bib entry uses "Proceedings of the Cambridge Philosophical Society" which is correct for 1930 — the "Mathematical" prefix was added in 1975.
