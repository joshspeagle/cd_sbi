# Papamakarios & Murray, 2016 — Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation

**Authors:** George Papamakarios, Iain Murray
**Year:** 2016
**Venue:** Advances in Neural Information Processing Systems 29 (NeurIPS / NIPS 2016)
**arXiv:** 1605.06376
**DOI:** —

## One-paragraph summary
Introduces a method for likelihood-free Bayesian inference that trains a
conditional density estimator (mixture density network) on simulated
`(θ, X)` pairs to approximate the posterior `p(θ | X_obs)` directly,
avoiding the rejection / kernel mechanics of classical ABC ("ε-free").
The proposal distribution is iteratively refined toward the current
posterior estimate, giving the first sequential variant of what is now
called Neural Posterior Estimation. This is the foundational paper of
the NPE / SNPE family — later renamed **SNPE-A** in the
Lueckmann (2017) / Greenberg (2019) taxonomy.

## Why CD-SBI cites it
- §1.1 NPE/SNPE row in the dominant-families list (paired with Greenberg
  et al. 2019).
- §3.7 ("Position in the SBI literature") classifies NPE/SNPE under
  Class 5 (per-sample log-density) but on the posterior, with no built-
  in coverage.
- §10 comparison-table row for "NPE / SNPE", target = "Bayesian
  posterior", calibration = "None built-in".

## Specific anchors
- Sequential proposal updates (their Algorithm 1) prefigure the
  proposal-distribution mechanics later used in SNL and APT.
- The mixture-density-network parameterization predates the
  flow-based estimators (NSF, MAF) now used in modern SBI.
- The "ε-free" framing is the historical hinge between ABC and modern
  amortized neural SBI.

## Notes
- **Attribution-check OK.** Manuscript references match: the paper does
  estimate the *posterior* with no built-in frequentist calibration.
- The "NPE vs SNPE-A" naming distinction matters only when contrasted
  with SNPE-B (Lueckmann et al. 2017) and SNPE-C/APT (Greenberg et al.
  2019). The manuscript's §1.1 and §10 bracket both papers
  (Papamakarios–Murray 2016 *and* Greenberg et al. 2019) under
  "NPE/SNPE", which is consistent with how the recent SBI literature
  uses the label.
