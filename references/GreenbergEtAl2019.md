# Greenberg, Nonnenmacher, Macke, 2019 — Automatic Posterior Transformation for Likelihood-Free Inference

**Authors:** David S. Greenberg, Marcel Nonnenmacher, Jakob H. Macke
**Year:** 2019
**Venue:** International Conference on Machine Learning (ICML 2019); PMLR vol. 97, pp. 2404–2414
**arXiv:** 1905.07488
**DOI:** —

## One-paragraph summary
Introduces Automatic Posterior Transformation (APT), a.k.a. SNPE-C, the
third major variant of Sequential Neural Posterior Estimation. APT
allows arbitrary, dynamically updated proposal distributions and is
compatible with flow-based density estimators (e.g.\ MAF, NSF) for
posterior estimation. Compared to SNPE-A (Papamakarios–Murray 2016) and
SNPE-B (Lueckmann et al. 2017), APT removes the importance-weighting
correction that limited SNPE-A and avoids the unstable proposal
inversion of SNPE-B, becoming the standard sequential variant of NPE in
modern SBI toolkits.

## Why CD-SBI cites it
- §1.1 NPE/SNPE row in the dominant-families list (paired with
  Papamakarios & Murray 2016).
- §3.7 ("Position in the SBI literature") classifies NPE/SNPE as a
  Class-5 log-density loss on the posterior.
- §10 comparison-table row for "NPE / SNPE", target = "Bayesian
  posterior", calibration = "None built-in".

## Specific anchors
- The APT objective is the proposal-corrected log-density of the
  current posterior estimator under the simulated `θ | X` samples.
- Compatibility with normalizing flows makes APT the modern face of
  the NPE family.

## Notes
- **Attribution-check OK.** APT = SNPE-C in standard SBI nomenclature
  (Lueckmann et al. 2021 benchmark uses both labels). The manuscript's
  §1.1 / §3.7 / §10 treatment is internally consistent.
- The conservativeness/overconfidence pattern that Hermans et al. 2022
  documented affects APT specifically; this is the empirical
  motivation behind both Balanced NRE and Calibrated NPE.
