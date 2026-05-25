# Masserano, Dorigo, Izbicki, Kuusela, Lee, 2023 — Simulator-Based Inference with WALDO

**Authors:** Luca Masserano, Tommaso Dorigo, Rafael Izbicki, Mikael Kuusela, Ann B. Lee
**Year:** 2023
**Venue:** International Conference on Artificial Intelligence and Statistics (AISTATS 2023); PMLR vol. 206, pp. 2960–2974
**arXiv:** 2205.15680
**DOI:** —

## One-paragraph summary
Introduces WALDO, a method for constructing frequentist confidence
regions in simulator-based inference. The key idea: take *any*
prediction algorithm or posterior estimator (e.g., a DNN regressor or
a normalizing flow posterior), construct a Wald-style test statistic
out of the prediction's mean and (estimated) variance, then perform
Neyman inversion to obtain confidence regions with the right
conditional coverage. WALDO sits within the LF2I framework (same
research group) but uses a Wald-statistic-based test rather than the
likelihood-ratio test of ACORE. The earlier arXiv-v1 title was
"Simulation-Based Inference with Waldo: Perfectly Calibrated Confidence
Regions Using Any Prediction or Posterior Estimation Algorithm"; the
published AISTATS version uses the shorter title.

## Why CD-SBI cites it
- §1.1 "Neyman-construction wrappers" — retrofit calibration family.
- §10 comparison-table row: target = "Confidence sets", calibration =
  "Neyman inversion on Wald-style stat".

## Specific anchors
- Wald statistic `W(θ) = (θ̂ − θ)² / Var(θ̂)` (or the multivariate
  Mahalanobis form) is the test stat.
- Critical values learned via quantile regression, as in LF2I.

## Notes
- **Attribution-check OK.** AISTATS 2023 venue confirmed (PMLR vol.
  206, pp. 2960–2974). The §10 row correctly identifies the target
  (confidence sets) and the calibration mechanism (Neyman inversion).
- Conceptually a sibling of LF2I/ACORE — the §10 prose grouping
  ("closest cousins are LF2I and WALDO") is accurate.
