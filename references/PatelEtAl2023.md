# Patel, McNamara, Loper, Regier, Tewari, 2023 — Variational Inference with Coverage Guarantees in Simulation-Based Inference (CANVI)

**Authors:** Yash Patel, Declan McNamara, Jackson Loper, Jeffrey Regier, Ambuj Tewari
**Year:** 2023 (arXiv); 2024 (ICML conference version)
**Venue:** International Conference on Machine Learning (ICML 2024); also arXiv preprint 2023
**arXiv:** 2305.14275
**DOI:** —

## One-paragraph summary
Introduces CANVI (Conformalized Amortized Neural Variational Inference),
a method that wraps amortized variational SBI with a *split conformal
prediction* layer to provide marginal coverage guarantees on the
resulting credible / prediction regions. Multiple candidate variational
approximators are trained, each is conformalized using a held-out
calibration set, and the most "predictively efficient" conformalized
predictor is selected. Coverage is marginal (averaged over the proposal
distribution) — not pointwise frequentist — and is guaranteed by the
conformal exchangeability argument, not by the variational fitting.

## Why CD-SBI cites it
- §1.1 "variational coverage corrections" — retrofit calibration
  family.
- §10 comparison-table row: target = "Posterior", calibration =
  "Conformal post-hoc".

## Specific anchors
- Split conformal prediction layer (calibration set, predictive
  efficiency selection).
- Marginal-coverage guarantee, not pointwise conditional coverage.

## Notes
- **Attribution-check on "Conformal post-hoc" — VERIFIED.** The
  paper's method is literally CANVI = Conformalized Amortized Neural
  Variational Inference; conformal prediction is the explicit
  calibration mechanism. The §10 row characterization is accurate.
- **Year note (minor).** The arXiv preprint is 2023, but the formal
  conference publication is ICML 2024. The current bibkey
  `PatelEtAl2023` and the `arXiv preprint arXiv:2305.14275` BibTeX
  entry preserve the arXiv year. If the manuscript wants to cite the
  conference version, the entry should be updated to
  `@inproceedings{...}` with venue ICML 2024 / PMLR vol. 235. This
  is a minor citation-quality improvement, not a substantive error.
- **Marginal vs pointwise distinction.** Worth noting (perhaps in a
  footnote in §1.1 or §10) that CANVI's coverage is *marginal*
  (averaged over the proposal), not the pointwise frequentist
  coverage CD-SBI targets. The current §10 row already distinguishes
  via the "Posterior" target label vs CD-SBI's "Confidence
  distribution" label, so this is implicit.
