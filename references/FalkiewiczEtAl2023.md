# Falkiewicz et al., 2023 — Calibrating Neural Simulation-Based Inference with Differentiable Coverage Probability

**Authors:** Maciej Falkiewicz, Naoya Takeishi, Imahn Shekhzadeh, Antoine Wehenkel, Arnaud Delaunoy, Gilles Louppe, Alexandros Kalousis
**Year:** 2023
**Venue:** Advances in Neural Information Processing Systems 36 (NeurIPS 2023)
**arXiv:** 2310.13402
**DOI:** —

## One-paragraph summary
Augments standard amortized SBI training (NPE, NRE, etc.) with a
differentiable approximation to the expected calibration / coverage
error, allowing this term to be back-propagated alongside the primary
log-density (or classification) loss. The relaxation softens the
indicator-function step in the coverage definition, producing a
gradient signal that pushes the learned posterior toward nominal
coverage. Empirically improves coverage on six standard SBI
benchmark tasks compared to NPE/NRE/etc. without the calibration term.

## Why CD-SBI cites it
- §1.1 lists Calibrated NPE among the "retrofit" calibration methods
  ("differentiable coverage probabilities").
- §3.7 puts Calibrated NPE on the same shelf as Balanced NRE — adding
  regularization terms to address conservativeness post hoc.
- §10 comparison-table row: target = "Posterior", calibration =
  "Differentiable coverage".
- §11.6 (sequential variants): related to amortized SBI extensions.

## Specific anchors
- Differentiable relaxation of the calibration error indicator.
- End-to-end backpropagation through coverage probability.

## Notes
- **Attribution-check OK.** NeurIPS 2023 acceptance confirmed via
  proceedings page; all seven authors confirmed.
- The §10 characterization "Differentiable coverage" is concise and
  accurate.
