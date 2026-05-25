# Wehenkel et al. (2025) — Addressing Misspecification in Simulation-based Inference through Data-driven Calibration

**Authors:** Antoine Wehenkel, Juan L. Gamella, Ozan Sener, Jens
Behrmann, Guillermo Sapiro, Jörn-Henrik Jacobsen, Marco Cuturi
**Year:** 2025 (arXiv 2024-05; revised 2025-05; ICML proceedings 2025)
**Venue:** International Conference on Machine Learning (ICML) 2025
(oral presentation).
**arXiv:** 2405.08719

## One-paragraph summary
The paper introduces **Robust Posterior Estimation (RoPE)**, a framework
for correcting simulator misspecification in SBI by means of a small
**real-world calibration set** of (parameter, observation) pairs.
The misspecification gap between real and simulated observations is
formalized as the solution of an **optimal-transport (OT) problem
between learned representations** of the two distributions. RoPE
combines the calibration set with this learned OT map to deliver a
controllable trade-off between calibrated uncertainty and informative
inference under (potentially severe) simulator misspecification.

## Why CD-SBI cites it
Cited at §11.4 (Open problem: Misspecification) as the closest existing
work studying simulator misspecification in SBI. The manuscript notes
that a direct coverage-degradation analysis of the forward-KL
projection of the true conditional onto the architectural class is
still missing in the literature, and Wehenkel et al. (2025) supply a
complementary mechanism (OT-based calibration) that CD-SBI does not yet
integrate.

## Specific anchors
- **§11.4 (line ~2269–2272):** "\citet{WehenkelEtAl2025} study a
  related question via a **reweighted / generalized-Bayes robust SBI
  objective**; a direct coverage-degradation analysis of the
  forward-KL projection of \(p\) onto \(\mathcal{F}\) is, to our
  knowledge, still missing."
- **Issue (misattribution):** RoPE is **not** a reweighted or
  generalized-Bayes / power-posterior mechanism. It is an
  **optimal-transport calibration mechanism that exploits a small
  ground-truth-labelled calibration set** of (θ, X_real) pairs to learn
  the misspecification gap between real and simulated representations.
  The Schmon-style generalized-Bayes route is a *separate* line of
  work (cited next door in §11.4 as `SchmonEtAl2020`). Recommend
  rewording §11.4 to: "Wehenkel et al. (2025) study a related question
  via an OT-based data-driven calibration framework (RoPE) that uses a
  small real-world calibration set to learn and correct the
  simulator-vs-real misspecification gap."

## Notes
- Bib entry is correct (authors, venue ICML 2025, arXiv id). The note
  field `arXiv:2405.08719` is right.
- Apple Machine Learning Research lists the paper as well; ICML 2025
  oral presentation slot 47170, poster 43556.
- Worth noting in CD-SBI's framing: RoPE assumes access to a
  calibration set of real (θ, X) pairs, which CD-SBI does **not** assume.
  This is a meaningful structural difference and reinforces the
  manuscript's claim that direct coverage analysis under
  no-calibration-set misspecification is still open.
