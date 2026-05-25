# Dellaporta, Knoblauch, Damoulas, Briol (2022) — Robust Bayesian Inference for Simulator-Based Models via the MMD Posterior Bootstrap

**Authors:** Charita Dellaporta, Jeremias Knoblauch, Theodoros Damoulas,
François-Xavier Briol
**Year:** 2022
**Venue:** *Proceedings of the 25th International Conference on
Artificial Intelligence and Statistics* (AISTATS 2022), PMLR Vol. 151,
pp. 943–970. Oral presentation.
**arXiv:** 2202.04744

## One-paragraph summary
Dellaporta et al. propose a **nonparametric posterior bootstrap** for
simulator-based models, using the **Maximum Mean Discrepancy (MMD)** as
the discrepancy between simulated and observed data in a kernel-induced
RKHS. The MMD distance is plugged into a posterior-bootstrap (Lyddon–
Holmes–Walker style) inference engine, which yields posterior samples
that are (i) parallelizable across bootstrap replicates, (ii) provably
frequentist-consistent, and (iii) **robust under misspecification** in
a precise sense (generalisation bounds and posterior robustness
theorems). The mechanism is fundamentally different from Schmon-style
generalized Bayes: rather than reweighting a likelihood by a loss-based
exponent, it constructs the posterior by bootstrap resampling under a
loss (MMD) that is itself robust to mis-specification.

## Why CD-SBI cites it
Cited at §11.4 as one of the two representative "Robust-SBI ideas"
that should plausibly integrate with CD-SBI but whose right framework
is unclear.

## Specific anchors
- **§11.4 (line ~2273–2274):** "Robust-SBI ideas \citep{SchmonEtAl2020,
  DellaportaEtAl2022} should also integrate with CD-SBI but the right
  framework is unclear."
- **Attribution check:** The §11.4 citation is correct in flagging this
  as a robust-SBI mechanism. The manuscript co-cites Schmon et al.
  2020 in the same parenthetical; the two **use distinct mechanisms**
  (Schmon = generalized-Bayes / power posterior on ABC; Dellaporta =
  MMD-driven nonparametric posterior bootstrap) and the prose could
  optionally distinguish them.

## Notes
- Bib entry currently has `booktitle = {International Conference on
  Artificial Intelligence and Statistics ({AISTATS})}` and `year =
  {2022}` but **no page numbers**. PMLR Vol. 151, pp. 943–970 is the
  canonical citation; consider adding `pages = {943--970}, volume =
  {151}, series = {Proceedings of Machine Learning Research}` for
  completeness. (Not required — the current entry resolves correctly
  via natbib.)
- The Lyddon–Holmes–Walker posterior-bootstrap framework underlying
  this work is itself worth a "see also" mention in §11.4 if the
  manuscript wants to expand on robust-SBI machinery.
