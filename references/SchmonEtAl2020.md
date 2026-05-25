# Schmon, Cannon, Knoblauch (2020) — Generalized Posteriors in Approximate Bayesian Computation

**Authors:** Sebastian M. Schmon, Patrick W. Cannon, Jeremias Knoblauch
**Year:** 2020 (arXiv submitted Nov 17, 2020; revised Feb 23, 2021)
**Venue:** Accepted to *Advances in Approximate Bayesian Inference*
(AABI) 2020 workshop. No subsequent journal/conference proceedings
publication; primary citation remains the arXiv preprint.
**arXiv:** 2011.08644

## One-paragraph summary
Schmon, Cannon, and Knoblauch reinterpret ABC's accept/reject mechanism
as an **implicit (and inherently misspecified) error model** for the
data. Rather than treating the resulting ABC posterior as a defective
approximation to a Bayesian posterior, they cast it as a **generalized
Bayesian / Gibbs posterior** — i.e., a posterior built from a general
loss function rather than a likelihood, in the sense of Bissiri–Holmes–
Walker and Knoblauch–Jewson–Damoulas. The framework yields a unified
view of several ABC algorithms (including kernel ABC and MMD-ABC),
clarifies the role of the bandwidth/tolerance ε, and provides
theoretical handles (e.g. robustness, predictive consistency) inherited
from the generalized-Bayes literature.

## Why CD-SBI cites it
Cited at §11.4 (Open problem: Misspecification) as one of two
representative "Robust-SBI ideas" (alongside Dellaporta et al. 2022)
that should plausibly integrate with CD-SBI but whose right framework
is unclear.

## Specific anchors
- **§11.4 (line ~2273–2274):** "Robust-SBI ideas \citep{SchmonEtAl2020,
  DellaportaEtAl2022} should also integrate with CD-SBI but the right
  framework is unclear."
- **Attribution check:** The citation correctly identifies SCK 2020 as
  a **generalized-Bayes / Gibbs-posterior** approach to ABC. The
  manuscript co-cites Dellaporta et al. 2022 in the same line, but
  those two papers use **distinct mechanisms** (Schmon = generalized
  Bayes / power-posterior reweighting; Dellaporta = MMD-based
  nonparametric posterior bootstrap). The §11.4 wording correctly
  treats them as parallel-but-distinct ideas, but a one-line
  parenthetical distinguishing them would aid the reader.

## Notes
- Bib entry is correct: `@article{SchmonEtAl2020, … journal = {arXiv
  preprint arXiv:2011.08644}, year = {2020}}`. Note that no
  peer-reviewed venue exists for this work — the AABI workshop is the
  only formal acceptance. Keep the `@article` arXiv entry as-is.
- Related, possibly worth a separate "see also" in §11.4 if CD-SBI
  wants to expand: Bissiri, Holmes, Walker (JRSS-B 2016) for the
  general theory of generalized Bayesian updating.
