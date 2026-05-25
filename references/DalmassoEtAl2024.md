# Dalmasso, Masserano, Zhao, Izbicki, Lee, 2024 — Likelihood-Free Frequentist Inference (LF2I)

**Authors:** Niccolò Dalmasso, Luca Masserano, David Zhao, Rafael Izbicki, Ann B. Lee
**Year:** 2024
**Venue:** Electronic Journal of Statistics, vol. 18, no. 2, pp. 5045–5090
**arXiv:** 2107.03920 (first posted 2021)
**DOI:** 10.1214/24-EJS2307

## One-paragraph summary
The consolidated journal paper of the LF2I framework, which provides
finite-sample frequentist confidence sets for parameters of
simulator-based models. The pipeline (i) learns a test statistic
(e.g., ACORE, BFF, or related variants) whose null distribution under
each `θ` is needed for Neyman construction, then (ii) learns a
critical-value function via quantile regression, and (iii) inverts the
test to obtain confidence sets. Includes a diagnostic step for
verifying conditional coverage at fixed `θ`. Earlier conference papers
(Dalmasso et al. 2020 ACORE, Masserano et al. 2023 WALDO) developed
specific test-statistic choices; LF2I is the unifying framework paper.

## Why CD-SBI cites it
- §1.1 "Neyman-construction wrappers (LF2I)" — retrofit calibration
  family.
- §3.7 explicitly notes that LF2I avoids the Class-4 (CRPS-against-
  reference) propriety collapse because the test statistic learning
  uses pinball loss in the *standard* direction.
- §7.3 (diagnostics) — LF2I's local-coverage diagnostic is the
  closest cousin to CD-SBI's coverage check.
- §10 comparison-table row: target = "Confidence sets", calibration =
  "Neyman construction on learned test stat".

## Specific anchors
- ACORE (Approximate Computation of Conditional Likelihood Ratios) and
  BFF (Bayes Frequentist Factor) are specific test-statistic choices
  bundled under LF2I.
- The Neyman-construction step requires a *separately-trained*
  critical-values branch — this is what the §10 prose contrasts with
  CD-SBI's single-stage architecture.

## Notes
- **Attribution-check OK.** The 2024 EJS reference is the right
  "consolidated" version (not the 2020 NeurIPS ACORE paper or the 2021
  arXiv preprint). DOI 10.1214/24-EJS2307 confirmed.
- The §10 row label "LF2I (ACORE, BFF)" correctly groups the
  framework with its specific test-statistic variants. ACORE was
  originally Dalmasso et al. ICML 2020; if the manuscript wants a
  separate citation for ACORE specifically, that would be an
  additional entry, but the consolidated 2024 paper covers both.
