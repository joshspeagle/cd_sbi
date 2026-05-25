# Lueckmann, Boelts, Greenberg, Gonçalves, Macke (2021) — Benchmarking Simulation-Based Inference

**Authors:** Jan-Matthis Lueckmann, Jan Boelts, David S. Greenberg,
Pedro J. Gonçalves, Jakob H. Macke
**Year:** 2021
**Venue:** *Proceedings of the 24th International Conference on
Artificial Intelligence and Statistics* (AISTATS 2021), PMLR Vol. 130.
**arXiv:** 2101.04653
**Companion code:** `sbi-benchmark/sbibm` (GitHub) and the
`sbi-benchmark.github.io` results portal.

## One-paragraph summary
The paper introduces the **sbibm benchmark** for simulation-based
inference, comprising ten tasks of varied difficulty and dimensionality
and a set of standardized performance metrics. Algorithms compared
include NPE/SNPE-C, NLE/SNLE, NRE/SNRE variants, and several ABC
baselines (Rejection-ABC, SMC-ABC, etc.). The headline empirical
findings are that **sequential variants are more sample-efficient than
amortized ones at small simulation budgets**, that **flow-based
methods broadly outperform ABC**, and that **no single method dominates
across tasks**. Evaluation metrics are **distributional**:
classifier-two-sample-test (C2ST), MMD, KSD, and median distance.

## Why CD-SBI cites it
Cited twice — at §11.5 (Higher-dimensional scaling) and §11.6
(Sequential / amortized variants).

## Specific anchors
- **§11.5 (line ~2279–2281):** "Autoregressive flows scale routinely to
  \(d \sim 30\) in the SBI literature \citep{LueckmannEtAl2021} and to
  \(d \sim 100\)+ in general normalizing-flow contexts
  \citep{PapamakariosEtAl2017, DurkanEtAl2019}."
  - **Attribution check (d ~ 30):** Of the ten sbibm tasks, the
    parameter dimensions \(d_\theta\) range from 2 to 10 (max:
    Gaussian Linear / Gaussian Linear Uniform / Bernoulli GLM, all
    \(d_\theta = 10\)). The observation dimensions \(d_x\) reach 96
    (SLCP Distractors) and 100 (Bernoulli GLM Raw). So **the "d ~ 30"
    figure is only justified if "d" is interpreted as the observation
    dimension \(d_x\), not the parameter dimension \(d_\theta\)**.
    The §11.5 prose just says "d ~ 30" without disambiguating;
    recommend either (i) rewording to "observation dimensions
    \(d_x \sim 100\)" (which is what sbibm actually demonstrates) or
    (ii) citing a different reference that demonstrates **parameter**
    dimensions in the d~30 range (e.g., Greenberg–Nonnenmacher–Macke
    SNPE-C 2019 on Hodgkin-Huxley-style models; or Gonçalves et al.
    2020 *eLife* mechanistic-model paper, \(d_\theta\) up to ~30).
    This is a minor but real misattribution.

- **§11.6 (line ~2298–2301):** "the broader sequential-SBI literature
  \citep{HermansEtAl2022, LueckmannEtAl2021} has documented empirical
  miscalibration of all naive sequential variants."
  - **Attribution check (sequential miscalibration):** Lueckmann et al.
    2021 evaluates SBI methods using **distributional** metrics
    (C2ST, MMD, KSD, median distance) and does **NOT report**
    coverage/calibration metrics for the sequential variants.
    Documenting empirical miscalibration of sequential SBI is the
    headline contribution of **Hermans et al. 2022 ("Averting a
    Crisis…")**, which is the correct citation. Lueckmann 2021 should
    be removed from this batched citation, or its role explicitly
    narrowed to "general benchmarking of sequential vs amortized
    behaviour." Recommend: keep `\citep{HermansEtAl2022}` for the
    miscalibration claim; if Lueckmann is to be retained alongside,
    soften the prose to "...the broader sequential-SBI literature
    has benchmarked these variants \citep{LueckmannEtAl2021} and
    documented empirical miscalibration of naive sequential variants
    \citep{HermansEtAl2022}."

## Notes
- Bib entry is correct.
- The full task table (for any future need): Gaussian Linear
  (\(d_\theta=10, d_x=10\)); Gaussian Linear Uniform (10, 10); SLCP
  (5, 4); SLCP Distractors (5, 96); Bernoulli GLM (10, 10); Bernoulli
  GLM Raw (10, 100); Gaussian Mixture (2, 2); Two Moons (2, 2); SIR
  (2, 10); Lotka-Volterra (4, 20).
- Hermans et al. 2022 ("Averting A Crisis in Simulation-Based
  Inference", arXiv 2110.06581) is already in the bib (`HermansEtAl2022`)
  and cited in §11.6 — the corrective is to acknowledge that this is
  the load-bearing reference for the miscalibration claim.
