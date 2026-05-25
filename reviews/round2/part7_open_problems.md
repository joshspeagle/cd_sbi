# Round 2 — Part VII (Open problems, §11) Lit-Review

## Summary

Part VII cites six primary references (Carlier–Galichon–Santambrogio
2010, Wehenkel et al. 2025, Schmon et al. 2020, Dellaporta et al. 2022,
Lueckmann et al. 2021, Papamakarios et al. 2017) plus three that
already appear in the bib (Hermans et al. 2022, Durkan et al. 2019,
Falkiewicz et al. 2023). All six metadata checks pass — authors, year,
venue, page count are correct as written. **Three attribution/wording
issues were found**, all moderate:

1. **§11.2 (CGS):** the phrasing "entropic/diagonal-scaling
   continuation" conflates two distinct OT-continuation paradigms.
   CGS 2010 is **diagonal-scaling only**; entropic OT (Cuturi 2013 et
   seq.) is a separate development. Drop "entropic/".
2. **§11.4 (Wehenkel 2025):** the framing "reweighted /
   generalized-Bayes robust SBI objective" misattributes the
   mechanism. Wehenkel et al. 2025 introduce **RoPE**, which is an
   **optimal-transport calibration** built on a small **real-world
   calibration set** of (θ, X) pairs. It is neither a power posterior
   nor a likelihood reweighting. Reword.
3. **§11.6 (Lueckmann 2021):** the batched citation
   `\citep{HermansEtAl2022, LueckmannEtAl2021}` for "documented
   empirical miscalibration of all naive sequential variants" is
   carried entirely by Hermans 2022. Lueckmann 2021 uses
   **distributional** metrics (C2ST, MMD, KSD, median distance) only
   and does not report coverage. Narrow or drop the Lueckmann citation
   for this specific claim.

Two smaller observations: §11.5's "d ~ 30 in the SBI literature
\citep{LueckmannEtAl2021}" only resolves if d means **observation
dimension** (sbibm's max d_x is 100; max d_θ is 10) — a clarification
would help. And §11.5's "d ~ 100+" for MAF understates the actual
demonstrated range (CIFAR-10, d = 3072) but is not wrong.

Citation gaps: §11.1 and §11.3 are mathematically substantive open
problems with no citations. Reasonable historical anchors exist
(Villani 2003/2009 for the OT side; Federer / Rademacher classical for
Lipschitz almost-everywhere differentiability). Optional — these are
open-problem sections, not survey sections.

## Per-citation findings

### `CarlierEtAl2010` — §11.2 (OP-11.2: KR ordering selection)

- **Metadata:** Correct. Carlier, Galichon, Santambrogio. SIAM J. Math.
  Anal. 41(6), 2554–2576 (2010). arXiv 0810.4153. DOI 10.1137/080740647.
- **Attribution issue (moderate):** The manuscript wording
  > "The optimal-transport literature \citep{CarlierEtAl2010} connects
  > KR maps to Brenier maps through an **entropic/diagonal-scaling
  > continuation**…"
  conflates two distinct continuation paradigms.
  - CGS 2010's continuation is parametrized **only** by anisotropic
    quadratic cost weights \(\lambda_k(t)\), with
    \(\lambda_1 \gg \cdots \gg \lambda_d\) as \(t \to 0^+\) recovering
    KR and equal weights recovering Brenier. **There is no entropy /
    KL term anywhere in the construction.**
  - Entropic OT (Cuturi NeurIPS 2013; Genevay–Peyré–Cuturi AISTATS
    2018; Feydy et al. AISTATS 2019) regularizes the Kantorovich
    primal by a KL/entropy penalty. That is a **separate** body of
    work.
  - **Recommendation:** Replace "entropic/diagonal-scaling
    continuation" with "diagonal-scaling continuation" (or
    "anisotropic-cost continuation"). Optionally cite Villani *Topics
    in Optimal Transportation* (2003) §2.3 as a textbook treatment of
    KR-as-OT-limit, and/or Bonnotte (arXiv 1205.1099) as a
    complementary reference.

### `WehenkelEtAl2025` — §11.4 (OP-11.4: Misspecification)

- **Metadata:** Correct. Wehenkel, Gamella, Sener, Behrmann, Sapiro,
  Jacobsen, Cuturi. ICML 2025 (oral). arXiv 2405.08719.
- **Attribution issue (moderate):** The manuscript wording
  > "\citet{WehenkelEtAl2025} study a related question via a
  > **reweighted / generalized-Bayes robust SBI objective**"
  misidentifies the mechanism. The actual mechanism is **Robust
  Posterior Estimation (RoPE)**: an **optimal-transport calibration**
  built on a small **real-world calibration set** of (θ, X_real) pairs.
  The OT step models the simulator-vs-real misspecification gap on
  learned representations; the calibration set anchors the OT
  correction. This is **not** prior-predictive reweighting,
  generalized Bayes, or a power posterior.
  - **Recommendation:** Replace the wording with something like
    "Wehenkel et al. (2025) study a related question via a data-driven
    OT calibration framework (RoPE) that uses a small real-world
    calibration set to learn and correct the simulator-vs-real
    misspecification gap." Also worth noting in CD-SBI's framing that
    RoPE *assumes access* to a real-data calibration set, which is a
    different problem setting from the no-calibration-data regime
    CD-SBI defaults to. That strengthens the manuscript's claim that
    "a direct coverage-degradation analysis … is still missing."

### `SchmonEtAl2020` — §11.4 (OP-11.4: Robust SBI)

- **Metadata:** Correct. Schmon, Cannon, Knoblauch. arXiv 2011.08644
  (2020). Accepted at AABI 2020 workshop; no further publication.
  Keep as `@article` arXiv entry.
- **Attribution check:** Correctly described as a "robust-SBI" idea.
  The mechanism is **generalized Bayesian / Gibbs-posterior**
  reinterpretation of ABC's accept/reject as a loss-based update. The
  current §11.4 prose batches it with Dellaporta et al. 2022 without
  distinguishing mechanisms — see below.
- **Recommendation (minor):** Optional one-line parenthetical
  distinguishing the two mechanisms (Schmon = generalized Bayes;
  Dellaporta = MMD posterior bootstrap).

### `DellaportaEtAl2022` — §11.4 (OP-11.4: Robust SBI)

- **Metadata:** Correct. Dellaporta, Knoblauch, Damoulas, Briol.
  AISTATS 2022 oral. PMLR Vol. 151, pp. 943–970. arXiv 2202.04744.
  - Optional: add `pages = {943--970}, volume = {151}, series =
    {Proceedings of Machine Learning Research}` for completeness.
    Not required for natbib resolution.
- **Attribution check:** Correct. Mechanism is **MMD posterior
  bootstrap** (nonparametric Lyddon–Holmes–Walker style) — distinct
  from Schmon's generalized-Bayes route, and the current §11.4 prose
  does not blur them, just batches them.

### `LueckmannEtAl2021` — §11.5 and §11.6

- **Metadata:** Correct. Lueckmann, Boelts, Greenberg, Gonçalves,
  Macke. AISTATS 2021. PMLR Vol. 130. arXiv 2101.04653. Companion
  benchmark `sbibm`.
- **Attribution issue 1 — §11.5 "d ~ 30 in the SBI literature":**
  Of the 10 sbibm tasks, parameter dimensions \(d_\theta\) range from
  2 to 10 (max = 10, for Gaussian Linear / Gaussian Linear Uniform /
  Bernoulli GLM). Observation dimensions \(d_x\) reach 100 (Bernoulli
  GLM Raw) and 96 (SLCP Distractors). So sbibm demonstrates **d_x ~
  100, d_θ ≤ 10**.
  - **Recommendation:** Reword to be specific. Either "observation
    dimensions \(d_x\) up to 100" with `LueckmannEtAl2021`, or, for
    parameter dimensions \(d_\theta \sim 30\), cite Greenberg et al.
    2019 / Gonçalves et al. 2020 / Boelts et al. 2022 (these are the
    actual high-\(d_\theta\) examples in the SBI literature).
- **Attribution issue 2 — §11.6 "documented empirical miscalibration
  of all naive sequential variants":** Lueckmann 2021 uses **C2ST,
  MMD, KSD, and median distance** (distributional metrics), **not
  coverage or calibration**. The miscalibration finding is Hermans et
  al. 2022 ("Averting a Crisis…"), already cited next to Lueckmann in
  this line.
  - **Recommendation:** Remove `LueckmannEtAl2021` from this specific
    claim, or split the citation: "the broader sequential-SBI
    literature has benchmarked these variants
    \citep{LueckmannEtAl2021} and documented empirical miscalibration
    of naive sequential variants \citep{HermansEtAl2022}."

### `PapamakariosEtAl2017` — §11.5

- **Metadata:** Correct. Papamakarios, Pavlakou, Murray. NeurIPS 2017.
  arXiv 1705.07057.
- **Attribution check (d ~ 100+):** Confirmed. From the MAF paper's
  Table 5, datasets and dimensions are POWER (6), GAS (8), HEPMASS
  (21), MINIBOONE (43), BSDS300 (63), MNIST (784), CIFAR-10 (3072).
  "d ~ 100+" is accurate but understates; the demonstrated range is
  up to ~3000. Optional rewording "up to \(d \gtrsim 10^3\) on image
  benchmarks" would be more faithful but not strictly required.

## Citation gaps

### OP-11.1 (Uniqueness beyond autoregressive triangular)
The §11.1 discussion uses the language of OT (KR rearrangement, Brenier
maps, calibration manifold, rotational ambiguity) but has no citation.
For an open-problem section this is acceptable, but two anchors might
help readers:
- **Brenier polar factorization** (already cited as `Brenier1991` in
  §6.3) and the resulting uniqueness theorem for OT maps with quadratic
  cost — relevant for the "beyond-autoregressive" picture.
- **Villani, *Topics in Optimal Transportation*** (2003, AMS GSM 58)
  or *Optimal Transport: Old and New* (2009, Springer Grundlehren 338)
  for a textbook treatment of the orbit-of-OT-maps question. Either
  is a natural single anchor for OT-uniqueness machinery in this
  section.
  - Bib entry not currently in `cd_sbi.bib`. Suggested key:
    `Villani2003` (Topics) or `Villani2009` (Old and New).
- *Optional, not required.*

### OP-11.3 (R3 from R1 + Lipschitz)
The §11.3 discussion of "Lipschitz V-shape" failure of the C¹ blow-up
argument touches on classical real-analysis territory:
- **Rademacher's theorem** (Lipschitz functions are differentiable
  almost everywhere) — covered in Federer, *Geometric Measure Theory*
  (1969) §3.1.6.
- For the modern transport-on-metric-measure-spaces treatment of
  Lipschitz transport maps, **Ambrosio, Gigli, Savaré, *Gradient Flows
  in Metric Spaces and in the Space of Probability Measures*** (2nd
  ed., Birkhäuser 2008) is the standard.
- *Optional, not required for the present manuscript scope.*

## Historical-context additions

None strictly required. Two soft suggestions:

1. **§11.2 KR–Brenier continuation:** if reworded per the CGS issue
   above, a one-line forward pointer that **entropic OT** (Cuturi et
   al.) is a *separate* and equally relevant continuation paradigm
   would head off the same conflation in future versions.
2. **§11.4 robust SBI:** the §11.4 paragraph batches Schmon (gen-Bayes)
   and Dellaporta (MMD bootstrap) and points to Wehenkel (RoPE-OT) in
   the preceding sentence. The taxonomy is "three distinct robustness
   mechanisms, each plausibly compatible with CD-SBI" — one sentence
   making this taxonomy explicit would clarify the §11.4 message.

## Status

Round-2 Part VII review complete. Six paper notes written to
`/mnt/c/Users/joshs/Dropbox/GitHub/cd_sbi/references/`. Three moderate
attribution issues identified (§11.2 CGS wording, §11.4 Wehenkel
mechanism, §11.6 Lueckmann/Hermans split); one minor clarification
issue (§11.5 dimensions). No metadata corrections required to the bib
file.
