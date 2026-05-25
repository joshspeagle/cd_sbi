# Papamakarios, Pavlakou, Murray (2017) — Masked Autoregressive Flow for Density Estimation

**Authors:** George Papamakarios, Theo Pavlakou, Iain Murray
**Year:** 2017 (arXiv submitted May 19, 2017; final v3 June 14, 2018)
**Venue:** *Advances in Neural Information Processing Systems* (NeurIPS)
30, 2017.
**arXiv:** 1705.07057

## One-paragraph summary
The paper introduces **Masked Autoregressive Flow (MAF)**, a
normalizing-flow architecture that stacks several Masked Autoencoder
for Distribution Estimation (MADE) layers to model an autoregressive
factorization of the joint density. Each MADE layer outputs the
location and scale of a conditional Gaussian for each coordinate given
its predecessors, giving an exact, tractable Jacobian (triangular) and
a flexible density model. MAF is shown to outperform Real NVP and to
match or beat existing autoregressive density estimators on
general-purpose benchmarks. A **Conditional MAF** variant for
conditional density estimation is also introduced.

## Why CD-SBI cites it
Cited at §11.5 (Higher-dimensional scaling) as evidence that
autoregressive normalizing flows scale to high dimensions in general
density-estimation contexts.

## Specific anchors
- **§11.5 (line ~2280–2281):** "Autoregressive flows scale routinely
  to \(d \sim 30\) in the SBI literature \citep{LueckmannEtAl2021} and
  to \(d \sim 100\)+ in general normalizing-flow contexts
  \citep{PapamakariosEtAl2017, DurkanEtAl2019}."
- **Attribution check (d ~ 100+):** Confirmed. From the MAF paper's
  Table 5 (appendix), the test datasets and their dimensionalities are:
  POWER (6), GAS (8), HEPMASS (21), MINIBOONE (43), BSDS300 (63),
  MNIST (784), CIFAR-10 (3072). MAF therefore demonstrates
  unconditional density estimation up to \(d = 3072\), and conditional
  modeling on MNIST/CIFAR. The "d ~ 100+" wording in §11.5 is
  accurate, though it understates MAF's range; "d up to thousands" or
  "d ~ 10^3" would be more faithful. No correction strictly required.

## Notes
- Bib entry is correct (authors, title, venue, year).
- Companion code at github.com/gpapamak/maf with preprocessed datasets
  on Zenodo (record 1161203).
- The MAF construction is essentially the autoregressive triangular
  flow that CD-SBI's Theorem A-d singles out as the canonical class —
  worth noting that the architectural choice in §6 inherits directly
  from MAF (and IAF / NSF) lineage.
