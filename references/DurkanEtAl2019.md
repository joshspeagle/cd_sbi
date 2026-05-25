# Durkan et al., 2019 — Neural Spline Flows

**Authors:** Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios
**Year:** 2019
**Venue:** Advances in Neural Information Processing Systems 32 (NeurIPS 2019), pp. 7511–7522
**arXiv:** 1906.04032
**DOI:** 10.48550/arXiv.1906.04032
**Publisher link:** https://papers.nips.cc/paper/8969-neural-spline-flows

## One-paragraph summary
Introduces **Neural Spline Flows (NSF)**: a normalizing-flow building
block based on **monotonic rational-quadratic splines** with learnable
knots, replacing the affine or single-knot piecewise transforms that
preceded them. The spline transform is analytically invertible and
differentiable. NSF is presented as a drop-in for both **coupling**
(à la Real NVP / Glow) and **autoregressive** (à la MAF / IAF) flow
architectures, demonstrating improved density-estimation and generative
performance on image and tabular benchmarks. The key design point is
that flexibility comes from the spline knot count, not from depth, so
each flow layer becomes expressive enough that fewer layers are needed.

## Why CD-SBI cites it
§6 intro lists NSF as a "standard reference on this architectural
pattern" (triangular autoregressive flows), alongside Papamakarios
et al. 2021. NSF is the canonical recent example of how an autoregressive
network can implement a monotone scalar transform per coordinate with
high flexibility — exactly the building block the CD-SBI architecture
needs for (R1\(^\mathrm{auto}\)) / (R2\(^\mathrm{auto}\)) at each level.
§11.5 (higher-d scaling) cites NSF as part of the toolkit that has
already been pushed to higher-dimensional density estimation.

## Specific anchors
- §6 intro (line ~1268): "Standard references on this architectural
  pattern include \citet{PapamakariosEtAl2021} and \citet{DurkanEtAl2019}."
- §11.5 (open problem on higher-d scaling): NSF and MAF are cited
  together as the existing scaling toolkit.

## Notes
- **Attribution check:** NSF *is* presented in autoregressive form
  (NSF-AR) as well as coupling form (NSF-C). However, NSF's headline
  contribution is the **spline transformer block**, not the
  autoregressive structure per se — the autoregressive scaffolding is
  inherited from MAF (`PapamakariosEtAl2017`) and the coupling scaffolding
  from Real NVP / Glow. If the §6 intro citation is meant to ground the
  triangular autoregressive *architectural pattern*, the more central
  citation is **MAF (PapamakariosEtAl2017)**, which is currently cited
  only in §11.5. NSF is the right citation for "modern flexible
  monotone scalar transform" but the right one for "autoregressive
  triangular structure" is MAF. See part4 report for the recommendation.
- Metadata in the .bib is largely correct (authors, title, venue,
  volume, year). The page range (7511–7522) is missing and the DOI
  is missing; adding `pages = {7511--7522}, doi = {10.48550/arXiv.1906.04032}`
  would round out the entry.
