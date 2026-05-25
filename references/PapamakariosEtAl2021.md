# Papamakarios et al., 2021 — Normalizing Flows for Probabilistic Modeling and Inference

**Authors:** George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, Balaji Lakshminarayanan
**Year:** 2021
**Venue:** Journal of Machine Learning Research, Vol. 22, Article 57, pp. 1–64
**arXiv:** 1912.02762
**DOI:** n/a (JMLR open-access)
**Publisher link:** https://jmlr.org/papers/v22/19-1028.html

## One-paragraph summary
A 64-page **review article** on normalizing flows, organized around the
core idea that an expressive distribution can be defined as a chain of
invertible, differentiable transformations applied to a simple base
distribution. The paper covers flow design (autoregressive flows,
coupling flows, linear flows, residual flows, continuous-time flows),
expressivity / universality results, applications to density estimation,
variational inference, generative modelling, and approximate inference.
Autoregressive flows with **triangular Jacobian structure** are a
major topic (Section 3 in the v2 arXiv version), with MAF, IAF, NAF, and
NSF discussed as instances. The review aims to be self-contained and
serves as the standard pedagogical reference for the field.

## Why CD-SBI cites it
§6 intro lists this review as a "standard reference on this
architectural pattern" (triangular autoregressive flows) — i.e., the
manuscript points readers to it as the textbook account of why
autoregressive flows have lower-triangular Jacobians and how this
structure enables \(O(d)\) density evaluation. §11.5 (open problem on
higher-d scaling) cites it as part of the NF toolkit.

## Specific anchors
- §6 intro (line ~1268): "Standard references on this architectural
  pattern include \citet{PapamakariosEtAl2021} and \citet{DurkanEtAl2019}."
- §11.5: cited (via the inventory) for the NF-review framing of the
  scaling problem.

## Notes
- **Attribution check passes.** The review *does* prominently cover
  autoregressive flows with triangular Jacobians; this is one of its
  central architectural categories. The §6 intro citation is accurate.
- The review does **not**, as far as I can verify, devote space to the
  Knothe–Rosenblatt rearrangement under that name — the OT lineage is
  noted but the survey's framing is computational rather than measure-
  theoretic. So this is the right citation for "how triangular
  autoregressive flows work as a deep-learning architecture," not for
  "why KR is the unique triangular OT map."
- Metadata in the .bib is correct (authors, title, journal, volume,
  pages, year). The page range "1--64" matches JMLR's count for
  article 57.
