# Carlier, Galichon, Santambrogio (2010) — From Knothe's Transport to Brenier's Map and a Continuation Method for Optimal Transport

**Authors:** G. Carlier, A. Galichon, F. Santambrogio
**Year:** 2010 (arXiv preprint Oct 2008; published Jan 2010)
**Venue:** SIAM Journal on Mathematical Analysis, Vol. 41, No. 6, pp. 2554–2576.
**arXiv:** 0810.4153
**DOI:** 10.1137/080740647

## One-paragraph summary
Carlier–Galichon–Santambrogio (CGS) prove that the Knothe–Rosenblatt
(KR) rearrangement arises as the limiting case of a one-parameter family
of Monge–Kantorovich optimal-transport problems with **anisotropic
quadratic cost**, where the diagonal weights on successive coordinates
asymptotically dominate one another. Concretely, for a quadratic cost
\(c_t(x,y) = \sum_k \lambda_k(t)\,(x_k - y_k)^2\) with weights
\(\lambda_k(t)\) chosen so that
\(\lambda_1(t) \gg \lambda_2(t) \gg \cdots \gg \lambda_d(t)\) as
\(t \to 0^+\), the optimal Brenier maps for each \(t\) converge to the
KR rearrangement at \(t = 0\); at \(t = 1\) (or any fixed isotropic
weights) the same family recovers the standard Brenier map for the
quadratic cost. This gives a **continuation/homotopy method** for
computing Brenier maps that starts at the cheap-to-compute KR map and
deforms it to the Brenier map by gradually equalizing the coordinate
weights.

## Why CD-SBI cites it
Cited at §11.2 (Open problem: KR ordering selection) as background for
the **conjecture** that "a more isotropic transport target might give
better-behaved finite-sample training." CGS supplies the precise
mathematical sense in which KR maps and Brenier maps are endpoints of a
single one-parameter family, motivating the suggestion that pivots in
the interior of this family (more isotropic than KR but architecturally
easier than full Brenier) might exist as design targets in CD-SBI.

## Specific anchors
- **§11.2 (line ~2237–2244):** "The optimal-transport literature
  \citep{CarlierEtAl2010} connects KR maps to Brenier maps through an
  **entropic/diagonal-scaling continuation**…"
- **Issue:** CGS's continuation is **purely diagonal/anisotropic scaling
  of the quadratic cost** — there is no entropy / KL term. The phrasing
  "entropic/diagonal-scaling" conflates this with the **entropic OT**
  thread (Cuturi 2013; Genevay–Peyré–Cuturi; Feydy 2019), which is a
  *separate* development. Recommend rewording §11.2 to drop "entropic/"
  and keep only "diagonal-scaling" (or "anisotropic-cost").

## Notes
- No author–year metadata issues. Bib entry is correct.
- The CGS result is the canonical reference for the KR-as-limit-of-OT
  characterization; alternative treatments are in Villani's *Topics in
  Optimal Transportation* (2003, AMS GSM 58) and his *Optimal Transport:
  Old and New* (2009, Springer Grundlehren 338), but CGS is the primary
  source for the continuation formulation specifically.
- Bonnotte (arXiv 1205.1099, "From Knothe's rearrangement to Brenier's
  optimal transport map") gives a closely related treatment and could
  be cited as a complementary reference if a second citation is wanted.
