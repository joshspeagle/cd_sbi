# Brenier, 1991 — Polar Factorization and Monotone Rearrangement of Vector-Valued Functions

**Authors:** Yann Brenier
**Year:** 1991
**Venue:** Communications on Pure and Applied Mathematics, Vol. 44, No. 4, pp. 375–417
**arXiv:** n/a (predates arXiv; an earlier announcement appeared as
  Brenier 1987, *C. R. Acad. Sci. Paris* 305, 805–808)
**DOI:** 10.1002/cpa.3160440402
**Publisher link:** https://onlinelibrary.wiley.com/doi/10.1002/cpa.3160440402

## One-paragraph summary
Brenier proved that any sufficiently regular vector-valued map
\(u: \Omega \to \mathbb{R}^d\) admits an essentially unique
**polar factorization** \(u = (\nabla \phi) \circ s\), where \(\phi\) is
a convex function on \(\mathbb{R}^d\) and \(s\) is a Lebesgue-measure-
preserving rearrangement of \(\Omega\). The gradient-of-a-convex-
function map \(\nabla \phi\) is the **monotone rearrangement** of
\(u\) — the unique \(L^2\)-optimal transport map between Lebesgue
measure on \(\Omega\) and the pushforward \(u_\sharp(\mathcal{L}|_\Omega)\).
The proof uses a Monge–Kantorovich variational principle and the
associated Monge–Ampère equation. This established what is now called
the **Brenier map** as the solution of the Monge problem with squared
Euclidean cost.

## Why CD-SBI cites it
§6.3 contrasts the **Knothe–Rosenblatt** map (triangular, order-
dependent) with the **Brenier** map (gradient of a convex function,
order-independent) as the two canonical uniqueness statements for
multivariate transport. The manuscript chooses KR because its
triangular structure aligns with autoregressive masking; Brenier is
cited for context, not used.

## Specific anchors
- §6.3 paragraph 1: "It differs from the **Brenier map**
  \citep{Brenier1991}, which is the unique transport map that is the
  gradient of a convex function (the \(L^2\)-optimal transport plan).
  KR depends on coordinate ordering; Brenier does not."
- §11.2 (OP-11.2, "Ordering selection in KR"): the natural follow-up,
  cited via `CarlierEtAl2010`, is the continuation from KR to Brenier.

## Notes
- The manuscript's one-sentence characterization is accurate: Brenier
  proved existence and uniqueness of the gradient-of-convex map as the
  \(L^2\)-OT plan. The Wikipedia "Polar factorization theorem" article
  matches this characterization.
- **McCann (1995, *Duke Math. J.* 80(2), 309–323, DOI
  10.1215/S0012-7094-95-08013-2)** generalized Brenier's result by
  removing the technical regularity assumptions on the source measure:
  McCann showed that if \(\mu\) gives mass zero to every Lipschitz
  \((n-1)\)-surface, then a convex potential exists whose gradient
  pushes \(\mu\) forward to \(\nu\) and is \(\mu\)-a.e. unique.
  Brenier 1991 had required absolute continuity with respect to
  Lebesgue measure plus integrability conditions. McCann is the
  standard "general-measure" companion to Brenier 1991.
- The manuscript does **not** rely on the McCann generalization — the
  Brenier statement at §6.3 is purely contrastive — so adding McCann
  1995 is not required for correctness, but it is the natural co-
  citation if one wants to give the reader the full picture of the
  Brenier-map literature.
- Metadata in the .bib is correct (author, title, journal, vol/no/pages,
  year). Missing: DOI. Adding `doi = {10.1002/cpa.3160440402}` would
  improve the entry.
