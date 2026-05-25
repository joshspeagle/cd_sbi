# Rosenblatt, 1952 — Remarks on a Multivariate Transformation

**Authors:** Murray Rosenblatt
**Year:** 1952
**Venue:** The Annals of Mathematical Statistics, Vol. 23, No. 3, pp. 470–472
**arXiv:** n/a (predates arXiv)
**DOI:** 10.1214/aoms/1177729394
**Publisher link:** https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-23/issue-3/Remarks-on-a-Multivariate-Transformation/10.1214/aoms/1177729394.full

## One-paragraph summary
A three-page note introducing what is now called the **Rosenblatt
transformation**. Given an absolutely continuous random vector
\(X = (X_1, \ldots, X_k)\), define the map componentwise by
\(U_1 = F_1(X_1), U_2 = F_{2|1}(X_2 \mid X_1), \ldots,
U_k = F_{k|1,\ldots,k-1}(X_k \mid X_{<k})\), where \(F_{j|<j}\) is the
conditional CDF. Rosenblatt proved that the image \(U\) is
distributed uniformly on \([0,1]^k\) — i.e., the multivariate PIT — and
discussed implications for goodness-of-fit testing in the multivariate
setting. The transform is **triangular by construction**: each
\(U_j\) depends only on \(X_{\le j}\).

## Why CD-SBI cites it
§6.3 names the multivariate uniqueness target the
**Knothe–Rosenblatt rearrangement** and credits Rosenblatt for the
triangular construction. The CD-SBI pivot
\(r_k^{\mathrm{KR}}(\theta, X) = \Phi^{-1}(1 - F_k^{(\theta)}(X_k \mid X_{<k}))\)
is exactly the Rosenblatt transform with the uniform output composed
with \(\Phi^{-1}\) to land on \(\mathcal{N}(0, I_d)\) instead of
\([0,1]^d\), and with the sign flipped via \(1 - F\) to align with the
(R2\(^\mathrm{auto}\)) convention \(\partial_{X_k} r_k < 0\).

## Specific anchors
- §6.3 paragraph 1: "the **Knothe–Rosenblatt rearrangement**
  \citep{Rosenblatt1952} is the unique transport map of triangular
  form."
- §6.3, statement of Theorem A-d: the unique element of \(\mathcal{M}\)
  is the KR rearrangement, with closed form
  \(r_k^{\mathrm{KR}} = \Phi^{-1}(1 - F_k^{(\theta)}(\cdot \mid X_{<k}))\).
- §6.3 Remarks (1): ordering-dependence is a direct corollary of the
  Rosenblatt construction (different coordinate orderings give
  different but inferentially equivalent triangular maps).

## Notes
- Rosenblatt 1952 establishes the **transformation** and proves its
  pushforward property; it does **not** state a uniqueness theorem
  for the triangular monotone map between two arbitrary measures.
  That uniqueness statement is folklore in OT and was formalized later
  (see Bonnotte 2013, Carlier–Galichon–Santambrogio 2010, Santambrogio's
  *Optimal Transport for Applied Mathematicians* §2.3 for the modern
  proof).
- The compound name "Knothe–Rosenblatt" reflects that **Knothe (1957)**
  arrived at the same triangular construction independently in convex-
  bodies / Brunn–Minkowski theory ("Contributions to the theory of
  convex bodies," Michigan Math. J. 4, 39–52). The current manuscript
  cites only Rosenblatt 1952 under the name "Knothe–Rosenblatt"; this
  is an attribution gap (see report).
- Metadata in the .bib is correct (author, title, journal, vol/no/pages,
  year). Missing: DOI. Adding `doi = {10.1214/aoms/1177729394}` would
  make the bibliography entry round out.
