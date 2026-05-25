# Gretton, Bousquet, Smola & Schölkopf, 2005 — Measuring Statistical Dependence with Hilbert-Schmidt Norms

**Authors:** Arthur Gretton, Olivier Bousquet, Alex Smola, Bernhard Schölkopf
**Year:** 2005
**Venue:** Algorithmic Learning Theory (ALT 2005), Lecture Notes in Computer Science (LNAI) vol. 3734, S. Jain, H.U. Simon & E. Tomita (eds.), Springer-Verlag Berlin Heidelberg, pp. 63–77
**arXiv:** n/a (proceedings paper; preprint available on gatsby.ucl.ac.uk/~gretton/papers/GreBouSmoSch05.pdf)
**DOI:** 10.1007/11564089_7

## One-paragraph summary
Introduces the **Hilbert–Schmidt Independence Criterion (HSIC)** as the squared Hilbert–Schmidt norm of the cross-covariance operator C_xy between two RKHSs F and G (Definition 1, eq. 7). Lemma 1 (eq. 8) expresses HSIC purely in terms of kernel expectations: HSIC = E[k(x,x') l(y,y')] + E[k(x,x')] E[l(y,y')] − 2 E[E_x'[k(x,x')] E_y'[l(y,y')]]. The empirical estimator (Definition 2, eq. 9) is HSIC(Z; F, G) = (m−1)^{-2} tr(KHLH), with H the centering matrix. **Theorem 4** (p. 69) proves the **iff direction**: for RKHSs with **universal kernels on compact domains**, ‖C_xy‖_HS = 0 if and only if x and y are independent. Convergence at O(m^{−1/2}) rate (Theorem 3, "Bound on Empirical HSIC") and O(m²) computation cost make HSIC simpler and faster than COCO/KGV alternatives. Applied to ICA in §6.

## Why CD-SBI cites it
§3.7 (line ~665), brief glossary entry in the "alternative-objectives taxonomy": "HSIC (Hilbert–Schmidt Independence Criterion; Gretton et al. 2005) is a kernel-based test statistic that is zero iff two random variables are independent." HSIC then appears as the canonical Class-2 independence term (line ~689): "Augment Class 1 with an independence term HSIC(θ, Φ(r)) or similar."

## Specific anchors
- **HSIC definition:** §2.3, Definition 1, eq. 7: HSIC(p_xy, F, G) := ‖C_xy‖²_HS.
- **Kernel-expectation form:** Lemma 1, eq. 8.
- **Empirical estimator:** Definition 2, eq. 9: HSIC(Z; F, G) = (m−1)^{-2} tr(KHLH).
- **Iff theorem:** Theorem 4, p. 69 — "Denote by F, G RKHSs with universal kernels k, l on the compact domains X and Y respectively... Then ‖C_xy‖_HS = 0 if and only if x and y are independent."
- **Convergence:** Theorem 3, p. 69 — O(m^{−1/2}) rate.

## Notes
- **Attribution check (✓ with minor caveat).** The manuscript's gloss "HSIC ... is zero iff two random variables are independent" is **correctly attributable to the 2005 paper**. Theorem 4 of G&L 2005 already proves the iff direction. So the round-2 worry that the "iff" is from a later paper (e.g., Fukumizu et al. 2008) is *not* warranted here — the iff is genuinely 2005.
- **Caveat: the iff requires kernel conditions.** Theorem 4 requires (a) universal kernels and (b) compact domains. The manuscript's brief glossary entry does not state these conditions. For the §3.7 use this is fine (the entry is one sentence in a taxonomy gloss, not a load-bearing claim), but if the round-3 manuscript expands the HSIC discussion (e.g., to motivate Class-2 fragility more formally), the kernel-class condition should appear — and that is precisely where the SriperumbudurEtAl2011 `\nocite` reference would carry weight, since it refines "universal" to "characteristic" and gives the Fourier-side characterization. A clean pair-cite is `\citep{GrettonEtAl2005, SriperumbudurEtAl2011}` for "HSIC = 0 iff independence, under characteristic kernels."
- **Later refinements (informational, not required for round-2 correctness).** The 2005 result uses *universal* kernels on *compact* domains. Later work (Fukumizu, Gretton, Sun, Schölkopf 2008 "Kernel measures of conditional dependence"; Sriperumbudur, Fukumizu, Lanckriet 2011) extends the iff to *characteristic* kernels on more general spaces (locally compact Hausdorff). For Gaussian / Laplace / inverse-multiquadric kernels on ℝ^n the characteristic property holds; for polynomial kernels it does not. The manuscript's Class-2 critique (fragile gradient signal, hyperparameter balancing, variance scaling) is correct *empirically* and does not turn on the kernel-class question — but a reader trained in kernel theory may want to know that the "iff" requires more than just "kernel-based."
- **Historical context worth weaving in.** HSIC's actual origin story: the *squared HS norm of cross-covariance* was suggested earlier as a footnote by Fukumizu et al. (footnote 2, p. 64 of G&L 2005, "The possibility of using a Hilbert-Schmidt norm was suggested by Fukumizu et al., although the idea was not pursued further in that work"). G&L 2005 also notes (intro, p. 64) that the empirical estimate is **identical to the quadratic dependence measure of Achard et al. (2003)**, though derived differently. So G&L 2005 are the standard citation, but the construction has a pre-history.
- **Versioning note:** The bib entry's "Algorithmic Learning Theory" without LNCS volume is technically correct but minimal. A fuller entry would be `series = {Lecture Notes in Computer Science}, volume = {3734}, editor = {Jain, S. and Simon, H. U. and Tomita, E.}, publisher = {Springer}` plus the DOI `10.1007/11564089_7`. Not required for citation correctness in plainnat style, but a polish item for round-3.
