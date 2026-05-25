# Sriperumbudur, Fukumizu & Lanckriet, 2011 — Universality, characteristic kernels and RKHS embedding of measures

**Authors:** B. K. Sriperumbudur, K. Fukumizu, G. R. G. Lanckriet
**Year:** 2011
**Venue:** Journal of Machine Learning Research, vol. 12, pp. 2389–2410
**arXiv:** 1003.0887
**DOI:** n/a (JMLR open access)

## One-paragraph summary
Unifies two parallel strands of kernel theory: *universal* kernels (from kernel classification, sufficient to achieve the Bayes risk in the limit) and *characteristic* kernels (from kernel-based hypothesis testing, sufficient to distinguish probability measures via their RKHS embedding). The main results show that on a locally compact Hausdorff space the two notions essentially coincide for translation-invariant kernels, give clean equivalent characterisations via the Fourier transform of the kernel, and clarify which classes of probability measures are distinguished by which kernels. This is the standard reference for *which kernels make MMD / HSIC actually work as divergences*.

## Why CD-SBI cites it
Background "see also" reference (`\nocite`) for §3.7's discussion of HSIC and kernel-based alternative losses in the alternative-objectives taxonomy. Sriperumbudur–Fukumizu–Lanckriet 2011 is the standard reference for the "is the kernel rich enough to characterize the relevant family of distributions" property that any HSIC-based strictness argument hinges on.

## Specific anchors
- Theorem 9 (the main equivalence of universal and characteristic kernels on locally compact Hausdorff groups) is the load-bearing result for §3.7's claim that HSIC with a characteristic kernel is a strict divergence.

## Notes
- Not cited inline in v7; `\nocite` only. The Part II agent should confirm the §3.7 HSIC discussion's exact attribution — Gretton et al. 2005 is the HSIC source, and Sriperumbudur 2011 is the characteristic-kernel source, so both are appropriate as a pair.
