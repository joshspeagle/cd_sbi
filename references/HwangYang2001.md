# Hwang & Yang, 2001 — An Optimality Theory for Mid p-Values in 2×2 Contingency Tables

**Authors:** J. T. Gene Hwang and Ming-Chung Yang
**Year:** 2001
**Venue:** Statistica Sinica, 11(3), 807–826
**DOI:** N/A (open access: https://www3.stat.sinica.edu.tw/statistica/oldpdf/A11n313.pdf)

## One-paragraph summary

Hwang & Yang provide a decision-theoretic justification for Lancaster's
mid-p value in the 2×2 contingency-table setting. They apply the
Neyman–Pearson fundamental lemma combined with an "estimated truth"
approach (viewing p-values as estimators of the indicator function for
the null hypothesis) to derive **expected p-values**, optimal under
a class of proper loss functions. Their key result is that, in the
one-sided case, the expected p-value coincides exactly with the mid-p
value of Lancaster 1961, and in the two-sided balanced binomial case
it reduces to a two-sided mid-p. Numerical comparisons show
expected/mid-p tests have type-I error very close to nominal, in
contrast to Fisher's conservative exact test and the anti-conservative
normal-approximation test.

## Why CD-SBI cites it

CD-SBI cites Hwang & Yang 2001 at §5.7.3 to justify the mid-p
alternative (deterministic `U = 1/2`) to the randomized PIT —
specifically as "the optimality theory of mid-p in the closely related
contingency-table setting." The citation is positioned as theoretical
backing for an applied recommendation: when the auxiliary randomness
is undesirable, the mid-p variant is the principled deterministic
choice.

## Specific anchors

- **Optimality result.** Theorem 3.1 + the explicit reduction to the
  mid-p in the one-sided case (and the balanced 2×2 case) is the core
  result CD-SBI invokes.
- **Scope.** Hwang & Yang's result is stated for 2×2 contingency
  tables under three sampling schemes (two-binomial, multinomial,
  two-Poisson). It does **not** establish a general optimality for
  mid-p across arbitrary discrete one-parameter exponential families.
  The CD-SBI manuscript phrases this carefully ("the closely related
  contingency-table setting") rather than overclaiming.
- **The (1/2) sup p_θ(t) Kolmogorov-distance bound.** The §5.7.3
  manuscript claim
  `K(θ) = (1/2) sup_t p_θ(t)`
  is **not** from Hwang & Yang 2001 — H&Y's analysis is
  decision-theoretic, not metric-theoretic. The Kolmogorov-distance
  bound is a direct calculation from the definition of the mid-p
  CDF (the manuscript even gives the achieving point `v = F_θ(t_*) -
  (1/2) p_θ(t_*)`) and does not require an external citation. The
  asymptotic refinement `sup_t p_θ(t) = Θ(1/√n)` for `Bin(n, θ)` at
  moderate θ is the local CLT (de Moivre–Laplace), standard textbook.

## Notes

- Bibliographic metadata verified at
  https://www3.stat.sinica.edu.tw/statistica/j11n3/j11n313/j11n313.htm
  (Statistica Sinica's official record) and the PDF is open access
  via Statistica Sinica.
- Attribution is correct as cited.
- The §5.7.3 sentence "See \citet{HwangYang2001} for the optimality
  theory of mid-p in the closely related contingency-table setting"
  is appropriately hedged with "closely related" — H&Y do not prove
  optimality for binomial-only settings, only for 2×2 tables.
