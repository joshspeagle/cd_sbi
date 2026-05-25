# Wehenkel & Louppe, 2019 — Unconstrained Monotonic Neural Networks

**Authors:** Antoine Wehenkel, Gilles Louppe
**Year:** 2019
**Venue:** Advances in Neural Information Processing Systems 32 (NeurIPS 2019), Vancouver
**arXiv:** 1908.05164
**DOI:** n/a (NeurIPS proceedings; DBLP/DOI via dl.acm.org 10.5555/3454287.3454425)

## One-paragraph summary
Proposes the **UMNN** architecture: a scalar function F(x; ψ) that is strictly monotone-increasing in x by construction, parameterized as the integral of a strictly positive free-form neural network from a reference point. Concretely (paper eq. 1): F(x; ψ) = β + ∫₀ˣ f(t; ψ) dt, where f : ℝ → ℝ₊ is the output of an unconstrained MLP passed through an **ELU activation increased by 1** (ELU(·) + 1) to force positivity. The forward pass evaluates the integral by **Clenshaw–Curtis quadrature**, exploiting the property that periodic-extension-via-cosine-transform gives exponential convergence for Lipschitz integrands. The backward pass uses the Leibniz rule (eq. 2–3) to compute ∇_ψ F as an integral of ∇_ψ f, avoiding the linear memory cost of backpropagating through the quadrature solver. Composed with autoregressive masking (MADE-style), the resulting **UMNN-MAF** is a universal density approximator for continuous random variables on ℝ^d (universality sketched in §3.3 via the universal approximation theorem applied to monotone scalar functions).

## Why CD-SBI cites it
- **§3.5 (line ~643):** named as the canonical "monotonicity-as-architecture" prescription for enforcing the (R2) regularity condition (strict monotonicity in X), the key fix for the autograd-folding pathology of §8.4.
- **§7.1 (line ~1566):** UMNN is invoked as the building block for the 1D experiments and as one of the two ingredients for the multivariate doubly-monotone construction in §6.3 / §7.1.

## Specific anchors
- **UMNN definition (eq. 1):** F(x; ψ) = ∫₀ˣ f(t; ψ) dt + β, with f the output of an MLP passed through ELU + 1 to enforce strict positivity.
- **Forward integration (§2, "Forward integration"):** uses **Clenshaw–Curtis quadrature**, motivated as exponential convergence for Lipschitz integrands.
- **Backward integration (eq. 2–3):** the Leibniz-rule trick that makes memory cost independent of the number of quadrature steps.
- **UMNN-MAF / universality (§3.3, §4 "Universality"):** UMNN composed with MADE-style autoregressive masking is a universal density approximator for continuous random variables.

## Notes
- **Attribution discrepancy in §7.1 (⚠).** The manuscript's §7.1 definition reads "g(z; c) = bias(c) + ∫ softplus(MLP(t, c)) dt" and "We use Gauss–Legendre quadrature (n = 12 nodes by default) to evaluate the integral." Both specifics are inconsistent with W&L 2019:
  - W&L use **ELU(·) + 1**, not softplus, as the positivity-inducing activation (paper eq. 1, "an ELU activation unit increased by 1"). The reference implementation (github.com/AWehenkel/UMNN, `models/UMNN/MonotonicNN.py`) also uses ELU+1.
  - W&L use **Clenshaw–Curtis** quadrature, not Gauss–Legendre (paper §2 "Forward integration," and `models/UMNN/NeuralIntegral.py` comment "Clenshaw-Curtis Quadrature Method").
  - The W&L paper's abstract claim is more general: "a free-form neural network whose only constraint is for its output to remain strictly positive" — softplus is one valid choice, Gauss–Legendre is one valid quadrature. So the manuscript's §7.1 choices are not *wrong* as a UMNN-style architecture; they are CD-SBI's specific implementation choices that *substitute for* W&L's specific choices, while preserving the architectural principle.
  - **Recommended fix:** §7.1 currently presents these as if they were part of W&L's "Definition (UMNN; Wehenkel & Louppe 2019)." Either (a) reframe the definition abstractly per W&L ("integral of any strictly positive parametric function"), then state CD-SBI's choices (softplus + Gauss–Legendre) as the deployed instantiation, or (b) keep the current form but add an explicit note that softplus and Gauss–Legendre are CD-SBI's substitutions for W&L's original ELU+1 and Clenshaw–Curtis.
- **The §3.5 mention is fine (✓).** "Represent a scalar monotone function as the integral of a positive function from a free neural net" is faithful to W&L's actual contribution at the abstract level.
- **Quadrature choice is a real design point.** Clenshaw–Curtis (W&L) has exponential convergence for Lipschitz integrands and the cosine-grid weights cache nicely; Gauss–Legendre (CD-SBI §7.1) gives exact integration of polynomials up to degree 2n-1 and is the textbook optimal-for-smooth-integrands choice. For softplus(MLP) integrands the difference at n = 12 is likely negligible (both reach near-machine precision), but it is a *substantive* design choice that deserves explicit framing — especially since one of the manuscript's open problems (§11.5) is higher-d scaling, where quadrature cost is non-trivial.
- The paper is also widely-cited as the basis for the *monotone-in-the-conditioning-variable* trick used downstream in flow architectures; the CD-SBI doubly-monotone construction (§6.3 / §7.1 "doubly-monotone UMNN form") generalizes W&L's "monotone in the integration variable" to a two-variable monotone-in-both setting, which is a real contribution worth flagging as such rather than implicit in the W&L reference.
