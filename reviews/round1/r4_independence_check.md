# (R4) Independence: Verification Report

## Verdict

**FALSIFIED — counterexample found.**

The leader's redundancy claim holds for `Bin(1, θ)` and `Bin(2, θ)` but **fails for `Bin(3, θ)`**.
The non-canonical permutation `(0, 2, 1, 3)` produces a `V'_θ` that satisfies (R1), (R2), (R3_U)
(with linear pieces, slopes equal to `p_θ(t)`, joint marginal uniform), yet violates (R4)
robustly across the full parameter space and the full `U`-interval.

Consequently:

- The leader's claims 1, 2 (Bin(2)-specific) are correct.
- Claim 3 ("under MLR-increasing on Bin(n, θ), only canonical-order partial sums of `p_θ(t)` over subsets are monotone non-increasing") is **wrong** as stated. It is wrong even for the very next case Bin(3): the partial sum `(1-θ)^3 + 3θ^2(1-θ)` (subset `{0, 2}`) has derivative `-3(2θ - 1)^2 ≤ 0`, i.e. it IS non-increasing on (0,1).
- Claim 4 (only canonical ordering yields all-non-increasing endpoints) was verified by the leader for Bin(2) **but extrapolated to higher n without checking**. The leader's brute-force enumeration appears not to have been run for n = 3.
- Claim 5 (the (R4)-redundancy conclusion) therefore does not hold. (R4) is genuinely independent.

(R4) should remain in the hypothesis list for Theorem C\*.

---

## Structural argument re-derivation

### Setup

Under MLR-increasing exponential family with sufficient statistic `T(X)` and `Theta = (0,1)`:

- `V_θ(t, ·)` must be strictly monotone in `U` (by R3_U) and is a bijection onto a measurable image set `A_θ(t) ⊆ [0,1]` of Lebesgue measure `p_θ(t)` (by R2).
- The image `A_θ(t)` is automatically a countable union of intervals (since `V_θ(t, ·)` is monotone on `[0,1]`, hence has only countably many jumps; between jumps the image is an interval).
- Uniformity of the joint mixture (V calibrated) **forces the within-piece slope** `dV_θ(t, U)/dU = p_θ(t)` wherever defined. Equivalently: if `A_θ(t)` decomposes as `⋃_k [L_{t,k}, R_{t,k}]` with `U`-range `[u_{t,k-1}, u_{t,k}]`, then `u_{t,k} - u_{t,k-1} = (R_{t,k} - L_{t,k})/p_θ(t)`.
- Pieces of all `A_θ(t)` partition `[0, 1]` for each θ.

### What (R1) actually pins down

For a fixed `(t, U)`, `V_θ(t, U)` evolves with `θ` along two channels:

(a) Within a fixed piece: linear in `U`, so `d/dθ V_θ(t, U) = L'_{t,k}(θ) + (slope · U-offset)' `. Because the slope is exactly `p_θ(t)`, by linearity in `U` it suffices to check the two endpoint derivatives `L'_{t,k}(θ) ≤ 0` and `R'_{t,k}(θ) ≤ 0`. (R1) within a piece is equivalent to both piece endpoints being non-increasing in θ.

(b) Across a piece boundary in θ: as θ varies, the `U`-position `u_{t,k}(θ)` of the boundary inside `T = t`'s image shifts. A fixed `U` can be kicked into an adjacent piece. For `V_θ(t, U)` to remain non-increasing through this transition, the jump direction at the boundary must be downward. Since within `T = t`'s image consecutive pieces ascend in `[0,1]` (by (R3_U) — strict monotone in U), the lower-piece-to-upper-piece transition is an upward jump; we therefore need `u_{t,k}(θ)` to be **non-decreasing in θ** (so that increasing θ pushes `U` from upper piece into lower piece, a downward jump).

### Where the leader's argument has a gap

The leader's claim 3 conflates two things:

> *"the endpoints are partial sums of `p_θ(t)` over subsets `S ⊆ {0, …, n}`"*

This is true **only for partitions that are "block-wise"** — every atom `T = t` is contained entirely in one piece. For block-wise tilings, claim 3 reduces to "which subsets of `{0, …, n}` have non-increasing total mass in θ?" For Bin(2) the answer is exactly the downward-closed sets `{}, {0}, {0,1}, {0,1,2}` — these are the canonical-prefix subsets — so the leader's reasoning is correct for Bin(2).

**But for Bin(3, θ), claim 3 fails.** The subset `{0, 2}` has total mass

```
P_θ(T ∈ {0, 2}) = (1-θ)^3 + 3 θ^2 (1-θ),
```

with derivative

```
d/dθ [(1-θ)^3 + 3 θ^2 (1-θ)] = -3(2θ - 1)^2 ≤ 0,
```

non-positive on all of (0, 1) (with weak equality only at θ = 1/2, which does not violate "non-increasing"). So `{0, 2}` is another downward-closed-in-mass set in the Bin(3) tiling.

This permits the block-wise tiling **(0, 2, 1, 3)**: T=0 first, then T=2, then T=1, then T=3, all as single intervals. Every endpoint of this tiling is non-increasing in θ, so (R1) holds within each piece. And (R4) is violated by construction (T=2 sits below T=1).

So the structural argument fails at claim 3 by missing that "downward-closed in θ-mass" is **not the same as** "downward-closed in the T-order" once we go past n = 2.

### Why Bin(2) is special

For Bin(2), the only non-trivial proper subsets of {0,1,2} are {0}, {1}, {2}, {0,1}, {0,2}, {1,2}. Direct calculation:

- {0}: mass `(1-θ)^2`, derivative `-2(1-θ)` ≤ 0. ✓
- {0,1}: mass `1 - θ^2`, derivative `-2θ` ≤ 0. ✓
- {2}: mass `θ^2`, derivative `2θ` ≥ 0. ✗
- {1}: mass `2θ(1-θ)`, derivative `2(1-2θ)` non-monotone. ✗
- {0,2}: mass `1 - 2θ(1-θ)`, derivative `-2(1-2θ)` non-monotone. ✗
- {1,2}: mass `1 - (1-θ)^2`, derivative `2(1-θ)` ≥ 0. ✗

Only the canonical prefixes work, as the leader observed. **What goes wrong with the extrapolation to n = 3 is purely combinatorial**: in dimension `n + 1 = 4`, the partial sums acquire new non-monotone candidates that happen to factor as squared polynomials with definite sign. Specifically the swap `(0, 2, 1, 3)` has partial sums `(1-θ)^3`, `(1-θ)^3 + 3 θ^2(1-θ) = 1 - 3 θ(1-θ)(?)` ... let me factor it differently:

```
(1-θ)^3 + 3 θ^2(1-θ) = (1-θ) [(1-θ)^2 + 3 θ^2] = (1-θ)(1 - 2θ + 4 θ^2).
```

Derivative is `-3(2θ-1)^2` — a perfect square with a minus sign. This kind of accidental non-positive structure does not exist for Bin(2)'s candidate subsets, but does exist for Bin(3) and presumably for higher n (more subsets to "luck out" with).

### Multi-piece tilings (also fail for Bin(2) but succeed for Bin(3))

I also checked multi-piece (non-block) tilings for Bin(2), to test whether the leader's claim survives multi-piece freedom:

- Multi-piece-T=2 (T=2 split into a lower piece below T=1 and an upper piece above T=1): the U-position of the T=2 piece-boundary, `frac(θ) = α(θ)/θ^2`, must be **non-decreasing** in θ to avoid an upward jump in `V_θ(2, U)` at fixed U as θ crosses the transition. Solving the resulting differential constraints (α(θ) ≤ θ(1-θ) for in-piece (R1), α/θ^2 non-decreasing) forces `α ≡ 0` globally on (0,1). So multi-piece-T=2 does NOT escape the redundancy for Bin(2).
- Multi-piece-T=1 (T=1 split with T=2 between the two T=1 pieces): same analysis. At θ > 1/2 the constraints force β(θ) (mass of T=1 below T=2) to equal the full T=1 mass (i.e. degenerate to canonical); at θ < 1/2 a non-degenerate two-piece T=1 is locally consistent. But any global construction with β changing from "non-degenerate" to "degenerate" at θ = 1/2 introduces a discontinuity in T=2's piece location in (R1) at θ = 1/2, violating (R1).

So for Bin(2) the leader's redundancy claim survives multi-piece extensions: it is a theorem.

For Bin(3), **multi-piece freedom is unnecessary** because the (0, 2, 1, 3) block-wise tiling already works.

---

## Counterexample search

### Bin(2, θ) — confirmed redundancy

For Bin(2, θ) the enumeration of all 6 permutations of {0, 1, 2} confirms only the canonical (0, 1, 2) ordering has all partial sums non-increasing in θ on (0, 1). Multi-piece extensions (T=2 split, T=1 split, T=0 split) all collapse to canonical under the joint constraints (R1) ∧ uniformity ∧ (R3_U) ∧ continuity-of-V-in-θ implicitly required by (R1). So in Bin(2), (R4) really is redundant given (R1) + (R3_U) + (R2).

### Bin(3, θ) — counterexample found

Enumeration of all 24 permutations of {0, 1, 2, 3} shows TWO orderings with all partial sums non-increasing in θ:

| Permutation | A_1 | A_2 | A_3 | All non-increasing? |
|---|---|---|---|---|
| (0, 1, 2, 3) | `(1-θ)^3` | `(1-θ)^3 + 3θ(1-θ)^2 = 1 - 3θ^2 + 2θ^3` | `1 - θ^3` | ✓ canonical |
| (0, 2, 1, 3) | `(1-θ)^3` | `(1-θ)^3 + 3θ^2(1-θ) = (1-θ)(1 - 2θ + 4θ^2)` | `1 - θ^3` | ✓ swap |

The second is the swap T=1 ↔ T=2. The middle endpoint has derivative `-3(2θ - 1)^2`. This is a working counterexample. (See "If FALSIFIED" section below for the explicit construction and verification.)

### Larger n

For Bin(n, θ) with n ≥ 3, there are increasingly many endpoint-non-increasing permutations beyond the canonical one. I did not enumerate Bin(4) exhaustively, but the (0, 2, 1, 3) construction already shows the redundancy fails. Almost certainly Bin(4) admits permutations like (0, 3, 1, 2, 4) or similar.

### Other discrete MLR families

I did not search Geometric(θ), Poisson(λ), negative binomial, etc., because the Bin(3) counterexample already falsifies the leader's general redundancy claim. The structural reason — that partial-sum monotonicity over non-prefix subsets can hold accidentally via squared-polynomial-with-minus-sign — generalizes naturally (no MLR family is immune at scale).

---

## If FALSIFIED: explicit V'_θ

**Model.** `T = T(X) ∼ Bin(3, θ)`, atoms `{0, 1, 2, 3}` with masses
```
p_θ(0) = (1-θ)^3,  p_θ(1) = 3θ(1-θ)^2,  p_θ(2) = 3θ^2(1-θ),  p_θ(3) = θ^3.
```

**Construction.** Define the tiling of `[0, 1]`:

```
A_1(θ) := (1-θ)^3                            (right endpoint of T=0 piece)
A_2(θ) := (1-θ)^3 + 3θ^2(1-θ) = (1-θ)(1 - 2θ + 4θ^2)
A_3(θ) := 1 - θ^3                            (right endpoint of T=1 piece)
```

`V'_θ(t, U)` is piecewise linear in U with slope `p_θ(t)` on each piece:

```
V'_θ(0, U) = U · (1-θ)^3                                       (U in [0,1])
V'_θ(2, U) = A_1(θ) + U · 3θ^2(1-θ)                            (U in [0,1])
V'_θ(1, U) = A_2(θ) + U · 3θ(1-θ)^2                            (U in [0,1])
V'_θ(3, U) = A_3(θ) + U · θ^3 = 1 - θ^3(1 - U)                 (U in [0,1])
```

Image sets: `A_θ(0) = [0, A_1]`, `A_θ(2) = [A_1, A_2]`, `A_θ(1) = [A_2, A_3]`, `A_θ(3) = [A_3, 1]`. These partition `[0, 1]` for every θ.

### Verification of (R1)

By symbolic differentiation:
```
∂_θ V'_θ(0, U) = -3 U (1-θ)^2                               ≤ 0 always
∂_θ V'_θ(2, U) = -3 (1-θ)^2 + 3 U θ(2 - 3θ)
                 endpoints: U=0 → -3(1-θ)^2 ≤ 0;
                            U=1 → -3(2θ-1)^2 ≤ 0  (algebraic identity)
                 linear in U ⇒ ≤ 0 for all U ∈ [0,1].          ≤ 0 always
∂_θ V'_θ(1, U) = (-3(2θ-1)^2) + U · 3(θ-1)^2 · (something)
                 endpoints: U=0 → -3(2θ-1)^2 ≤ 0;
                            U=1 → -3θ^2 ≤ 0
                 linear in U ⇒ ≤ 0 for all U ∈ [0,1].          ≤ 0 always
∂_θ V'_θ(3, U) = 3 θ^2 (U - 1)                              ≤ 0 always
```

So (R1) holds everywhere on `(0, 1) × [0, 1]`. The "weak equality at θ = 1/2" for the swap-piece endpoints does not violate (R1), which only requires non-increasing.

### Verification of (R2)

The four images `[0, A_1], [A_1, A_2], [A_2, A_3], [A_3, 1]` are disjoint (mod endpoints) and their union is `[0, 1]`. The map `(t, U) → V'_θ(t, U)` is a bijection onto `[0, 1]` for each θ.

### Verification of (R3_U)

Each `V'_θ(t, ·)` is linear with positive slope `p_θ(t) > 0` (since θ ∈ (0, 1)), hence strictly increasing in U on `[0, 1]`. No jumps in U.

### Verification of joint uniformity (the calibration manifold condition)

By construction the slope `dV'_θ(t, U)/dU` equals `p_θ(t)`, so the density of `V'_θ(X, U)` at any `v ∈ A_θ(t)` is `p_θ(t) · 1/p_θ(t) = 1`. Joint marginal is `U(0,1)`. (Numerical Monte Carlo over 200,000 samples at `θ ∈ {0.2, 0.5, 0.7, 0.9}` shows histograms uniform to within sampling noise.)

### Verification of (R4) violation

`V'_θ(1, U) - V'_θ(2, U) = 3θ^2(1-θ) + 3 U θ(1-θ)(1-2θ)`. At `U = 0`: equals `3θ^2(1-θ) > 0` for all `θ ∈ (0, 1)`. So `V'_θ(1, U) > V'_θ(2, U)` at `U = 0` for every `θ ∈ (0, 1)`, violating (R4) (which requires `V'_θ(1, U) ≤ V'_θ(2, U)`).

Concrete instance: at `θ = 0.7, U = 0.5`:
```
V'_θ(0, 0.5) = 0.0135
V'_θ(1, 0.5) = 0.5625      ← above
V'_θ(2, 0.5) = 0.2475      ← below
V'_θ(3, 0.5) = 0.8285
```

Maximum (R4) violation across `(θ, U) ∈ (0, 1) × [0, 1]`: `V'_θ(1, U) - V'_θ(2, U) ≈ 0.444` at `θ ≈ 1/3, U = 1`.

### Sanity: V'_θ is not r*_rand

The canonical `V*_θ` has T=1 in `[A_1, A_1 + 3θ(1-θ)^2]` and T=2 in `[A_1 + 3θ(1-θ)^2, 1-θ^3]`. Our `V'_θ` puts T=2 below T=1. So `r'_rand = Φ^{-1}(1 - V'_θ) ≠ r*_rand`. Both are calibrated. Both satisfy (R1) ∧ (R2) ∧ (R3_U). They are distinguished only by (R4).

---

## If CONFIRMED: scope

Not applicable — the verdict is FALSIFIED.

For completeness, **the redundancy DOES hold for Bin(1, θ) and Bin(2, θ)** (small atom counts), but does NOT hold for Bin(n, θ) with `n ≥ 3`. The Bin(2) verification in the leader's argument is genuine; only the extrapolation to higher n was unjustified.

---

## Recommendation to leader

**Keep (R4) in T-C\*'s hypothesis list.** The Bin(3) construction `(0, 2, 1, 3)` is a clean, robust counterexample with (R1), (R2), (R3_U) all satisfied and (R4) violated everywhere on `(0, 1) × [0, 1]`. It uses only single-interval pieces with the forced uniformity slope, so it is unambiguous.

### Replacement for the manuscript's Bin(1, θ) explicit counterexample

Use the Bin(3, θ) `(0, 2, 1, 3)` construction. Suggested text (replaces lines ~1153–1172):

> **Explicit counterexample.** Take `X ∼ Bin(3, θ)` with `T(X) = X ∈ {0, 1, 2, 3}` and masses `p_θ(t) = \binom{3}{t} θ^t (1-θ)^{3-t}`. The canonical `V*_θ` tiles `[0, 1]` in the order `(T=0, T=1, T=2, T=3)`. Consider instead the tiling in the order `(T=0, T=2, T=1, T=3)`:
> ```
> V'_θ(0, U) = U(1-θ)^3
> V'_θ(2, U) = (1-θ)^3 + 3 U θ^2(1-θ)
> V'_θ(1, U) = (1-θ)^3 + 3θ^2(1-θ) + 3 U θ(1-θ)^2
> V'_θ(3, U) = 1 - θ^3(1 - U).
> ```
> Direct computation gives `∂_θ V'_θ(t, U)` non-positive on `(0,1) × [0,1]` for every `t ∈ {0,1,2,3}` (the binding term is `−3(2θ−1)^2 ≤ 0` arising at the swap-piece endpoints), confirming (R1). Each `V'_θ(t, ·)` is strictly increasing on `[0, 1]`, confirming (R3_U). The map `(t, U) ↦ V'_θ(t, U)` is a bijection of `\{0,1,2,3\} × [0,1]` onto `[0, 1]`, confirming (R2). The joint mixture (sample `X` then `U`) is exactly `U(0, 1)` because the slope of `V'_θ(t, ·)` is `p_θ(t)` on its piece. But `V'_θ(1, 0) - V'_θ(2, 0) = 3θ^2(1-θ) > 0`, so `V'_θ(1, U) > V'_θ(2, U)` at `U = 0` (and a positive-measure neighbourhood) for every `θ ∈ (0, 1)`, violating (R4).

### Why the leader's "endpoint partial-sum" intuition is *almost* right

The leader's argument can be salvaged with the following correction. The endpoints of a block-wise tiling are partial sums of `p_θ(t)` over **subsets**, not just prefixes. The leader's "only canonical-order partial sums are non-increasing" is true for Bin(2) but **not for general MLR-increasing discrete models**. (R4) is needed precisely to exclude swaps within the interior of `T`-support that happen to be endpoint-monotone-in-θ. For Bin(2) such swaps don't exist, which is why the leader's Bin(2)-only check looked decisive but didn't generalize.

### Optional manuscript update (orthogonal to (R4))

The TODO comment at lines 1139–1148 should still note that the *original* Bin(1, θ) example doesn't separate (R4) from (R3_U); the replacement is the Bin(3) construction above (not the Bin(2) construction the round-1 review suggested — that one also fails to separate (R4), as I verified independently).
