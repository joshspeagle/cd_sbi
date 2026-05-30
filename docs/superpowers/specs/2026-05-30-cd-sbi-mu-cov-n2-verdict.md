# CD-SBI bivariate-normal (μ, Σ) — N2 Stage-A verdict + μ₂ investigation

**Milestone:** N2 (Stage-A diagnostics + replication) for the d=5 unknown-(μ,Σ)
target. **Branch:** `feat/musigma-m3-verdict`. **Design:**
`2026-05-30-cd-sbi-bivariate-normal-mu-cov-design.md`.

---

## 1. Headline: the framework generalizes to d=5

The Wishart **Bartlett-decomposition** closed-form pivot (the multivariate analog
of the 1-D χ²/Student-t structure) drops cleanly into the existing
`SingleIndexMonotoneFlow` — **no architecture change**. Stage-A validation:

- **`r*` is N(0,I₅) at truth** (N1: all 5 coords KS ≤ 0.011, correlations ≈ I).
- **4 of 5 pivot coordinates calibrate to the noise floor** (per-coord PIT KS
  ~0.02–0.05 across θ₀, incl. extremes): `ℓ₁₁, ℓ₂₂` (covariance χ²), `L₂₁`
  (off-diagonal), `μ₁` (mean).
- **Aggregate calibration all holds:** 3 covariance χ² marginal-CD KS ≤ 0.047;
  the **joint Hotelling-T² μ-marginal** KS ≤ 0.019 (the multivariate
  nuisance-marginalized inference recovered — the analog of the 1-D Student-t);
  joint Mahalanobis ‖r‖²~χ²₅ KS 0.017; final loss above the entropy floor (2.26 vs
  H 2.04 finite; ≈floor+0.02 in the clean regime) — no cheat.
- New harness: `entropy_lower_bound` (triangular 5×5 floor, ≈2.03),
  `analytic_marginal_cd_pit` (3 χ² + Hotelling), `MultivariateMarginalCDRecovery`
  (covariance direct PITs + μ-Hotelling-recovery via `autoregressive_invert`),
  16-point LHS coverage grid, `paper_table_mu_cov`.

**Conclusion: the d=5 (μ,Σ) confidence distribution calibrates** — the Bartlett
pivot + single-index flow is the right generalization.

## 2. The documented limitation: μ₂'s mild miscalibration

The one caveat, found by a sharp **per-coordinate** PIT (the aggregate χ²₅ /
Hotelling checks mask it): **μ₂ — the doubly-cross-coupled mean coordinate** (its
truth `r*₅ = √n/C₂₂·(μ₂−X̄₂) − √n L₂₁/(C₁₁C₂₂)·(μ₁−X̄₁)` depends on μ₂, μ₁, *and*
all three covariance parameters) — is **mildly miscalibrated at extreme θ₀**:
per-coord PIT KS up to ~0.13 (finite-sample) / ~0.05 (clean), vs the ~0.02 floor.
Central-region coverage is fine; the deviation grows toward the prior's edges.

### The investigation (what it is NOT, and what it IS)

A multi-probe investigation **ruled out** every cheap explanation:

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Finite-sample artifact | **ruled out** | clean (infinite-sample) regime still mildly off (KS ~0.05 at extremes) |
| Slow convergence | **ruled out** | 12k→24k steps: μ₂ loss gap 0.014→0.015 (no improvement) |
| Capacity | **ruled out** | hidden 64→128: no improvement |
| G-curvature (UMNN adds nonlinearity) | **ruled out** | G₅'s derivative CoV 0.009–0.015 — G₅ is *linear* (vs the χ² coord's 0.136) |
| w_θ≠w_f asymmetry (μ−X̄ distorted) | **ruled out** | \|w_θ\|≈\|w_f\| for the mean coords (3.59≈3.54) |

**What it IS:** a **ctx-MLP expressivity limit on the affine index `z₅`** at the
prior's edges. The per-coordinate loss gap is **concentrated in μ₂** (0.014 nats;
others ~0) and **persistent** (a near-flat loss direction — the `w×G`-slope
redundancy makes μ₂'s scale weakly identified). The ctx-conditioned MLPs
(`w_θ(ctx)=√n/C₂₂` and the cross-term `off₅(ctx)`) don't perfectly track `r*₅`'s
θ₀-dependence at extreme θ₀.

**The identity-G test** (constrain `G_k=identity` for the linear coords, removing
the `w×G` redundancy) **confirmed the diagnosis without fixing it**: it sharpened
the *scale* (std → 0.99–1.04) but exposed a *shape* mismatch at extreme θ₀
(KS 0.05→0.16). The two G-choices merely **trade which moment absorbs the residual**
(UMNN-G keeps shape, loses scale; identity-G keeps scale, loses shape) — neither
eliminates it, because the residual lives in `z₅`'s ctx-dependence, not in G.

### Severity and decision

The miscalibration is **mild** (KS ≤ ~0.13; std within ~0.9–1.3; central coverage
fine) and confined to the hardest of five coordinates at the prior's edges.
**Decision (accepted):** document it as a known single-index-flow limit on the most
cross-coupled coordinate, rather than invest in a richer index (a 2-feature index
for μ₂, a feature-rectifier, or a monotone-MLP combiner — candidate future work).
The N2 intensive test (`tests/intensive/test_replicate_mu_cov.py`) reflects this:
the **aggregate calibration + the 4 well-calibrated coords are hard gates**; μ₂'s
mild limit is **pinned as a regression** (per-coord PIT KS < 0.16), precedent
`tests/ablation/test_trained_folding.py`.

## 3. Why pivot-RMSE-vs-`r*` was the wrong primary metric

The N1/early-N2 band `RMSE(r_trained, r*) < 0.15` failed on μ₂ (0.35) and is what
first flagged the issue — but it is the **wrong primary criterion**: `r*` is *one
specific* calibrated pivot (the KR rearrangement), and calibration only requires
the trained pivot to lie **on the manifold M** (`r~N(0,I)` at truth), not to match
`r*` pointwise. A different-but-calibrated pivot has RMSE>0 from `r*`. The sharp
per-coordinate PIT is the correct calibration check — and it confirmed that 4/5
coords are exactly calibrated and μ₂ is *mildly* off (a real but small deviation,
not the 0.35-sized failure the RMSE metric suggested). RMSE-vs-`r*` is retained
only as a **loose recovery sanity** (< 0.25).

## 4. Open thread (carried forward)

A richer ctx-conditioned index for cross-coupled coordinates (2-feature index /
feature-rectifier / monotone-MLP combiner) would likely push μ₂ to the floor — a
candidate refinement if/when the (μ,Σ) target needs edge-θ₀ μ-precision. Deferred;
the framework generalization (the N2 headline) stands without it.

**Next:** N3 — Stage-B I-A invertible learned summary `ℝ²⁰→ℝ⁵`.
