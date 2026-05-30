# CD-SBI bivariate-normal unknown (μ, Σ) — design

**Status:** validated design (brainstorm 2026-05-30; N0 risk-gate prototype passed).
Ready for plans. **Target:** bivariate normal `Xᵢ ~ N(μ, Σ)`, `n_iid = 10`
observations per dataset, `X ∈ ℝ^{n_iid × 2}` (flattened ℝ²⁰). Parameters:
`μ ∈ ℝ²` + `Σ` SPD 2×2 → **d_theta = 5**; sufficient statistic `(X̄ ∈ ℝ²,
scatter A ∈ SPD)` → **5-D**. The multivariate generalization of the landed
unknown-(μ,σ²) work (d_theta 2→5, suff-stat 2→5, data 10→20).

**Lineage / what carries over:** this reuses, essentially unchanged, the
`SingleIndexMonotoneFlow` (now d=5), the Stage-A→Stage-B structure, the M3.0
harness (`SufficiencyRecovery`, `FloorIntegrity`, `MarginalCDRecovery`, `Coverage`,
`JointMahalanobis`), and the **Stage-B verdict** (only the information-preserving
**I-A invertible exact-density** arm calibrates a learned summary — see
`2026-05-29-cd-sbi-stage-b-bakeoff-verdict.md`). Per that verdict we go **straight
to Arm I-A** for the learned summary and do **not** re-run I-B/II-A.

---

## 1. Why it generalizes — the Wishart Bartlett decomposition

The 1-D case worked because `(n−1)s²/σ² ~ χ²` gave a clean `r*_σ`, plus an `X̄`-based
`r*_μ`. The multivariate analog is the **Bartlett decomposition**, which is
*inherently triangular* — a Knothe–Rosenblatt rearrangement by construction, so it
matches the autoregressive single-index flow:

- `X̄ ~ N(μ, Σ/n)`, independent of the scatter `A = Σᵢ(Xᵢ−X̄)(Xᵢ−X̄)ᵀ` (Basu).
- `A ~ Wishart₂(n−1, Σ)`. With `Σ = C Cᵀ` (Cholesky) and `A = D Dᵀ` (Cholesky of the
  *data's* scatter), the Bartlett factor is `T = C⁻¹ D` (lower-triangular), whose
  entries are **independent**: `T₁₁² ~ χ²_{n−1}`, `T₂₂² ~ χ²_{n−2}`, `T₂₁ ~ N(0,1)`.
- The mean pivot `√n·C⁻¹(μ − X̄) ~ N(0, I₂)` (Σ known), independent of the above.

So the closed-form joint pivot `r*(θ;X) ∈ ℝ⁵ ~ N(0, I₅)` is built triangularly.

### N0 prototype (passed — this design is grounded, not hoped)

A scratch prototype (`/tmp/proto_mvn_bartlett.py`, not committed) confirmed both
halves of the central claim on `n_iid=10`:
- **`r*` is N(0,I₅) at truth:** per-coord mean ≈ 0, std ≈ 1, KS vs N(0,1) ≤ 0.025.
- **`SingleIndexMonotoneFlow` (d=5, unchanged) recovers `r*`:** per-coord RMSE
  `(ℓ₁₁ 0.06, ℓ₂₂ 0.11, L₂₁ 0.11, μ₁ 0.05, μ₂ 0.14)`, overall **0.098** ≈ the 1-D
  0.10. **No feature-rectifier needed** (the design risk we flagged did not bite —
  the flow's ctx-conditioned magnitudes supply the data-dependent cross-scaling).

---

## 2. Parameterization, ordering, signs (all validated by N0)

**Parameterization — log-Cholesky.** `θ = (ℓ₁₁, ℓ₂₂, L₂₁, μ₁, μ₂)` with
`Σ = L Lᵀ`, `L = [[exp ℓ₁₁, 0], [L₂₁, exp ℓ₂₂]]`. Unconstrained ℝ⁵, triangular,
in the same Cholesky basis as the Bartlett pivot. Prior (matching the prototype):
`μ ~ U(−3,3)²`, `ℓᵢᵢ ~ U(log 0.4, log 2.5)`, `L₂₁ ~ U(−1.5, 1.5)`.

**Autoregressive (KR) order — diagonals → off-diagonal → mean:**
`(ℓ₁₁, ℓ₂₂, L₂₁, μ₁, μ₂)`. Rationale (settles the manuscript's §11 ordering
question for this target): the two diagonal covariance pivots are "pure χ²"
(`T₁₁`, `T₂₂` need only their own `Cᵢᵢ`); `T₂₁` needs *both* diagonals (so it must
come after them); the mean pivots need the full `C⁻¹` and are μ-free in the
covariance block (Basu), so they come last, conditioned on the covariance coords.
N0 confirmed this order is single-index-representable.

**Closed-form `r*` (increasing-θ convention; the Stage-A truth):**
- `r₁ = Φ⁻¹(1 − F_{χ²_{n−1}}(T₁₁²))`, `T₁₁ = D₁₁/C₁₁`
- `r₂ = Φ⁻¹(1 − F_{χ²_{n−2}}(T₂₂²))`, `T₂₂ = D₂₂/C₂₂`
- `r₃ = T₂₁ = (D₂₁ − (L₂₁/C₁₁)·D₁₁)/C₂₂`
- `r₄ = √n·(μ₁ − X̄₁)/C₁₁`
- `r₅ = √n·[ −(L₂₁/(C₁₁C₂₂))·(μ₁ − X̄₁) + (μ₂ − X̄₂)/C₂₂ ]`
where `C₁₁=exp ℓ₁₁`, `C₂₂=exp ℓ₂₂`, and `D` = Cholesky of the data scatter `A`.

**Per-coordinate signs (read off `r*`, validated):**
`theta_signs = [+1, +1, −1, +1, +1]`, `feat_signs = [−1, −1, +1, −1, −1]`.
**Oracle features (the conditioner output, in coord order):**
`(log D₁₁, log D₂₂, D₂₁, X̄₁, X̄₂)`.

---

## 3. Components (mirror the 1-D stack)

- **`NormalBivariateUnknownCov` simulator.** `sample`, `sample_x_given_theta`,
  the Bartlett `r_star` (§2), `theta_signs`/`feat_signs` properties,
  `oracle_summary(X) → (log D₁₁, log D₂₂, D₂₁, X̄₁, X̄₂)`, `entropy_lower_bound`
  (NF-MLE floor on the oracle-feature scale), `data_entropy_lower_bound`
  (= `H(X|θ)` for `X ~ N(μ, Σ)` over n_iid: `n·[½ p(1+log2π) + log|L|]` averaged
  over the prior, `p=2`), and marginal-CD references (§4).
- **`BartlettSummaryConditioner` (oracle).** `encode(X) → ((log D₁₁, log D₂₂,
  D₂₁, X̄₁, X̄₂), const log-det)` — the Cholesky/Bartlett transform of the scatter
  + the sample mean; θ-independent log-det constant (sufficiency).
- **`SingleIndexMonotoneFlow(d=5)`** — unchanged; signs injected from the simulator
  at wire-time. (N0: recovers `r*` to RMSE 0.098.)
- **Stage-B learned summary (I-A only):** `AffineCouplingBijection` on ℝ²⁰ +
  `InvertibleSummaryConditioner(d_theta=5)` (S∈ℝ⁵ inference block + A∈ℝ¹⁵
  ancillary) + `ExactDensityCDSBIRunner` — all already exist; generalize their
  dims from (2, 10) to (5, 20).

---

## 4. Harness at d=5

- **`MarginalCDRecovery`** generalizes:
  - the **three covariance** marginals read off the Bartlett pivots (the two
    diagonal ones are χ²; the off-diagonal `T₂₁` is N(0,1) directly);
  - the **μ marginal**, integrating Σ out of the joint CD, is **Hotelling-T²**
    (the multivariate Student-t, the direct analog of the 1-D t-marginalization):
    `n(μ−X̄)ᵀ S⁻¹(μ−X̄) · (n−p)/(p(n−1)) ~ F_{p, n−p}`. The diagnostic gets a
    `simulator.analytic_marginal_cd_pit` returning the χ²/χ²/N (covariance) and the
    F/Hotelling (mean) PITs at truth, KS-compared to U(0,1).
  - **Validate the μ-marginalization numerically first** (as we did the 1-D
    Student-t to ~1e-4) before wiring the diagnostic — the multivariate
    marginalization is the delicate piece.
- **`Coverage` — LHS θ₀ grid.** 3⁵ = 243 product points is infeasible; use a
  **Latin-hypercube sample of ~16–24 θ₀** over the 5-D prior box (implements the
  manuscript's deferred "high-d θ₀-grid strategy"). `JointMahalanobis` (‖r‖²~χ²₅)
  and `FloorIntegrity` work unchanged (the loss→floor map already covers NF-MLE +
  ExactDensity).

---

## 5. Decomposition

- **N0 — single-index adequacy prototype. ✅ DONE** (scratch; RMSE 0.098, no
  rectifier needed). Recorded here; not committed code.
- **N1 — Stage-A core.** `NormalBivariateUnknownCov` (+ Bartlett `r*`, signs,
  oracle_summary, entropies) + `BartlettSummaryConditioner` + d=5 flow wiring +
  configs + recovery smoke (expect RMSE ≲ 0.15).
- **N2 — Stage-A diagnostics + replication.** `MarginalCDRecovery` generalization
  (χ² covariance marginals + Hotelling-T² mean, μ-marginalization validated
  first), LHS coverage grid, `paper_table_mu_cov`, intensive replication
  (recovery + marginal-CD + coverage + floor).
- **N3 — Stage-B I-A.** Generalize `InvertibleSummaryConditioner`/
  `ExactDensityCDSBIRunner` to (d_theta=5, d_x=20); intensive harness verdict —
  does the learned invertible summary recover the **covariance** directions
  (Spearman) and calibrate? (Hypothesis from the 1-D verdict: yes.)

---

## 6. Success criteria

- **N1:** trained flow recovers `r*` (overall RMSE ≲ 0.15; N0 got 0.098 against
  the analytic pivot, training adds slack) and `JointMahalanobis` ‖r‖²~χ²₅ passes
  at the oracle summary.
- **N2:** coverage error ≤ 0.05 across the LHS grid; covariance marginal-CD KS and
  μ Hotelling-T² KS within the harness bands (≤ 0.06); loss at the entropy floor.
- **N3:** the learned I-A summary recovers all 5 sufficient directions
  (`SufficiencyRecovery` Spearman > 0.9 each — **the 3 covariance directions
  included**, the analog of σ² in 1-D) and calibrates near the oracle, loss above
  `H(X|θ)` (no cheat).
- **Milestone:** a fully multivariate (d=5), data-driven (learned-summary)
  calibrated confidence distribution for `(μ, Σ)` — the headline generalization.

---

## 7. Scope / open threads

**In scope:** the d=5 Stage-A oracle (Bartlett pivot, log-Cholesky, single-index
flow), the d=5 harness (Hotelling-T² marginal, LHS coverage), and the Stage-B I-A
learned summary. **Out of scope (deferred):** re-running I-B/II-A (verdict known);
permutation-equivariant / variable-`n_iid` bijections; p > 2 (general `p×p` Σ,
where the Bartlett structure still holds — this is the `p=2` stepping stone);
manuscript integration.

**Carried-in working principles:** ρ is not a prior (frequentist pointwise
coverage); regularity is architectural; validate the delicate math (the
μ-marginalization) numerically before building on it — the same disciplines that
made the 1-D work land.
