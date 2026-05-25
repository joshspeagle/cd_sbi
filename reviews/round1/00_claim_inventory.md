# Round 1 — Claim & Evidence Inventory

**Document under review:** `cd_sbi_v7.tex`
**Round 1 scope:** factual / logical correctness of claims and proofs.
Numbers in tables (RMSE, KS, coverage) are assumed correct; do not
re-derive. Exposition / clarity issues are out of scope (deferred to
round 3).

## Schema

Each claim has:

- **ID** — short stable identifier (used in critic reports).
- **Statement** — one-sentence paraphrase.
- **Location** — section / equation anchor in `cd_sbi_v7.tex`.
- **Evidence** — `proof` / `empirical` / `definition` / `derivation` /
  `citation` / `informal-argument`.
- **Depends-on** — upstream claim IDs this rests on.
- **Type** — `load-bearing` (downstream proofs / theorems rest on it),
  `standalone` (positioning / background), or `illustrative` (worked
  example).
- **Status** — `pending` initially; updated to `holds` / `error` /
  `under-justified` / `resolved` / `deferred` / `escalated` during round
  1.

ID prefixes:

- `D-*` — definitions.
- `R-*` — regularity conditions.
- `C-N.M-*` — claims tied to subsection N.M.
- `L-N.M` — lemmas.
- `T-*` — theorems.
- `P-*` — propositions.
- `E-*` — experiments.
- `OP-*` — open problems.

---

## Part I — Framework (§1–§2)

### §1 Motivation

| ID | Statement | Location | Evidence | Depends-on | Type | Status |
|---|---|---|---|---|---|---|
| C-1.1-overconf | Standard SBI methods (NPE/NLE/NRE) produce systematically overconfident posteriors at realistic simulation budgets. | §1.1 | citation (Hermans et al. 2022) | — | standalone | pending |
| D-CD | Confidence distribution: data-dependent `H(·;X)` on Θ with `H(θ₀;X) ~ U(0,1)` when `X ~ P_{θ₀}`, for every `θ₀`. | §1.2 | definition (cites Schweder–Hjort 2016) | — | load-bearing | pending |
| C-1.2-LocNorm | For `X ~ N(θ,1)`, the UMPU CD is `H*(θ;X) = Φ(θ−X)`. | §1.2 | derivation + citation | D-CD | load-bearing | pending |
| C-1.3-C1 | Calibration (C1): `r(θ₀;X) ~ N(0,I_d)` under `X ~ P_{θ₀}` for every `θ₀`. | §1.3 | definition | — | load-bearing | pending |
| C-1.3-C2 | Monotonicity (C2): `θ_k ↦ r_k(θ;X)` strictly increasing (autoregressively in `d>1`). | §1.3 | definition | — | load-bearing | pending |
| C-1.3-rho | `ρ` is a sampling device, not a Bayesian prior; population-limit CD pivot does not depend on `ρ` provided full support. | §1.3 | informal-argument | — | load-bearing | pending |

### §2 Pivot and calibration manifold

| ID | Statement | Location | Evidence | Depends-on | Type | Status |
|---|---|---|---|---|---|---|
| D-pivot | Pivot `r: Θ × X → ℝ^d` measurable. | §2.2 | definition | — | load-bearing | pending |
| R1 | Monotonicity in θ (autoregressive in `d>1`). | §2.2 | definition | — | load-bearing | pending |
| R2 | Invertibility in X at fixed θ (C¹ or Lipschitz a.e. diffeomorphism). | §2.2 | definition | — | load-bearing | pending |
| D-M | Calibration manifold `M_F` within function class F. | §2.2 | definition | C-1.3-C1, D-pivot | load-bearing | pending |
| C-2.3-connected | (R1) ensures `H_r(·;X)` is a proper CDF on Θ with connected level sets; eliminates `Z_2^d` sign-flip ambiguity. | §2.3 | informal-argument | R1, D-M | load-bearing | pending |
| C-2.4-folding | Without (R2), `p̂_r` is not a valid density and NF-MLE has spurious sub-optima below the true conditional entropy; loss can dip arbitrarily low under folding. | §2.4 | informal-argument | R2, C-3.2-KL | load-bearing | pending |

## Part II — Loss design (§3)

### §3 NF-MLE

| ID | Statement | Location | Evidence | Depends-on | Type | Status |
|---|---|---|---|---|---|---|
| D-NF | Normalizing flow: parametric diffeomorphism transforming source density to standard Gaussian via change-of-variables `p̂_r = φ_d(r) · |det ∂r/∂X|`. | §3.1 | definition | — | load-bearing | pending |
| D-NFMLE | NF-MLE loss `L(r) = E[½‖r‖² − log|det ∂r/∂X|]` (the NLL under `p̂_r` minus constant). | §3.1 (boxed eqn) | definition | D-NF | load-bearing | pending |
| C-3.2-KL | `L(r) = E_ρ[KL(p(·|θ) ‖ p̂_r(·|θ))] + const`; minimizers over F (with R2) are exactly `M_F`. | §3.2 | proof (KL non-negativity) | D-NFMLE, R2 | load-bearing | pending |
| C-3.2-strictprop | NF-MLE is strictly proper for `M_F` within architectural class (no auxiliary indep/sharpness penalties needed). | §3.2 | follows from C-3.2-KL | C-3.2-KL | load-bearing | pending |
| C-3.3-manifold-large | Calibration manifold over unconstrained pivots is infinite-dimensional in 1D and rotation-ambiguous in d>1; architecture picks a unique element. | §3.3 | informal-argument | D-M | standalone | pending |
| C-3.4-SNL | NF-MLE loss `L(r)` is identical to the SNL loss; CD-SBI differs from SNL only in architectural constraints + inference machinery. | §3.4 | by inspection | D-NFMLE | standalone | pending |
| C-3.5-loss-below | Folded `p̂_r` is not normalized in X at fixed θ, so `−log p̂_r` has no lower bound at conditional entropy and loss can fall below truth. | §3.5 | derivation | C-2.4-folding | load-bearing | pending |
| C-3.7-Class1 | Class 1 (two-sample distance on marginal pushforward) is strictly proper for marginal calibration only — insufficient (Hermans pattern lives here). | §3.7 | informal-argument | — | standalone | pending |
| C-3.7-Class2 | Class 2 (marginal + HSIC) is strictly proper for M but empirically fragile (weak indep gradient, balance hyperparameters, finite-sample decoupling). | §3.7 | informal-argument | C-3.7-Class1 | standalone | pending |
| C-3.7-Class3 | Class 3 (stratified/conditional matching) requires `K^d` bins; doesn't scale. | §3.7 | scaling-argument | — | standalone | pending |
| C-3.7-CRPS | Class 4 (scoring rule on standardized residuals) has directional propriety error; for CRPS vs `N(0,1)`, optimal sample distribution is `δ_0` (collapses to median, opposite of calibration). | §3.7 | derivation (explicit integral) | — | load-bearing | pending |
| C-3.7-Class5 | Class 5 = NF-MLE: per-sample conditional log-density, scales pointwise to any d, identical to SNL, only subtlety is R2. | §3.7 | follows from §3.2 | C-3.2-KL, C-3.5-loss-below | standalone | pending |
| C-3.7-positioning | Position of NPE/SNPE, NLE/SNL, NRE, Balanced NRE, Calibrated NPE, LF2I/WALDO/Box CD vs CD-SBI within this taxonomy. | §3.7 end | informal-argument + citations | C-3.7-Class5 | standalone | pending |

## Part III — Theory in one parameter (§4–§5)

### §4 Location-normal uniqueness

| ID | Statement | Location | Evidence | Depends-on | Type | Status |
|---|---|---|---|---|---|---|
| T-A | For `X ~ N(θ,1)` on Θ=ℝ, `M ∩ F_{C¹} = {r*}` with `r*(θ;X) = θ − X`; induced CD `H_{r*}(θ;X) = Φ(θ−X)` (= Schweder–Hjort UMPU CD). | §4.1 | proof (three lemmas, §4.3) | L-4.1, L-4.2, L-4.3 | load-bearing | pending |
| L-4.1 | 1D monotone rearrangement: strictly monotone ψ pushing `N(θ_0,1)` to `N(0,1)` is `X − θ_0` (increasing) or `θ_0 − X` (decreasing). | §4.2 | proof (CDF/inverse-CDF uniqueness) | — | load-bearing | pending |
| L-4.2 | Level-set rigidity (C¹): C¹ ψ pushing `N(θ_0,1)` to `N(0,1)` is strictly monotone. | §4.2 | proof (critical-point density blow-up) | — | load-bearing | pending |
| L-4.3 | NF-MLE strictly proper, conditional: any minimizer `r ∈ F_{C¹}` satisfies `r(θ_0;X) | θ_0 ~ N(0,1)` ρ-a.e. | §4.2 | proof (from C-3.2-KL + R2) | C-3.2-KL, R2 | load-bearing | pending |
| R3 | Strict X-monotonicity a.e.: `|∂_X r(θ;X)| ≥ c > 0` on the differentiability set. | §4.4 | definition | — | load-bearing | pending |
| T-A* | Lipschitz r satisfying (R1), (R2), (R3): if `r ∈ M` then `r(θ;X) = θ − X` a.e. | §4.4 | proof (Rademacher + L-4.1) | L-4.1, L-4.3, R3 | load-bearing | pending |
| P-4.5 | Unique element of `M ∩ F_{C¹}` produced by CD-SBI is the Schweder–Hjort UMP-unbiased CD `Φ(θ−X)`. | §4.5 | identification | T-A | standalone | pending |

### §5 Exponential families

| ID | Statement | Location | Evidence | Depends-on | Type | Status |
|---|---|---|---|---|---|---|
| D-EF | One-param exp. family with continuous sufficient statistic `T(X)`, increasing MLR, `T` has positive density under each `P_θ`. | §5.1 | definition | — | load-bearing | pending |
| C-5.1-pivot | Canonical pivot `r*(θ;X) = Φ^{-1}(1 − F_θ(T(X)))` satisfies (R1) (by MLR) and (R2). | §5.1 | derivation (PIT) | D-EF, R1, R2 | load-bearing | pending |
| T-C | Within `F^T_{C¹}`, `M ∩ F = {r*}`. | §5.2 | proof (T-A in t-coordinates) | T-A, C-5.1-pivot | load-bearing | pending |
| C-5.3-tloc | Location-normal with unknown variance + ancillary `S²`: pivot is Student-t CD `Φ^{-1}(F_{t_{n-1}}(√n(θ−X̄)/S))`. | §5.3 | derivation | T-C | illustrative | pending |
| C-5.3-exp | Exponential rate: `r*(θ;T) = Φ^{-1}(F_{χ²_{2n}}(2θT))` since `2θT ~ χ²_{2n}` and MLR is decreasing. | §5.3 | derivation | T-C | illustrative; reused in E-8.4 | pending |
| D-randPIT | Randomized PIT `V_θ(X,U) = F_θ(T(X)^-) + U·p_θ(T(X))`. | §5.7.1 | definition | — | load-bearing | pending |
| C-5.7-randU | Under `X ~ P_{θ_0}` and `U ~ U(0,1)` indep, `V_{θ_0}(X,U) ~ U(0,1)` exactly. | §5.7.1 | derivation | D-randPIT | load-bearing | pending |
| R4 | Order-preservation in `T(X)`: for fixed `(θ,U)`, `r(θ;X_1,U) ≤ r(θ;X_2,U)` whenever `T(X_1) < T(X_2)` (strict somewhere). | §5.7.2 | definition | — | load-bearing | pending |
| T-C* | Under (R1)–(R4), unique pivot in the randomized calibration manifold is `r*_rand = Φ^{-1}(1 − V_θ)`. | §5.7.2 | proof (lex-order monotone rearrangement) | C-5.7-randU, R4 | load-bearing | pending |
| C-5.7-counterex | Explicit `Bin(1,θ)` counterexample: alternative V′_θ interleaves T-values, satisfies (R1)–(R3_U), yields a calibrated pivot — (R4) is what selects V_θ*. | §5.7.2 | explicit construction | T-C* | load-bearing | pending |
| C-5.7-R4-auto | (R4) is automatic in the continuous case via L-4.2 / R3 + sufficiency. | §5.7.2 | informal-argument | L-4.2, R3 | standalone | pending |
| C-5.7-midp | Mid-p CD with `U=1/2` has total-variation error from exact `U(0,1)` bounded by `½·sup_{θ,t} p_θ(t) = O(1/√n)` for `Bin(n,θ)` at moderate θ. | §5.7.3 | informal-argument + citation | D-randPIT | load-bearing | pending |

## Part IV — Multivariate (§6)

| ID | Statement | Location | Evidence | Depends-on | Type | Status |
|---|---|---|---|---|---|---|
| D-autoflow | Autoregressive triangular flow class with (R1^auto) autoregressive θ-monotonicity and (R2^auto) autoregressive X-monotonicity. | §6.1 | definition | R1, R2 | load-bearing | pending |
| D-additive | Additive parameterization: `r_k = a_k(θ_k; θ_<k, X_<k) − b_k(X_k; θ_<k, X_<k)` with UMNN a_k, b_k. | §6.1 | definition | D-UMNN | standalone | pending |
| D-doublymono | Doubly-monotone parameterization: `r_k = b_k(X_k; X_<k) + ∫ softplus(α_k + β_k) dt`. | §6.1 | definition | D-UMNN | standalone | pending |
| C-6.2-tri-jac | Both Jacobians lower-triangular; det = product of diagonals; `log|det ∂_X r| = Σ_k log|∂_{X_k} r_k|`, all closed-form. | §6.2 | by construction | D-autoflow | load-bearing | pending |
| C-6.2-Olin | Triangular construction scales O(d) per sample (vs O(d³) for dense Jacobian). | §6.2 | by construction | C-6.2-tri-jac | standalone | pending |
| C-6.2-connectedCI | (R1^auto) gives `H_r(·;X)` connected level sets in autoregressive sense; chi-square inversion confidence sets are well-defined. | §6.2 | informal-argument | D-autoflow | load-bearing | pending |
| T-A-d | Within autoregressive triangular pivots with each `r_k ∈ C¹` in X_k, M is the singleton Knothe–Rosenblatt rearrangement `r_k^KR = Φ^{-1}(1 − F_k^{(θ)}(X_k | X_<k))`. | §6.3 | proof (induction on k, using T-A in X_k-coordinates) | T-A, D-autoflow | load-bearing | pending |
| C-6.3-Gaussian | For `X | θ ~ N(θ, Σ)` with Cholesky `LL^T = Σ`: `r^KR(θ,X) = L^{-1}(θ − X)`. | §6.3 corollary | direct computation | T-A-d | load-bearing | pending |
| C-6.3-ordering | KR depends on coordinate ordering; different orderings give different valid pivots in their respective M's. | §6.3 remark 1 | informal-argument | T-A-d | standalone | pending |
| C-6.3-non-unique-arch | Architectural decomposition `r_k = a_k − b_k` is not uniquely determined (shift by `h(θ_<k, X_<k)`); only the difference is identified. | §6.3 remark 2 | informal-argument | D-additive | standalone | pending |
| C-6.3-R2auto-role | Without (R2^auto), the inductive step fails — `X_k ↦ r_k` could be non-bijective and the conditional pushforward is no longer a monotone rearrangement. | §6.3 remark 3 | informal-argument | T-A-d, D-autoflow | load-bearing | pending |
| C-6.4-CI | `C_α(X_obs) = {θ : ‖r(θ;X_obs)‖² ≤ χ²_{d,α}}` has exact coverage `P(θ_0 ∈ C_α) = α` under (C1). | §6.4 | derivation (chi-square inversion) | C-1.3-C1 | load-bearing | pending |
| C-6.4-CIalgo | `C_α` found by 1D root-finding autoregressively (θ_k | θ_<k slices). | §6.4 | informal-argument | C-6.2-connectedCI | standalone | pending |

## Part V — Empirical validation (§7–§8)

### §7 Architecture / training / diagnostics

| ID | Statement | Location | Evidence | Depends-on | Type | Status |
|---|---|---|---|---|---|---|
| D-UMNN | UMNN: scalar `g(z;c) = bias(c) + ∫_{z_0}^z softplus(MLP(t,c)) dt` is strictly monotone in z by construction; derivative is closed-form. | §7.1 | definition (Wehenkel–Louppe 2019) | — | load-bearing | pending |
| C-7.1-init | Final-layer weights init to zero so model starts near-identity; learnable α init at `0.5/ln2 ≈ 0.72` giving init effective slope 0.5 (deliberately wrong). | §7.1 | implementation choice | D-UMNN | standalone | pending |
| C-7.3-noisefloor | Under the null, `√N · KS` has Kolmogorov limit with `P(>1.628)=0.01`; at N=5000, KS noise floor ≈ 0.023; at N=6000, ≈ 0.021. | §7.3 | derivation | — | load-bearing (calibrates "borderline" verdicts) | pending |
| C-7.3-diagnostics | Five diagnostics: pivot RMSE vs `r*`, marginal PIT KS, conditional PIT KS, joint Mahalanobis χ²_d KS, coverage at α∈{0.5,0.68,0.9,0.95}. | §7.3 | informal-argument | — | standalone | pending |

### §8 Experiments

| ID | Statement | Location | Evidence | Depends-on | Type | Status |
|---|---|---|---|---|---|---|
| E-8.1 | 1D location-normal: RMSE 0.030, ∂_θ r=0.997, ∂_X r=−0.983; marginal PIT KS 0.008; conditional PIT bulk KS≈0.01, edge KS up to 0.05; coverage error <0.01 interior. | §8.1 | empirical | T-A, D-additive | load-bearing (theory↔empirics link) | pending |
| C-8.1-edge | Conditional PIT degradation near support boundary of ρ is a known finite-sample effect (undersampling), not structural. | §8.1 | informal-argument | E-8.1 | standalone | pending |
| E-8.2 | 2D loc Σ=I: total RMSE 0.044, per-component 0.014/0.061; marginal KS 0.007/0.006; joint Mahalanobis KS 0.011; conditional Mahalanobis KS 0.008–0.016; coverage <0.01. | §8.2 | empirical | D-autoflow, T-A-d | load-bearing | pending |
| C-8.2-borderline | "Borderline" conditional PIT for r_2 reflects KS within ~2× noise floor at N=5000; joint Mahalanobis and coverage are clean. | §8.2 | informal-argument | E-8.2, C-7.3-noisefloor | standalone | pending |
| E-8.3 | 2D loc Σ correlated: total RMSE 0.27 (~3% relative); marginal KS 0.010/0.003; joint Mahalanobis KS 0.012; coverage <0.015; recovered `E[J_θ]` matches `L^{-1}` to 1–2% over 200 points. | §8.3 | empirical | C-6.3-Gaussian | load-bearing | pending |
| C-8.3-interp | High pivot RMSE 0.27 is dynamic-range-relative (~3%); proper multivariate diagnostic is recovered Jacobian + coverage, not pointwise RMSE; residuals reflect finite-network noise around unique KR target. | §8.3 commentary | informal-argument | E-8.3, T-A-d | load-bearing | pending |
| E-8.4 | Exponential rate doubly-monotone: pivot RMSE 0.045; final loss 0.87 vs truth 0.88; marginal PIT KS 0.006 (p=0.23); conditional PIT KS 0.016–0.028 (borderline vs N=6000 noise floor 0.021); coverage <0.015. | §8.4 | empirical | D-doublymono, C-5.3-exp | load-bearing | pending |
| E-8.4-ablation | Exponential rate autograd-Jacobian ablation: RMSE 1.56, conditional PIT KS 0.045–0.080 (all FAIL), final loss 0.56 — *below* the analytical truth's loss of 0.88. | §8.4 ablation | empirical | E-8.4 | load-bearing | pending |
| C-8.4-ablation-interp | Loss-below-truth in ablation is direct empirical evidence for C-3.5-loss-below: non-bijective `T↦r` gives a non-normalized `p̂_r`, and locally inflated `\|∂r/∂T\|` from folding contributes positively to surrogate log-density. | §8.4 | informal-argument | E-8.4-ablation, C-3.5-loss-below | load-bearing | pending |
| C-8.5-recipe | Distilled prescription across experiments: NF-MLE loss + architecturally monotone-in-θ (R1) + architecturally monotone-in-X (R2). | §8.5 | synthesis | C-3.2-KL, R1, R2 | standalone | pending |

## Part VI — Practice and position (§9–§10)

### §9 Implementation recipe

| ID | Statement | Location | Evidence | Depends-on | Type | Status |
|---|---|---|---|---|---|---|
| C-9-choose-arch | Architecture choice from structure of truth: additive → additive flow; multivariate cross-coupling → triangular additive; multiplicative θ–X → doubly-monotone; unknown → doubly-monotone with autoregressive conditioning. | §9 step 1 table | informal-argument | D-additive, D-doublymono | standalone | pending |
| C-9-arch-not-penalty | Never rely on autograd for valid Jacobian; enforce both monotonicities architecturally. (§8.4 is the empirical reason.) | §9 step 2 | follows from E-8.4-ablation | E-8.4-ablation, C-3.5-loss-below | load-bearing | pending |
| C-9-rho | ρ with full support on inferential region; edge effects (§8.1) appear near support boundary. | §9 step 3 | informal-argument + E-8.1 | C-1.3-rho, C-8.1-edge | standalone | pending |
| C-9-training | Batch size 256–1024; Adam at ~3e-3; 3,000–10,000 steps; no annealing. | §9 step 4 | informal-argument | — | standalone | pending |
| C-9-validate | Run §7.3 diagnostics; coverage is the critical one — if marginal PIT passes but conditional fails, you have a Hermans-style problem. | §9 step 5 | informal-argument | C-7.3-diagnostics, C-1.1-overconf | standalone | pending |
| C-9-invert | At test time, compute `C_α` via 1D root-finding autoregressively. | §9 step 6 | informal-argument | C-6.4-CIalgo | standalone | pending |

### §10 Position in SBI literature

| ID | Statement | Location | Evidence | Depends-on | Type | Status |
|---|---|---|---|---|---|---|
| C-10-tbl | Comparison table of SBI methods (NPE/SNPE, NLE/SNL, NRE/SNRE, Balanced NRE, Calibrated NPE, LF2I, WALDO, Box CD, Variational SBI) vs CD-SBI: target + calibration type per row. | §10 | citations | C-1.1-overconf | standalone | pending |
| C-10-closest | Closest cousins: LF2I and WALDO (test-statistic CDFs + Neyman inversion); SNL (same NF-MLE loss, no monotone-in-θ constraint, no frequentist inference machinery). | §10 | by inspection | C-3.4-SNL | standalone | pending |
| C-10-singlestage | CD-SBI is single-stage (no held-out calibration sample), unlike LF2I/WALDO which require a separate critical-values branch. | §10 | by inspection | C-3.2-strictprop | standalone | pending |
| C-10-IRT | Framework relates to implicit-LR tradition (Cranmer 2015); optimal pivot in loc-normal is the standardized log-likelihood gradient; ties to signed-root log-LR in exp families (SH 2016 ch.5). | §10 | derivation + citation | T-A, T-C | standalone | pending |

## Part VII — Open problems (§11)

For each: is the problem actually open (not resolvable from prior work)?
Is the characterization accurate?

| ID | Statement | Location | Evidence | Depends-on | Type | Status |
|---|---|---|---|---|---|---|
| OP-11.1 | Uniqueness beyond autoregressive triangular class is open — rotation ambiguity in dense flows since `N(0,I_d)` is rotation-invariant. | §11.1 | informal-argument | T-A-d | open-problem-claim | pending |
| OP-11.2 | KR ordering selection for unordered data: different orderings give different valid pivots; finite-sample optimization behavior unclear; OT/Brenier connection (Carlier–Galichon–Santambrogio 2010) unexplored in CD-SBI context. | §11.2 | informal-argument + citation | C-6.3-ordering | open-problem-claim | pending |
| OP-11.3 | Whether (R3) follows from (R1) + Lipschitz + calibration alone is open. | §11.3 | informal-argument | R3, T-A* | open-problem-claim | pending |
| OP-11.4 | Misspecification: NF-MLE population minimizer becomes moment-projection; coverage properties of that object are unstudied. | §11.4 | informal-argument + citations | C-3.2-KL | open-problem-claim | pending |
| OP-11.5 | Higher-d scaling: triangular Jacobian linear in d, but conditioning context grows; d ≲ 10 next milestone. | §11.5 | informal-argument | C-6.2-Olin | open-problem-claim | pending |
| OP-11.6 | Sequential/amortized variants: whether calibration survives sequential refinement is non-trivial. | §11.6 | informal-argument | C-3.2-KL | open-problem-claim | pending |
| OP-11.7 | Discrete with no scalar SS (images, sequences): whether Lancaster randomization extends to learned-feature settings. | §11.7 | informal-argument | T-C*, C-5.7-randU | open-problem-claim | pending |
| OP-11.8 | Efficient `C_α` computation in d ≥ 3 — direct sampling from implicit set, or learned set-boundary parameterization. | §11.8 | informal-argument | C-6.4-CI | open-problem-claim | pending |

## Status of claims (per the manuscript's own table)

The .tex includes its own "Status of claims" table at the end. Round-1
critics should cross-check that table against their findings; a claim
marked "Proven" there but flagged as `error` or `under-justified` here
is a high-priority discrepancy.

## Notes for critic agents

- A claim is **load-bearing** if its falsification breaks a downstream
  theorem or empirical interpretation. Prioritize these.
- For each load-bearing claim, the critic should verify both the
  statement and its proof / derivation chain.
- For citations (e.g., Schweder–Hjort 2016, Wehenkel–Louppe 2019,
  Carlier–Galichon–Santambrogio 2010), round 1 does not re-verify the
  cited papers — that is round 2's job. Flag only if the *attribution*
  appears wrong on its face (claim doesn't match cited paper's
  contribution).
- Update `Status` fields here only after integration in step 1.3.
  Critic-agent reports use their own status fields per claim.
