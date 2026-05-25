# Round 1 — Part III (1D theory) Review

Reviewer profile: mathematical statistician — 1D monotone rearrangement / OT, regular exponential families with MLR, ancillarity and conditional inference, randomized PIT (Lancaster 1961 / Hwang–Yang 2001). Measure-theoretic + Lipschitz/Rademacher arguments.

Scope: §4 (Theorem A and lemmas, §4.1–§4.5) and §5 (exponential families, randomized PIT, mid-p; §5.1–§5.7) of `cd_sbi_v7.tex`, lines 695–1026.

## Summary

- Total claims vetted: 19
- ✓ holds: 11 | ⚠ under-justified: 6 | ✗ error: 2
- Cross-Part findings: 4

The two outright errors are both in §5.7: (i) the explicit Bin(1, θ) counterexample (C-5.7-counterex) does not actually do its advertised job — under the natural monotone-in-U completion of V'_θ(1, U), the resulting V'_θ still satisfies (R4); under any completion that violates (R4), (R3_U) is also broken by a downward jump; (ii) the mid-p bound (C-5.7-midp) is stated as a total-variation bound, but the true total variation between any purely atomic mid-p law and U(0,1) is 1 — the ½·sup p_θ(t) bound is the Kolmogorov (CDF-sup) distance, not TV. Both are repairable but as written are wrong. A handful of smaller items in §4 (the sign-selection step in T-A, the Lipschitz applicability of L-4.3 in T-A*, the implicit non-Gaussian generalization of L-4.2 used in T-C) are under-justified rather than wrong.

## Per-claim findings

### T-A
**Verdict:** ⚠ under-justified
**Quote:** "Within the class F_{C¹} ... M ∩ F_{C¹} = {r*}, r*(θ; X) = θ − X." (line 707–708)
**Diagnosis:** The statement holds, but the sign-selection step in §4.3 is incomplete. Lemma 4.1 gives, *for each θ_0*, ψ_{θ_0}(X) ∈ {X − θ_0, θ_0 − X}. The text writes "The case ψ_{θ_0}(X) = X − θ_0 for ρ-a.e. θ_0 gives r(θ, X) = X − θ, ... contradicting (R1)" — this only addresses two of three possibilities. A measurable mixture (increasing branch on a set A ⊂ Θ with 0 < ρ(A) < 1, decreasing branch on A^c) is the third. It is ruled out by (R1) — for any fixed X, the map θ ↦ r(θ, X) would then be decreasing on A and increasing on A^c, violating global monotonicity — but the proof never says so.
**Suggested fix:** Add one sentence: "By (R1), for ρ⊗P_θ-a.e. X the map θ ↦ r(θ, X) is strictly increasing in θ; this rules out any positive-ρ-measure subset of Θ on which the increasing branch holds and a complementary subset on which the decreasing branch holds. Hence one branch holds ρ-a.e."

### L-4.1
**Verdict:** ✓ holds
**Quote:** "Strictly monotone ψ pushing N(θ_0, 1) to N(0,1) is X − θ_0 (increasing) or θ_0 − X (decreasing)." (line 717–722)
**Diagnosis:** Standard 1D monotone-rearrangement. The proof correctly handles both branches: increasing case yields Φ∘ψ = Φ(·−θ_0), decreasing case uses the identity 1 − Φ(z) = Φ(−z) to convert to Φ(θ_0 − ·). The statement also correctly allows both branches, so any later sign selection must come from (R1).
**Suggested fix:** None.

### L-4.2
**Verdict:** ⚠ under-justified (correct on substance; one phrase is misleading)
**Quote:** "For higher-order critical points the divergence is faster; the conclusion is unchanged." (line 759–760)
**Diagnosis:** The main blow-up argument is correct. At a quadratic local max ψ''(X_*) ≠ 0, |ψ'(X_i(v))| ~ |v − v_*|^{1/2} and the two-term sum diverges like |v − v_*|^{−1/2} — verified. The "WLOG local maximum" is harmless; local minima give a blow-up *above* v_* by symmetric argument. For higher-order critical points where ψ(X) − v_* ~ ±c(X − X_*)^{2m} with m ≥ 2, |ψ'| ~ |v−v_*|^{(2m−1)/(2m)} and the divergence exponent is (2m−1)/(2m) > 1/2 — yes, faster.

There is one real concern with phrasing, however: the lemma's *hypothesis* is that ψ' changes sign, not merely that ψ has a critical point. Cases like ψ(X) = X^3 (a critical point at 0 with ψ' ≥ 0, no sign change) are excluded from the contradiction argument *and* are consistent with the conclusion (X^3 is strictly monotone). The current text says "by continuity of ψ', there is some X_* with ψ'(X_*) = 0 and ... values X_1 < X_* < X_2 with ψ(X_1) = ψ(X_2)" — this two-preimage configuration requires that ψ have a *local extremum* at X_*, not just a zero of ψ'. For ψ' that changes sign by continuity it does have a local extremum, so the argument is fine, but the reader should be told explicitly.
**Suggested fix:** Insert: "Since ψ' changes sign and ψ' is continuous, there exists X_* where ψ' attains zero *and* is a sign-change point, so ψ has a strict local extremum at X_*; the two-preimage configuration follows." Also: trim "for higher-order critical points the divergence is faster" — what is meant is "for any X_* where ψ has a local extremum, the leading behaviour ψ(X) − v_* ~ −c(X−X_*)^{2m} for some m ≥ 1 makes the divergence at least as fast as |v−v_*|^{−1/2}".

### L-4.3
**Verdict:** ✓ holds
**Quote:** "Any minimizer r ∈ F_{C¹} satisfies r(θ_0; X) | θ_0 ~ N(0,1) for ρ-a.e. θ_0." (line 763–766)
**Diagnosis:** Follows from C-3.2-KL plus (R2). KL = 0 ⇒ p̂_r = p ρ-a.e., and under (R2) the implied density is the pushforward of N(0,1) under r, so calibration holds ρ-a.e. No propriety/full-support issue here — ρ-a.e. is exactly the right quantifier. The downstream application in T-A *does* implicitly need ρ to have full support (otherwise the per-θ_0 monotone-rearrangement argument only fixes r on supp(ρ)×R), but this is the standing assumption from §2.1, so no error.

For the singular-ρ case raised in the scrutiny list: KL non-negativity still works, but the conclusion is only at a.e. θ_0 ∈ supp(ρ); if supp(ρ) is a measure-zero set (a point mass, say), then T-A's per-θ_0 argument can only pin r down on that set. So T-A's "M ∩ F_{C¹} = {r*}" should be read as "the function r is uniquely determined on supp(ρ) × R, then extended to R × R by R1-monotone interpolation". This is fine given the standing full-support assumption, but the conditional-propriety argument itself is robust: it does *not* need full ρ-support.
**Suggested fix:** None for L-4.3 itself. Optional: in T-A, add a one-line acknowledgement that ρ-full-support (§2.1) is what lifts the per-θ_0 result to a global identity.

### R3
**Verdict:** ✓ holds (definition only)
**Quote:** "(R3) [strict X-monotonicity, a.e.]: ∂_X r has constant sign and is bounded away from zero on the differentiability set: |∂_X r(θ; X)| ≥ c > 0 a.e." (line 813–816)
**Diagnosis:** Clean definition. Compatible with Lipschitz r (where ∂_X r exists Lebesgue-a.e. by Rademacher). The "constant sign" piece is needed for L-4.1; the "bounded below by c" piece is needed to ensure global strict monotonicity (rules out flat plateaus on null sets that would still defeat invertibility under a.e. derivative ≥ 0).
**Suggested fix:** None.

### T-A*
**Verdict:** ⚠ under-justified
**Quote:** "By (R3), ψ_{θ_0}(X) := r(θ_0; X) is strictly monotone in X. By Lemma 4.3, ψ_{θ_0} pushes N(θ_0, 1) to N(0,1)." (line 825–827)
**Diagnosis:** The Rademacher → strict-monotone step is correct (Lipschitz + ∂_X r ≥ c a.e. ⇒ r(θ_0, X_2) − r(θ_0, X_1) ≥ c(X_2 − X_1) for X_1 < X_2, by the fundamental theorem of calculus for absolutely continuous functions; and "constant sign" gives global strict increase/decrease without loss). L-4.1 then applies. Good.

The under-justification is in invoking Lemma 4.3 for a Lipschitz (not C¹) r. L-4.3 is stated and proved over F_{C¹}. Its proof rests only on C-3.2-KL plus (R2) "C¹ or Lipschitz a.e. diffeomorphism" (§2.2 explicitly allows both). So L-4.3 trivially extends to Lipschitz r — but the statement of L-4.3 as written (§4.2) cites F_{C¹}, and T-A* applies it outside that class.
**Suggested fix:** Either (a) restate L-4.3 with class "F^{Lip} ∪ F_{C¹}" or simply "any r satisfying (R2)" — the proof goes through unchanged; or (b) in T-A*, write "L-4.3, whose proof depends only on (R2) and C-3.2-KL, applies equally under the Lipschitz form of (R2)."

### P-4.5
**Verdict:** ✓ holds
**Quote:** "In the location-normal model, the unique element of M ∩ F_{C¹} produced by CD-SBI is identically the Schweder–Hjort UMP-unbiased CD." (line 853–855)
**Diagnosis:** Direct identification given T-A: r*(θ;X) = θ − X ⇒ H_{r*}(θ;X) = Φ(θ−X), which Schweder–Hjort (2016, ch. 5) give as the UMPU CD for the location-normal model. The attribution is correct on its face (round 2 will verify the citation). Beyond restating T-A, there is no separate content.
**Suggested fix:** None.

### D-EF
**Verdict:** ✓ holds (definition)
**Quote:** "{P_θ : θ ∈ R} regular one-parameter exponential family with sufficient statistic T : X → R. Assume T(X) has a continuous positive density under each P_θ, and assume increasing MLR." (line 877–881)
**Diagnosis:** Standard setup. "Increasing MLR" is automatic in the natural-parameter canonical form with real-valued T; calling it out is helpful. The "continuous positive density" assumption excludes the discrete case (handled separately in §5.7) and discontinuous-support continuous cases (e.g., Uniform(0, θ) — see cross-Part note).
**Suggested fix:** None for round 1. Round 3 (exposition) may want to flag that "regular" implicitly precludes non-canonical-support models like the uniform family.

### C-5.1-pivot
**Verdict:** ✓ holds
**Quote:** "r*(θ; X) := Φ^{-1}(1 − F_θ(T(X)))." (line 884–885)
**Diagnosis:** Calibration follows from PIT (F_{θ_0}(T(X)) ~ U(0,1) under continuity); 1 − U ~ U, so Φ^{-1}(1−F_{θ_0}(T(X))) ~ N(0,1). (R1): under increasing MLR, T is stochastically increasing in θ, so F_θ(t) is decreasing in θ at fixed t, so 1 − F_θ(t) is increasing in θ; composing with the increasing Φ^{-1} preserves monotonicity. (R2): r* is a strictly monotone function of T(X) under the positive-density assumption. In d=1, where the pivot is parameterized through T directly (the F^T_{C¹} class), the (R2)-as-X-diffeomorphism question is replaced by "diffeomorphism in t = T(X)", which holds.
**Suggested fix:** None.

### T-C
**Verdict:** ⚠ under-justified
**Quote:** "Proof sketch. Apply Theorem A in t = T(X) coordinates. ... The same three-lemma argument applies, with the source density playing the role of the Gaussian. Direction is fixed by MLR." (line 903–907)
**Diagnosis:** The substance of the proof transfers, but Lemma 4.2's proof was *not* written in source-density-agnostic form. The blow-up argument needs only two ingredients from the source: (a) the source density g_{T|θ_0} is finite-valued (so the pushforward at v_* would need to be finite) and (b) the target Gaussian density at v_* is finite. The blow-up at a critical X_* gives g_ψ(v) → ∞ provided g_{T|θ_0}(X_*) > 0 — which is the positive-density assumption in D-EF. So L-4.2 extends, but the proof of L-4.2 as written only invokes φ(X_* − θ_0) > 0. The author should either (i) restate L-4.2 as "source: continuous positive density on R; target: N(0,1)" or (ii) prove the EF lift as a self-contained argument rather than "apply Theorem A". The sign selection by MLR is fine — MLR gives directional monotonicity of F_θ in θ.
**Suggested fix:** Re-state Lemma 4.2 in the more general source form (the proof requires no change beyond replacing φ(·−θ_0) with the source density f_{T|θ_0}). Then T-C is a clean corollary.

### C-5.3-tloc
**Verdict:** ✓ holds
**Quote:** "r*(θ; X̄, S) = Φ^{-1}(F_{t_{n-1}}(√n(θ − X̄)/S))." (line 919–921)
**Diagnosis:** Conditional on the ancillary S², the conditional distribution of X̄ given S is N(θ, S²/n) (in the appropriate conditioning sense for the Cox 1958 / Fisher ancillary-conditioning argument). The pivot √n(X̄ − θ)/S ~ t_{n-1}; by symmetry of the t-distribution, √n(θ − X̄)/S ~ t_{n-1} also. So F_{t_{n-1}}(√n(θ − X̄)/S) ~ U(0,1) under X ~ P_θ, and r* ~ N(0,1). Monotone in θ: ∂_θ[√n(θ − X̄)/S] = √n/S > 0, so r* is strictly increasing in θ. (R1) ✓.
**Suggested fix:** None.

### C-5.3-exp
**Verdict:** ✓ holds
**Quote:** "r*(θ; T) := Φ^{-1}(F_{χ²_{2n}}(2θT))." (line 923–928)
**Diagnosis:** T = Σ X_i ~ Gamma(n, θ) (rate parameterization), 2θT ~ χ²_{2n} (standard scale conversion). Under P_θ_0, F_{χ²_{2n}}(2θ_0 T) ~ U(0,1), so r*(θ_0; T) ~ N(0,1). For monotonicity: ∂_θ F_{χ²_{2n}}(2θT) = 2T · f_{χ²_{2n}}(2θT) > 0, so r* increasing in θ. The decreasing-MLR convention ("drop the 1−") then gives the right sign. Consistent with E-8.4.
**Suggested fix:** None.

### D-randPIT
**Verdict:** ✓ holds (definition)
**Quote:** "V_θ(X, U) := F_θ(T(X)^-) + U · p_θ(T(X))." (line 949–951)
**Diagnosis:** Lancaster (1961) randomized PIT, exactly as standard. Generalizes to any discrete or mixed distribution via T-spread-by-U on each atom.
**Suggested fix:** None.

### C-5.7-randU
**Verdict:** ✓ holds
**Quote:** "Under X ~ P_{θ_0}, U ~ U(0,1) independent: V_{θ_0}(X, U) ~ U(0,1) exactly." (line 952–953)
**Diagnosis:** Conditional on T = t, V_{θ_0} = F_{θ_0}(t^-) + U · p_{θ_0}(t) is uniform on [F_{θ_0}(t^-), F_{θ_0}(t)] — an interval of length p_{θ_0}(t). Marginalizing over T (whose mass at t is exactly p_{θ_0}(t)), the resulting density on each tile is p_{θ_0}(t) · (1/p_{θ_0}(t)) = 1; tiles partition [0,1]. So V_{θ_0} ~ U(0,1) exactly. Edge cases: atoms with p_θ(t) = 0 contribute a zero-length tile (no contribution). Atoms outside supp(P_{θ_0}) but inside the union of supports for varying θ also have p_{θ_0}(·) = 0, consistent.
**Suggested fix:** None.

### R4
**Verdict:** ✓ holds (definition)
**Quote:** "(R4) Order-preservation in T(X): for fixed (θ, U), r(θ; X_1, U) ≤ r(θ; X_2, U) whenever T(X_1) < T(X_2), with strict inequality somewhere." (line 968–974)
**Diagnosis:** Clean definition. "Strict inequality somewhere" prevents the degenerate case r constant in T, which would violate (R2)'s invertibility. It is, however, *weaker* than the lex-order rearrangement uniqueness might appear to need — see T-C* note.
**Suggested fix:** None for the definition itself, but the wording allows V_θ(0, ·) and V_θ(1, ·) to share boundary values at the partition seam — see T-C* and C-5.7-counterex for the consequence.

### T-C*
**Verdict:** ⚠ under-justified (proof, not statement)
**Quote:** "The proof is a 'lex-order monotone rearrangement': the source measure P_θ ⊗ U(0,1) on supp(T) × [0,1] has, in lexicographic order (T, U), a unique monotone measure-preserving map to U(0,1), namely V*_θ. (R4) is what fixes the lex order across T-values; without it, 'interleaving' counterexamples exist." (line 979–983)
**Diagnosis:** The statement is morally right, but the proof is one sentence and leaves open what "the unique monotone measure-preserving map to U(0,1)" means in the discrete-continuous mixture setting. (R3_U) gives monotone in U for each fixed (θ, X); (R4) gives monotone in T for each fixed (θ, U). Together they give joint monotonicity in (T, U) under the lex order on the product. Given the unique monotone-rearrangement map between any two atomless probability measures on [0,1], V*_θ is then identified as the unique such map. Two missing pieces: (i) the (T, U) → V map is required to factor through this lex order — which requires (R1)+(R2)+(R3_U)+(R4) — but the proof never spells out the dependency on each condition; (ii) "monotone measure-preserving" between a discrete-continuous source and a continuous target is a special case of the increasing-rearrangement theorem and warrants either a reference or a one-line construction.

Does T-C* really require all of (R1)–(R4)? Yes: (R1) fixes the θ-direction at the Φ^{-1} step; (R2) makes (T, U) → V_θ a bijection onto [0,1]; (R3_U) gives monotonicity within each T-atom; (R4) gives order across T-atoms. Drop any one and either calibration breaks (R2) or a different lex permutation becomes valid (R4) — though (R3_U) alone is somewhat redundant given (R2) plus (R4), since (R2)-as-diffeomorphism would force monotonicity within each atom.
**Suggested fix:** Replace the one-sentence proof with: "Conditional on T = t, U → V_θ(X, U) is a monotone bijection from [0, 1] to a sub-interval of [0,1] of length p_θ(t) (by (R3_U) + (R2)); the sub-interval is positioned by (R4) immediately above the corresponding sub-interval for any T(X') < t. By induction (or by direct lex-order argument), the only such positioning consistent with the cumulative-density-1 constraint is the V*_θ tiling. Then r = Φ^{-1}(1 − V_θ) is uniquely r*_rand. (R1) fixes the sign of Φ^{-1}'s composition with V."

### C-5.7-counterex
**Verdict:** ✗ error (as written; repairable with a different completion)
**Quote:** "V'_θ(0, U) = { (1−θ)/2 · 2U if U ∈ [0, 1/2), (1/2) + (1−θ)/2 · 2(U − 1/2) if U ∈ [1/2, 1] }, with V'_θ(1, U) defined so that the joint mixture remains uniform on (0,1) ... . Both V*_θ and V'_θ satisfy (R1)–(R3_U) and both yield calibrated pivots." (line 996–1002)
**Diagnosis:** I worked through this in detail. V'_θ(0, ·) sends X = 0 into the two sub-intervals A_0 = [0, (1−θ)/2] ∪ [1/2, 1/2 + (1−θ)/2]. The natural monotone-in-U completion of V'_θ(1, ·) fills the complementary intervals A_1 = [(1−θ)/2, 1/2] ∪ [1/2 + (1−θ)/2, 1] in increasing-U order: V'_θ(1, U) = (1−θ)/2 + θU for U ∈ [0, 1/2), and 1/2 + (1−θ)/2 + θ(U − 1/2) for U ∈ [1/2, 1]. Under this completion:
- Each of V'_θ(0, ·) and V'_θ(1, ·) is strictly monotone in U (with an upward jump at U = 1/2), so (R3_U) holds.
- (R1) holds (∂_θ checks out for all four pieces).
- The joint mixture is uniform on [0,1] by construction.
- **(R4) holds, not fails.** At U ∈ [0, 1/2): V'(0, U) ∈ [0, (1−θ)/2] and V'(1, U) ∈ [(1−θ)/2, 1/2], so V'(0, U) ≤ (1−θ)/2 ≤ V'(1, U). At U ∈ [1/2, 1]: V'(0, U) ∈ [1/2, 1/2 + (1−θ)/2] and V'(1, U) ∈ [1/2 + (1−θ)/2, 1], so V'(0, U) ≤ V'(1, U). So this V'_θ is *not* a counterexample to T-C*.

Conversely, any completion of V'_θ(1, ·) that *does* violate (R4) — e.g., assigning [1/2 + (1−θ)/2, 1] to U ∈ [0, 1/2) and [(1−θ)/2, 1/2] to U ∈ [1/2, 1] — has V'_θ(1, ·) jump *downward* at U = 1/2, violating (R3_U). So the example in the manuscript does not separate (R4) from (R3_U) under the binary atom: any V' satisfying both (R3_U) and the desired interleaving is ruled out.

To get a genuine counterexample where (R3_U) holds but (R4) fails, you need either (i) a model with more atoms (e.g., Bin(2, θ) with supports {0, 1, 2}, so there's room for a non-trivial permutation of T-atoms across U) or (ii) a re-design of the binary example using bypasses around the seam (which then need an extra explanation of why monotone-in-U is preserved).
**Suggested fix:** Replace the Bin(1, θ) construction with a Bin(2, θ) construction: list the three atoms (0, 1, 2) with probabilities ((1−θ)², 2θ(1−θ), θ²), and define V*_θ as the lex-stacking and V'_θ as the permutation that swaps the assignments of T = 1 and T = 2 sub-intervals (so V'_θ assigns T = 2's mass to the middle interval and T = 1's mass to the top interval). Verify (R3_U) and the calibrated mixture, and check that for fixed U, V'(1, U) > V'(2, U) — violating (R4). This gives a clean separation.

### C-5.7-R4-auto
**Verdict:** ⚠ under-justified
**Quote:** "Lemma 4.2 (or (R3)) implies ψ_{θ_0} is monotone in X, and by sufficiency monotone in T(X). Only the discrete case has the extra freedom that requires (R4) as a separate condition." (line 1007–1010)
**Diagnosis:** The intuition is right but the argument has gaps. In the continuous exponential-family setting, by sufficiency the pivot can be written as r(θ; X) = r̃(θ; T(X)). Whether r̃ is monotone in T does *not* follow directly from r being monotone in X — it follows because the calibration constraint forces r to be a monotone rearrangement of T(X)|θ_0's distribution onto N(0,1). The "by sufficiency monotone in T(X)" line skips the step "and the calibration constraint forces the rearrangement to be increasing in T". Also, "Lemma 4.2 (or (R3)) implies ψ_{θ_0} is monotone in X" is the C¹ or Lipschitz version of (R2) plus the rigidity arguments — it is not a statement *only* about R3. The R3 part is fine; the L-4.2 part is what was lifted to general source density in T-C and inherits whatever incompleteness was there.
**Suggested fix:** Tighten to: "In the continuous case, T-C's lift of T-A in t-coordinates already concludes that r̃(θ; t) is monotone-increasing in t (under increasing MLR). This is exactly (R4) at the t-level; (R4) becomes a non-redundant assumption only when t is discrete, since monotone rearrangement of an atomic source onto a continuous target leaves a permutation freedom across the atoms."

### C-5.7-midp
**Verdict:** ✗ error (the bound is correct, but it bounds Kolmogorov distance, not TV)
**Quote:** "This is approximately calibrated, with total-variation error from exact U(0,1) bounded by (1/2) sup_{θ,t} p_θ(t) = O(1/√n) for Bin(n, θ) at moderate θ." (line 1016–1019)
**Diagnosis:** The mid-p random variable V^mid_θ(X) = F_θ(T(X)) − (1/2) p_θ(T(X)) takes values in a finite (countable) set. Its law is purely atomic, while U(0,1) is continuous; the total variation between any purely atomic and any purely continuous probability measure on [0,1] is exactly 1. So the stated TV ≤ (1/2) sup p_θ(t) bound is *false as a TV bound* (½ sup p_θ(t) is generically much less than 1).

What the bound actually controls is the Kolmogorov (sup-CDF) distance K = sup_v |P_θ(V^mid ≤ v) − v|. At v = F_θ(t_*) − (1/2) p_θ(t_*) (an atom location), P_θ(V^mid ≤ v) = F_θ(t_*) while v = F_θ(t_*) − (1/2) p_θ(t_*), giving discrepancy (1/2) p_θ(t_*). Taking sup over t_*: K ≤ (1/2) sup_t p_θ(t). For Bin(n, θ) at moderate θ ∈ (0, 1), the mode probability is Θ(1/√n) by the local CLT, so K = O(1/√n) — this rate *is* correct, just under the right metric.

This is a notational error rather than a content error; the rate and the bound formula are correct under the Kolmogorov metric, and Hwang & Yang (2001) state the mid-p result in essentially this form.
**Suggested fix:** Replace "total-variation error" with "Kolmogorov (CDF supremum) error" or "uniform CDF distance from U(0,1)". Optionally also note Wasserstein-1 distance is bounded similarly by sup_t p_θ(t) up to a small constant.

## Cross-Part findings

- **[Part I, §2.1 standing assumption]:** The full-support assumption on ρ is used implicitly in T-A (and downstream T-C, T-A-d) to lift per-θ_0 results from supp(ρ) to all of Θ. Worth a one-line reminder in §4.3 or §2.1 highlighting this dependence (also flagged in [Part II, C-1.3-rho]).

- **[Part II, §3.2, C-3.2-KL]:** L-4.3 is a direct corollary of C-3.2-KL plus (R2). Any weakening of (R2) (e.g., the Lipschitz form invoked in T-A*) needs to be carried through both: (a) the validity of `p̂_r` as a density (the change-of-variables / area-formula step), and (b) L-4.3's "iff equal densities" implication. The Lipschitz extension is fine but should be made explicit.

- **[Part IV, §6.3, T-A-d induction]:** The induction repeatedly invokes Theorem A "in X_k-coordinates with conditional law F_k^{(θ)}(·|X_<k) playing the role of the Gaussian source". This is the same general-source extension noted in T-C — Lemma 4.2 needs to be either restated in source-agnostic form or re-derived for each new setting. Currently both T-C and T-A-d invoke the source-agnostic form without proving it. A single restated L-4.2 (general continuous-positive-density source) would close this gap across §5.2, §5.3, §5.7 (R4-auto note), and §6.3.

- **[Part V, §7.3 / §8.4, mid-p empirics]:** If the mid-p variant is used in any experiment, the metric mismatch in C-5.7-midp may have led to overstated calibration claims in the experimental write-up. Worth double-checking §8.4 and §7.3 for any explicit mid-p TV statement that should instead be a Kolmogorov-distance statement. (Part V is not in my scope but flagging.)
