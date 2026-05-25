# Round 1 — Part I (Framework) Review

## Summary
- Total claims vetted: 12
- ✓ holds: 4  |  ⚠ under-justified: 7  |  ✗ error: 1
- Cross-Part findings: 4

## Per-claim findings

### C-1.1-overconf
**Verdict:** ✓ holds
**Quote (lines 174-178):** "Hermans et al.~(2022) ... documented that all four major SBI families systematically produce *overconfident* posteriors at realistic budgets --- credible intervals have empirical coverage well below nominal."
**Diagnosis:** Standalone positioning claim, evidentially backed by a single load-bearing citation. The attribution matches the cited paper's headline finding on its face. The hedge "at realistic budgets" is appropriate and not overclaimed. Round-1 scope does not re-verify the citation.
**Suggested fix:** none (re-check the citation in round 2).

### D-CD
**Verdict:** ⚠ under-justified
**Quote (lines 198-200):** "The defining property is **exact coverage**: H(theta_0;X) ~ U(0,1) when X ~ P_{theta_0}, for every theta_0 in Theta."
**Diagnosis:** As stated, this is *one* common operationalization of a CD, but it is not literally the Schweder--Hjort definition — SH (2016, Def. 3.1) require H(theta;x) to be a CDF in theta for each x, plus the U(0,1) marginal condition. The manuscript here drops the "CDF-in-theta" half from the definition and only reintroduces it later via (R1) / §2.3. Strictly the equation given is the calibration property of an *implied* CD given a pivot, not the full definition of a CD. The text also conflates "H(theta_0;X) ~ U(0,1)" with the PIT in the next sentence; this is fine when the pivot is continuous but is not literally the PIT statement without monotonicity-in-theta machinery.
**Suggested fix:** Either (a) state explicitly that the U(0,1) condition is the *calibration* property of a CD and that being a proper CDF in theta is the other half, deferred to §2.3; or (b) add a half-sentence "(together with the requirement that H(.;x) be a CDF on Theta for each x; this is enforced by (R1) in §2.2)". Either preserves the rest of the exposition.

### C-1.2-LocNorm
**Verdict:** ⚠ under-justified
**Quote (lines 226-230):** "In regular one-parameter models the optimal (uniformly most powerful unbiased, ``UMPU'') CD is explicit and unique. For X ~ N(theta,1) it is H*(theta;X) = Phi(theta-X)."
**Diagnosis:** The location-normal computation is correct as a worked check. But the umbrella sentence "in regular one-parameter models the optimal (UMPU) CD is explicit and unique" is stronger than the standard result. Existence and uniqueness of the UMPU CD in Schweder--Hjort and earlier (Fisher, Cox, Pitman) require more than "regular": one typically needs (i) a one-parameter exponential family (or at least MLR in a real sufficient statistic), (ii) a continuous sufficient statistic, and (iii) the test family being two-sided + unbiasedness restriction having a unique solution. "Regular" by itself (e.g. Cramér regularity for MLE) does not deliver UMPU-ness of the resulting CD; that is the content of §5 (one-parameter exp. family + MLR). As written, §1.2 promises something §5 only later earns.
**Suggested fix:** Replace "regular one-parameter models" with "regular one-parameter exponential families with monotone likelihood ratio in a continuous sufficient statistic" — or simply "in the location-normal model" since that is the only worked example in §1.2 anyway. The full hypothesis chain is in §5.1 (D-EF); §1.2 should not claim more than §5 delivers.

### C-1.3-C1
**Verdict:** ✓ holds
**Quote (lines 261-265):** "(C1) Calibration. Under X ~ P_{theta_0}, r(theta_0;X) ~ N(0, I_d) for every theta_0 in Theta. Equivalently, H_r(theta_0;X) ~ U(0,1)^d."
**Diagnosis:** Clean operational definition; the "equivalently" follows because Phi_d is a homeomorphism and N(0, I_d) has independent coordinates whose marginal-Phi pushforward is U(0,1). This is restated in §2 and matches D-M.

### C-1.3-C2
**Verdict:** ⚠ under-justified
**Quote (lines 267-272):** "(C2) Monotonicity in theta. For each fixed X and each coordinate k, theta_k -> r_k(theta;X) is strictly increasing (with autoregressive conditioning in d > 1; see §6). This ensures connected level sets, so confidence regions are well-defined geometric objects."
**Diagnosis:** Two issues. (i) "Coordinate-wise strictly increasing in theta_k holding others fixed" does *not* in general guarantee that H_r(.;X) is a proper CDF on R^d or that the level sets are connected — coordinate-wise monotonicity gives a quasi-monotone but not necessarily order-coherent map. The parenthetical "autoregressive conditioning" is the real fix but is deferred to §6 without a forward-pointer to what the autoregressive form actually does (it makes the joint pushforward Phi_d(r(.;X)) a Knothe--Rosenblatt-type transport, which is what gives connectedness). (ii) "Confidence regions are well-defined geometric objects" is qualitative — what is actually used downstream (e.g. §6.4) is that {theta : ||r||^2 <= chi^2_{d,alpha}} is connected, which needs the autoregressive monotonicity, not just coordinate-wise monotonicity. The 1D version is fine.
**Suggested fix:** Either (a) state (C2) for d=1 in §1.3 and forward-reference §6 for the d>1 autoregressive version, or (b) state (C2) once in the autoregressive form (r_k strictly increasing in theta_k at fixed theta_{<k}, X) so the multi-D reader knows what is being assumed. The current "in each coordinate" wording is ambiguous between simultaneous coordinate-wise and autoregressive.

### C-1.3-rho
**Verdict:** ⚠ under-justified
**Quote (lines 305-309):** "(C1) is a *frequentist* property --- it makes no reference to a prior. The training proposal rho on Theta used to draw simulations is purely a sampling distribution; in the population limit the CD pivot does not depend on rho, provided rho has full support on the region of inferential interest."
**Diagnosis:** This is *almost* right but is overstated for the typical use case. In the population limit, if (a) the architectural class F is rich enough to contain the calibration manifold elementwise, and (b) NF-MLE convergence happens *at every* theta_0 in supp(rho), then the minimizer does not depend on rho qua weights — fine. But in the *non-saturated* class case (F too small to contain M, which is the realistic SBI setting in §11.4), the population minimizer of L(r) = E_rho[KL(p || p_hat_r)] is the rho-weighted KL projection onto F, and this projection does depend on rho. The asserted "does not depend on rho provided full support" is only a population-saturated statement. The §11.4 open problem ("Misspecification: NF-MLE population minimizer becomes moment-projection") acknowledges this in the misspecified case but not in the under-parameterized one.
**Suggested fix:** Add a one-sentence qualifier: "in the population limit *and assuming F contains the calibration manifold*". Or restate as "in the population limit on a saturated class". This also affects C-9-rho.

### D-pivot
**Verdict:** ⚠ under-justified
**Quote (lines 353-354):** "A pivot in the sense used here is a measurable function r: Theta x X -> R^d satisfying: (R1)... (R2)..."
**Diagnosis:** The base definition "measurable r" is fine. The issue is that this is a *non-standard* use of "pivot": classically a pivot is a function whose distribution does not depend on theta when X ~ P_theta; here it is bundled with (R1) and (R2) but the calibration *property* (C1 = N(0, I_d) distribution at the truth) is *not* part of the definition of "pivot" in §2.2 — it is part of the definition of "calibration manifold". So an arbitrary "pivot" in §2.2 need not be calibrated. The text then says "in (location-normal): r* is a pivot in this sense", but r*'s being a pivot at all (in the classical sense) rests on its distribution being theta-free; the §2.2 definition doesn't actually require that. This is at most a terminological gap, but it is load-bearing because the rest of the paper says "the calibration manifold = the set of calibrated pivots", and "pivot" carries different content in different sentences.
**Suggested fix:** Split terminology — call the §2.2 object a "candidate pivot" or simply "an (R1)-(R2)-admissible map", and reserve "pivot" for elements of the calibration manifold (i.e. (R1)-(R2)-admissible *and* calibrated). Alternatively, add the (C1)-equivalent to (R1) and (R2) and call the conjunction "pivot"; then M is the set of all pivots in F.

### R1
**Verdict:** ⚠ under-justified
**Quote (lines 359-364):** "(R1) Monotonicity in theta. For rho ⊗ P_theta-a.e. X and each coordinate k, the map theta_k -> r_k(theta;X) is strictly increasing when other components are held fixed (autoregressively in d > 1; see §6). In 1D this is just: theta -> r(theta;X) is strictly increasing."
**Diagnosis:** The parenthetical "autoregressively in d>1" is doing double duty: the *base* statement says "each coordinate strictly increasing with others held fixed", which is ordinary coordinate-wise monotonicity, but the parenthetical seems to swap that out for autoregressive monotonicity (each theta_k -> r_k strictly increasing at fixed theta_{<k}, ignoring theta_{>k} entirely). These are different conditions and the proofs in §6 use the latter, not the former. Also: "for rho ⊗ P_theta-a.e. X" is an a.e. statement on a joint measure, but monotonicity is a statement about a *function*, not a value. It would be cleaner to say "for rho-a.e. theta and P_theta-a.e. X" or, since this is an architectural constraint, just "for every X". And: nothing in (R1) specifies whether r is increasing or *decreasing* — §2.2's example has partial_theta r* = +1, but Lemma 4.1 explicitly produces *both* signs as compatible with calibration; (R1) is what selects "increasing". A footnote making this explicit would help.
**Suggested fix:** State (R1) directly in the form used in §6 (the autoregressive form), and apply a sign convention explicitly: "strictly *increasing* (as opposed to decreasing) is a convention; the decreasing version corresponds to inverting the direction of H_r." Resolve "a.e. X" wording.

### R2
**Verdict:** ✗ error (as stated; logically equivalent fix is small)
**Quote (lines 366-370):** "(R2) Invertibility in X. For each fixed theta, the map X -> r(theta;X) is a C^1 (or Lipschitz, a.e. differentiable) diffeomorphism onto its image. Equivalently, partial r / partial X has full rank a.e."
**Diagnosis:** Three problems.
(i) The conjunction "C^1 *or* Lipschitz, a.e. differentiable diffeomorphism" treats the two as interchangeable, but they are not equivalent regularity classes — in fact §4.4 (Theorem A*) explicitly adds (R3) precisely because the Lemma 4.2 argument that converts (R2) into strict monotonicity *fails* under Lipschitz regularity. So as far as the downstream proofs are concerned, "C^1 diffeomorphism" and "Lipschitz a.e.-differentiable diffeomorphism" are *not* interchangeable; the Lipschitz route needs the extra (R3). Stating them as an "or" in §2.2 is misleading — the proofs in §4 explicitly distinguish them.
(ii) A "diffeomorphism onto its image" is the wrong object class for Lipschitz maps: a Lipschitz map is not in general invertible-to-Lipschitz on its image, even if a.e. differentiable with full-rank Jacobian (you need quantitative bounds on the Jacobian, i.e. (R3)). The standard fact is: a Lipschitz a.e.-differentiable map with `|det Jac| >= c > 0` a.e. is a bi-Lipschitz homeomorphism onto its image (essentially the inverse-function theorem for Lipschitz maps), but plain full-rank a.e. is *not* enough.
(iii) "Full rank a.e." is *not* equivalent to "diffeomorphism onto its image": e.g. a sinusoidal map on R can have nonzero derivative a.e. but fail to be injective. The equivalence stated in the last sentence is therefore false; injectivity is an independent property.
**Suggested fix:** Either (a) state (R2) only in the C^1 form ("X -> r(theta;X) is a C^1 diffeomorphism onto its image; equivalently, C^1 with partial r / partial X full rank everywhere *and* injective") and defer the Lipschitz extension to §4.4 where (R3) is added; or (b) reformulate the disjunction as: "(R2) [C^1 version] ... and (R2-Lip) [used in §4.4] ... bi-Lipschitz onto its image with `|det Jac| >= c > 0` a.e. This is the form used for ReLU/UMNN architectures in the Lipschitz proof." Delete the "Equivalently, full rank a.e." sentence — it is false.
**Downstream impact:** §4.4 already addresses (iii) by introducing (R3); the issue is that §2.2 should not pretend (R2) alone covers the Lipschitz case. This is presentation, but the false-equivalence sentence is a factual error.

### D-M
**Verdict:** ✓ holds
**Quote (lines 374-378):** "M_F := { r in F : Law(r(theta_0;X) | theta_0) = N(0, I_d) for every theta_0 in supp(rho) }. Any element of M_F satisfying (R1) is a valid CD pivot."
**Diagnosis:** Sound definition. Note: M_F is defined *only* with respect to supp(rho), not Theta — consistent with the population-limit framing in §1.3, and with §5 / §11.4 having to be careful about extrapolation outside supp(rho). The second sentence "any element of M_F satisfying (R1) is a valid CD pivot" is consistent if M_F is conceived as relaxing pivots from "(R1) + (R2) + calibrated" to "F-membership + calibrated"; that matches how §3.2 uses M_F as the argmin of NF-MLE. (Tangentially: D-M presumes F already builds in (R2) — see §3.2 line 472 — but (R1) is not built into F per the definition, so it has to be reimposed in the second sentence; this is mildly confusing but not wrong.)

### C-2.3-connected
**Verdict:** ⚠ under-justified
**Quote (lines 409-411):** "Without (R1), the calibration manifold has a Z_2^d ambiguity (sign flips in each coordinate of r), and the induced ``confidence sets'' can be disconnected. (R1) eliminates this."
**Diagnosis:** The Z_2^d count is *much* too small. Any measure-preserving bijection of N(0, I_d) to itself, applied componentwise to r, stays in M; concretely:
- All orthogonal transformations O(d) preserve N(0, I_d) (rotation-invariance), a continuous family of dimension d(d-1)/2.
- Even in 1D, in addition to sign-flip there are infinitely many measure-preserving (non-monotone) bijections of N(0,1) — e.g. swap two intervals of equal mass — so the ambiguity is uncountably infinite, not Z_2.
The §3.3 paragraph at lines 488-494 says exactly this: "any composition of a monotone transport with a measure-preserving bijection of N(0,1) to itself remains in the manifold; the manifold is infinite-dimensional." So §2.3 *contradicts* §3.3 on the size of the ambiguity.
Furthermore, (R1) alone does *not* eliminate the full ambiguity even in 1D: it eliminates the sign-flip + non-monotone permutations of intervals, but in d>1 it eliminates only the coordinate-wise monotone reorderings, not the rotational ambiguity (rotations don't preserve coordinate-wise monotonicity, but they do preserve calibration). The d>1 rotation issue is what §11.1 (OP-11.1) flags as open. So (R1) is sufficient to eliminate Z_2^d in 1D but not in d>1.
**Suggested fix:** Replace "Z_2^d ambiguity" with "an infinite-dimensional ambiguity (including sign flips, non-monotone measure-preserving rearrangements, and in d>1 orthogonal rotations of r, since N(0, I_d) is rotation-invariant)." Then say "(R1) eliminates the monotone-rearrangement part of the ambiguity in 1D; in d>1, eliminating the rotational ambiguity requires the autoregressive triangular structure of §6 (see also §11.1)." Reconciles with §3.3 and §11.1.

### C-2.4-folding
**Verdict:** ⚠ under-justified
**Quote (lines 422-429):** "In §3 we will see that NF-MLE applied to a non-bijective r has *spurious* sub-optima below the entropy of the true model... Note that under folding, p_hat_r does *not* integrate to one in X at fixed theta, so there is no lower bound on -log p_hat_r from the true conditional entropy; the loss can dip arbitrarily far below the true model's NF-MLE value."
**Diagnosis:** The qualitative argument is correct in direction (the §8.4 ablation gives loss 0.56 vs truth's 0.88, which is direct empirical evidence). But the §2.4 wording leans on §3 / §8.4 to do the actual work — §2.4 by itself is informal and contains two slightly slippery moves:
(i) The phrase "the loss can dip arbitrarily far below the true model's NF-MLE value" is asserted but not derived. *Arbitrarily* is a strong word — it requires unboundedness from below, which in turn requires the architectural class to support unbounded fold amplitude (i.e. unbounded `|det partial r/partial X|`). For an a-priori-Lipschitz architecture this is bounded, and the loss cannot actually be arbitrarily low. For UMNN-via-autograd with no Lipschitz constraint, it can. The text doesn't distinguish.
(ii) "p_hat_r does not integrate to one" is correct under proper folding (multivalued preimages summed without the absolute-value-of-Jacobian correction), but the formula in §2.4 line 417 *does* contain the Jacobian factor — the issue is rather that change-of-variables on a non-injective map produces an *over-counting* (sum over preimages), and the surrogate density as defined ignores this sum and writes one term. This is what §3.5 (lines 561-573) makes explicit; §2.4 should at least flag the over-counting/preimage-sum issue.
**Suggested fix:** Tighten §2.4 to: "When r is not injective in X, the change-of-variables formula requires summing over preimages, and the surrogate p_hat_r as defined keeps only one preimage's contribution. The result fails to integrate to one in X; consequently -log p_hat_r is not bounded below by the conditional entropy, and (for architectures that admit unbounded fold amplitude) the NF-MLE loss is not bounded below at all. §3.5 develops the formal version; §8.4 gives the empirical demonstration." Avoids the "arbitrarily" overclaim while preserving the message.

## Cross-Part findings

- [§3.3, lines 488-494]: The §3.3 description of the calibration manifold ambiguity ("any monotone transport composed with a measure-preserving bijection of N(0,1); the manifold is infinite-dimensional; in multivariate, rotations") directly contradicts §2.3's claim of a finite Z_2^d ambiguity. Need to reconcile (Part II claim, but the source is the C-2.3-connected mismatch).

- [§4.4, lines 800-832]: §4.4 introduces (R3) and Theorem A* precisely because the Lipschitz-only version of the §4.2 / §4.3 argument fails — which is direct evidence against the §2.2 claim that "C^1 *or* Lipschitz a.e.-differentiable diffeomorphism" are interchangeable choices for (R2). §4.4 confirms the false-equivalence in R2; Part III should re-examine whether the Theorem A* statement is being properly forward-referenced from §2.2.

- [§11.1, lines 1810-1822]: The §11.1 open problem ("orthogonal rotations of r preserve calibration since N(0, I_d) is rotation-invariant") confirms that the d>1 rotational ambiguity is *not* eliminated by (R1) — direct conflict with C-2.3-connected's "(R1) eliminates this". Part VII should be cross-checked.

- [§11.4 + §1.3, lines 305-309 vs. §11.4 open problem]: C-1.3-rho's "does not depend on rho given full support" is in tension with §11.4 ("Misspecification: NF-MLE population minimizer becomes moment-projection") and with the under-parameterized-architecture case throughout. Part VII / Part VI should be reviewed for whether they consistently assume saturated-architecture or moment-projection framing.
