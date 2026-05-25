# Round 1 — Cross-cutting Consistency Review

**Reviewer scope.** Dependency chains spanning multiple Parts only; no
per-claim re-vetting (parallel critics own that).

## Summary
- Dependency chains audited: 5
- Major issues: 2 (one load-bearing gap in T-A→T-C/T-A-d carry-over;
  one notation conflict between (R3) and (R3_U))
- Notes / minor inconsistencies: 4

---

## Per-chain findings

### Chain 1: C-3.2-KL → L-4.3 → T-A → T-C → T-A-d
**Status:** ⚠ minor inconsistency (with one load-bearing gap that should be patched, but is recoverable).

**Trace.**
1. §3.2 derives `L(r) = E_ρ[KL(p ‖ p̂_r)] + const` and concludes that
   `argmin L = M_F` *"assumed to satisfy (R2) so p̂_r is a valid
   density"* (line 472). The KL non-negativity needs *both* sides to be
   normalized; the proof implicitly uses that the change-of-variables
   formula gives a normalized density, which requires (R2).
2. L-4.3 (§4.2, lines 763–774) invokes §3.2 verbatim: *"From §3.2,
   L(r) is the expected KL ... Under (R2), p̂_r pushes X | θ to
   N(0,1) iff r(θ_0; X) | θ_0 ~ N(0,1)."* This is fine in 1D.
3. T-A's proof (§4.3) calls L-4.3 to get conditional calibration, then
   L-4.2 to upgrade to monotonicity, then L-4.1 to pin down the form.
4. T-C (§5.2): *"Apply Theorem A in t = T(X) coordinates. The
   pushforward density of T(X) | θ_0 is continuous and positive."*
5. T-A-d (§6.3): *"by Theorem A applied in X_k-coordinates with
   conditional law `F_k^{(θ)}(· | X_<k)` playing the role of the
   Gaussian source."*

**Issue.**

**(1a) L-4.2 carry-over.** L-4.2's proof (§4.2 lines 746–761) uses
*only* (i) that the source density `φ(X_i − θ_0)` is positive at the
critical point and (ii) that the *target* density `φ(v_*)` is finite.
Both hold for arbitrary continuous positive source pushed to a Gaussian
target. So the proof *does* carry over to T-C and T-A-d's inductive
step, but **the manuscript never says this**. T-C and T-A-d cite
"Theorem A" as a black box without flagging that L-4.2 has been
re-applied with a non-Gaussian source. A careful reader cannot verify
the application without re-deriving L-4.2 in the general case. **This
is the most load-bearing gap of Chain 1.**

**(1b) (R2) vs (R2^auto) substitution in §3.2.** The §3.2 KL argument
is written in terms of (R2) (full-rank Jacobian a.e.). In §6, the flow
class is (R2^auto) (autoregressive *triangular* with each
`X_k ↦ r_k` strictly monotone at fixed `X_{<k}`). The two are not
equivalent statements: (R2^auto) implies (R2) (because a triangular
matrix with non-zero diagonal has full rank), so the §3.2 KL argument
*does* carry over. But §6.3 never explicitly checks this; it inherits
"strict propriety" from §3.2 silently. Worth one sentence.

**(1c) L-4.3 in §6.3.** T-A-d's inductive step derives
`r_k(θ, X_≤k) | θ, X_<k ~ N(0,1)` from "the calibration constraint
... implies that the components are jointly standard normal." This
reasoning is *NF-target-distribution-shaped*, not L-4.3-shaped; it
needs the joint marginal `r | θ ~ N(0, I_d)` plus that `r_1,...,r_{k−1}`
are deterministic functions of `(θ, X_<k)`. Both are explicit, so
the step is fine — but the proof never re-invokes §3.2/L-4.3 here, so
the connection back to the KL argument is broken. A reader who wonders
*why* M_F is even the right minimizer set in the multivariate case has
to chase the implicit (R2^auto) ⇒ (R2) inclusion through §3.2.

**Suggested fix.**
- In L-4.2's proof, add one sentence: *"the only properties used of
  the source `N(θ_0,1)` are continuity, positivity at any critical
  point, and a target density finite at all attained values; the lemma
  thus applies verbatim to any continuous positive source pushed to a
  continuous target."* Then the T-C / T-A-d invocations are clean.
- In §3.2, state explicitly that the argument applies to (R2^auto)
  as well (since (R2^auto) ⇒ (R2)).

---

### Chain 2: C-3.5-loss-below ↔ C-2.4-folding ↔ E-8.4-ablation ↔ C-8.4-ablation-interp
**Status:** ✓ consistent (one minor logical wrinkle worth flagging).

**Trace.**
1. §2.4 informally introduces the folding mechanism, defers to §3 / §8.4.
2. §3.5 (lines 561–574): "Because the folded `p̂_r` is not normalized
   in X at fixed θ, the loss `−E_X[log p̂_r]` has no lower bound at
   the conditional entropy, so it can fall *below* the loss attained
   by the true `r*`."
3. §8.4 ablation: loss 0.56 < truth 0.88; RMSE 1.56; conditional PIT
   fails.
4. §8.4 interpretation: ties the empirical sub-truth loss back to §3.5.

**Issue (minor logical wrinkle).** §3.5 says the loss "*can* fall
below truth"; §8.4 says the loss "*would be impossible* if the
ablation model lived in a valid (bijective-in-X) class." These are
logically aligned but the §8.4 statement is **stronger**: it's
asserting that loss < truth is *sufficient* evidence for non-bijectivity.
That sufficiency direction is correct (if `p̂_r` were normalized, then
`E[−log p̂_r] ≥ H(X|θ)` by Gibbs, so loss < truth ⇒ not normalized in
X), but the manuscript never explicitly writes that direction of the
implication. C-8.4-ablation-interp treats the ablation as "direct
empirical evidence for C-3.5-loss-below," but C-3.5-loss-below is
really just the existence of *one* mechanism (folding) that *could*
produce sub-truth loss; the ablation, strictly, only certifies "some
deviation from valid-density structure," not folding *specifically*.

In practice the other plausible mechanism (a sign mistake in the
Jacobian, a numerically broken log-det) is ruled out by the
architecture (autograd computes the actual Jacobian of a well-defined
`r`), so folding is the only failure mode left. This is fine but
should be stated.

**Suggested fix.** In §8.4 ablation paragraph, state the Gibbs
inequality direction explicitly: "any `p̂_r` that integrates to 1 in X
satisfies `E[−log p̂_r] ≥ H(X|θ)`; loss 0.56 < 0.88 = H ⇒ p̂_r is not
normalized in X."

---

### Chain 3: Regularity conditions across Parts — R1, R2, R1^auto, R2^auto
**Status:** ⚠ minor inconsistency.

**Trace.** See the audit table below for the full map. Two specific
cross-Part concerns:

**(3a) Is "the multivariate (R1) of §2.2" the same as "(R1^auto) of §6.1"?**
§2.2 (line 359–364): *"(R1) ... For ρ⊗P_θ-a.e. X and each coordinate
k, the map θ_k ↦ r_k(θ; X) is strictly increasing when other
components are held fixed (autoregressively in d > 1; see §6)."*
§6.1 (R1^auto): *"r_k depends on θ only through θ_{≤k}, and θ_k ↦ r_k
is strictly increasing at fixed θ_{<k}, X."*

These are **not the same**. (R1) of §2.2 has *all* other components
of `θ` held fixed (it gives partial monotonicity in θ_k holding *all*
other θ_j fixed); (R1^auto) restricts `r_k` to depend only on θ_{≤k}.
The latter is the architectural class (triangular), the former is the
calibration-of-CD-level-sets requirement. (R1^auto) ⇒ §2.2's (R1) in
the autoregressive sense, but the §2.2 wording elides this: it
parenthetically says "autoregressively in d > 1," signaling that the
"strictly increasing" coordinate-wise statement is *the* (R1^auto)
restriction. A naive reader could think the §2.2 (R1) is the
non-restricted full-monotonicity condition.

**(3b) (R2) vs (R2^auto).** §2.2 (R2): C¹ diffeomorphism, full-rank
Jacobian a.e. §6.1 (R2^auto): triangular structure plus strict
monotonicity in the diagonal entry. (R2^auto) ⇒ (R2) (as noted in
Chain 1). But the manuscript at one point (§8.5 synthesis,
line 1617) writes *"(R2) is a correctness requirement"* citing §8.4 —
where the ablation experiment actually violates (R2^auto), not (R2)
*per se* (it's the architectural enforcement of monotone-in-X in the
triangular class that's at issue). The slogan in §8.5 conflates the
two, though in practice this is benign.

**Suggested fix.** In §6.1, explicitly note "(R1^auto) and (R2^auto)
are the autoregressive specializations of (R1) and (R2); they imply
(R1) (coordinate-wise) and (R2) (full-rank Jacobian) respectively."

---

### Chain 4: (R3) usage and "automatic in continuous case" claim
**Status:** ⚠ minor inconsistency.

**Trace.**
1. §4.4 defines (R3): `|∂_X r| ≥ c > 0` on differentiability set. (R3)
   is *stronger* than "strictly monotone in X" — it requires a uniform
   lower bound, not just bijection.
2. T-A* (§4.4) uses (R3) plus (R1), (R2), Lipschitz to recover T-A.
3. §5.7.2 (line 1007): *"Why (R4) is automatic in the continuous case:
   Lemma 4.2 (or (R3)) implies ψ_{θ_0} is monotone in X."*
4. OP-11.3 declares whether (R3) follows from (R1)+Lipschitz+calibration
   alone is open.

**Issue.** §5.7.2 says "(R3) implies monotone in X" — which is
correct but weaker than needed for what (R3) really buys you. In §4.4
(R3) is doing the *Lipschitz strengthening* job (no critical points
in the a.e. sense). For the continuous-case (R4)-automatic claim,
what is needed is bijectivity (i.e., (R2) in C¹ form), not a uniform
lower bound. So §5.7.2 is slightly over-citing: (R3) in §4.4's
strong sense is more than sufficient; the relevant condition is just
strict monotonicity from L-4.2.

Separately: the manuscript's notation (R3_U) in §5.7.2 (line 965) for
"strict monotone in U" is a *third* condition distinct from (R3). The
two names are visually similar and could trip up a reader who's just
seen (R3) in §4.4 — a footnote disambiguating would help.

OP-11.3 vs the (R3)-automatic claim: §4.4 says (R3) *is* automatic
under C¹ + calibration (via L-4.2), and §5.7.2 reiterates this. So
OP-11.3 is specifically about the **Lipschitz** case where L-4.2 fails
and one cannot derive (R3) from the other conditions. Consistent, but
the open-problem framing in OP-11.3 should explicitly say "Lipschitz,
not C¹."

**Suggested fix.** §5.7.2 line 1007: change "Lemma 4.2 (or (R3))" to
"Lemma 4.2 (in C¹) or strict monotonicity in X under (R3)." §11.3:
add "in the Lipschitz case" to the open problem.

---

### Chain 5: Theorem statement-format consistency (function classes)
**Status:** ✓ consistent, one mild discrepancy.

**Trace.**
- T-A: `F_{C¹} := { r : R² → R satisfying (R1), (R2), with X ↦ r(θ_0; X) ∈ C¹(R) for each θ_0 }`.
- T-C: `F^T_{C¹} := { r : t ↦ r(θ_0; T^{-1}(t)) ∈ C¹ for each θ_0 }`.
- T-A-d: "autoregressive triangular pivots satisfying (R1^auto) and
  (R2^auto) with each `r_k ∈ C¹` in its `X_k`-argument."

**Issue (mild).** T-C's class spec drops the explicit `(R1), (R2)`
membership requirement that T-A's spec includes (it's implied by
*"under the setup of §5.1,"* but still). T-A-d's class spec gives only
*"C¹ in its X_k-argument"* — which is what L-4.2 needs for the
inductive step, so this is technically the right condition (no joint
C¹ in `(θ, X_k)` required). Still, the three theorems read like
they're stated by three different authors. Function-class definitions
in T-C and T-A-d would ideally use the same pattern as T-A.

**Suggested fix (cosmetic).** Add `(R1)/(R1^auto)` and `(R2)/(R2^auto)`
to the explicit definitions of `F^T_{C¹}` and the multivariate class.

---

## Regularity conditions audit

| ID | Defined at | Used at | Cross-Part consistency |
|---|---|---|---|
| (R1) | §2.2 (line 359) | §2.3, §3.1, §3.5, §4.1 (T-A class), §4.3, §4.4 (T-A*), §5.1 verif., §5.3 fig./examples, §5.7.2, §6.1 (motivates R1^auto), §8.4, §8.5, §9 | Wording in §2.2 parenthetically subsumes the autoregressive case. Could be sharper. |
| (R2) | §2.2 (line 366) | §2.4, §3.1, §3.2 (KL needs R2 for valid density), §3.5, §3.7 Class 5, §4.1 (T-A class), §4.3, §4.4 (T-A*), §5.1 verif., §6.1 (motivates R2^auto), §8.4 (ablation slogan), §8.5, §9 | (R2^auto) ⇒ (R2) but §6/§8 use the names interchangeably. |
| (R3) | §4.4 (line 813) | §4.4 (T-A* proof), §5.7.2 ("R4 automatic"), §11.3 (OP) | Cited in §5.7.2 in a weakened sense (just "monotone in X"); naming collision with (R3_U). |
| (R3_U) | §5.7.2 (line 965) | §5.7.2 (T-C*), §5.7.2 counterexample | Distinct from (R3) but easy to confuse. |
| (R4) | §5.7.2 (line 971) | §5.7.2 only | Cleanly demarcated as discrete-only. |
| (R1^auto) | §6.1 (line 1053) | §6.1, §6.2 (connected level sets), §6.3 base & inductive steps, T-A-d | Internally consistent. |
| (R2^auto) | §6.1 (line 1058) | §6.1, §6.2, §6.3 base & inductive steps, §6.3 remark 3, T-A-d, §8.4 ablation interp | Internally consistent. |

**Key audit findings:**
1. (R2) and (R2^auto) used interchangeably in §8.4–§8.5 (the slogan
   "(R2) is a correctness requirement"). This is benign — (R2^auto)
   ⇒ (R2) — but should be tightened.
2. (R3) vs (R3_U) is a notation collision.
3. (R3) is invoked in §5.7.2 to make a *weaker* statement than (R3)
   actually says (just "monotone in X" rather than "uniformly bounded
   away from 0").

---

## Status-table cross-check

The .tex's "Status of claims" table (lines 1891–1944) marks the
following as **Proven**:

| Claim in .tex table | Inventory ID(s) | Reviewer note |
|---|---|---|
| NF-MLE strictly proper for M | C-3.2-KL, C-3.2-strictprop, L-4.3 | Holds under (R2). The table notes "KL non-negativity, conditional" but does not flag the *implicit* dependence on R2 for normalizability. |
| Theorem A (1D, C¹) | T-A | OK — relies on L-4.1, L-4.2, L-4.3. |
| Theorem A* (Lipschitz) | T-A* | Adds (R3). OK. |
| Schweder–Hjort bridge | P-4.5 | OK. |
| Theorem C | T-C | **Marked Proven, but as flagged in Chain 1, the proof says "same three-lemma structure in transformed coordinates" without verifying L-4.2 transfers from a Gaussian source to an arbitrary continuous positive source.** Manuscript treats this as immediate; reviewer would flag as **under-justified rather than proven**. |
| Theorem C* (discrete) | T-C* | OK. |
| Theorem A-d | T-A-d | **Marked Proven, but (a) inductive step uses Theorem A in `X_k`-coordinates without re-verifying the L-4.2 carry-over, and (b) the §3.2 KL argument is implicitly extended from (R2) to (R2^auto) without comment.** Reviewer would mark **under-justified** modulo the trivial implications, which the manuscript leaves unwritten. |
| Corollary KR = L⁻¹(θ − X) | C-6.3-Gaussian | OK — direct Cholesky computation. |
| Empirical claims (§8.1–§8.4) | E-8.1–E-8.4 | OK, marked "Validated." |
| OP rows | OP-11.1, OP-11.2, OP-11.3, OP-11.4, OP-11.5, OP-11.6 | OK, marked "Open." |

**Claims in inventory NOT represented in the .tex's Status table:**

- C-1.3-C1, C-1.3-C2, C-1.3-rho (definitions/calibration goal)
- D-pivot, D-M (definitions)
- C-2.3-connected, C-2.4-folding (mechanism claims)
- C-3.3-manifold-large, C-3.4-SNL, C-3.5-loss-below, C-3.7-* (taxonomy claims)
- C-5.3-tloc, C-5.3-exp (worked examples)
- C-5.7-randU, C-5.7-counterex, C-5.7-R4-auto, C-5.7-midp
- C-6.2-tri-jac, C-6.2-Olin, C-6.2-connectedCI
- C-6.3-ordering, C-6.3-non-unique-arch, C-6.3-R2auto-role
- C-6.4-CI, C-6.4-CIalgo
- §7 architecture/diagnostic claims, §9 recipe claims, §10 positioning claims
- OP-11.7, OP-11.8

This is expected — the .tex table is a *summary* of major theorems and
empirical headlines, not an exhaustive list. **No claim marked
"Proven" in the table is flagged "error" in the inventory.** Two
claims (T-C and T-A-d) marked "Proven" in the table are reviewer-flagged
as **under-justified** (Chain 1) until L-4.2's carry-over is made
explicit.

---

## Bottom-line for the integrator

The one substantively load-bearing finding is **Chain 1's L-4.2
carry-over**: §5.2 (T-C) and §6.3 (T-A-d) both invoke "Theorem A in
[transformed/X_k] coordinates" without ever spelling out that L-4.2's
proof — which is *literally* written for a Gaussian source — does in
fact transfer to an arbitrary continuous positive source pushed to a
Gaussian target. The transfer is mechanically valid (the proof only
uses positivity of the source at the critical point and finiteness of
the target), but the manuscript leaves the reader to discover this.
A one-sentence generalization in L-4.2 closes both T-C and T-A-d in
one move.

Everything else is naming hygiene: (R1) vs (R1^auto), (R2) vs
(R2^auto), and (R3) vs (R3_U) are conflated in three places. None of
this is wrong — the implications go the right way — but it leaves the
reader doing the bookkeeping. Per round-1 rules these are not flagged
as errors; they are minor inconsistencies.
