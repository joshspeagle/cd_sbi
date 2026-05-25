# Round 1 — Part II (Loss design) Review

Scope: §3.1–§3.7 of `cd_sbi_v7.tex`, lines ~437–690. Numbers in
tables / experiments assumed correct. Citation correctness deferred to
round 2.

## Summary

- Total claims vetted: 13
- ✓ holds: 8 | ⚠ under-justified: 5 | ✗ error: 0
- Cross-Part findings: 2

The load-bearing core of Part II — that NF-MLE is strictly proper for
M_F via KL non-negativity, that (R2) is what licenses this, and the
algebraic propriety direction for CRPS — is **substantively correct**.
The main weaknesses are presentational rather than mathematical: §3.5's
"loss can dip below the conditional entropy" argument is true but
informal in §3.5 itself (the more precise statement is in §2.4, with
§3.5 referring back); §3.7's CRPS derivation is correct in substance
but uses divergent-looking integrals without flagging regularization;
the Class 2 fragility list contains one near-tautological item; and the
"Hermans pattern lives here" claim has no explicit counterexample.

## Per-claim findings

### D-NF
**Verdict:** ✓
**Quote:** "A normalizing flow is a parametric family of diffeomorphisms
\(r: \mathcal{X} \to \mathbb{R}^d\) that transforms a complicated source
density \(p(X)\) into a simple base density \(\phi_d\) ... the implied
density on \(\mathcal{X}\) is the standard change-of-variables
expression \(\hat p_r(X | \theta) = \phi_d(r(\theta, X)) \cdot
|\det \partial r / \partial X|\)."
**Diagnosis:** Standard change-of-variables formula. r is conditional
on θ here, so technically each fixed θ gives a diffeomorphism on X; the
text writes \(r:\mathcal{X}\to\mathbb{R}^d\) which is slightly loose but
matches the convention used elsewhere.

### D-NFMLE
**Verdict:** ✓
**Quote:** boxed equation, line 452: \(L(r) = \mathbb{E}_{(\theta,X)\sim
\rho\otimes P_\theta}[\tfrac12\|r\|^2 - \log|\det \partial r/\partial X|]\).
**Diagnosis:** Correct: \(-\log\phi_d(r) = \tfrac12\|r\|^2 +
\tfrac{d}{2}\log(2\pi)\); dropping the additive constant gives the
boxed expression, which is the NLL under \(\hat p_r\).

### C-3.2-KL
**Verdict:** ⚠ under-justified (substantively correct, light on
hypotheses)
**Quote:** "\(L(r) = \mathbb{E}_\rho[\mathrm{KL}(p(\cdot|\theta) \| \hat
p_r(\cdot|\theta))] + \mathrm{const}\), where the constant is the
conditional entropy of \(X | \theta\) averaged over \(\rho\)
(independent of \(r\))."
**Diagnosis:** The algebra is correct under (R2): given (R2), \(\hat
p_r(\cdot|\theta)\) is a *normalized* density on \(\mathcal{X}\) at
every fixed \(\theta\), so

\(\mathbb{E}_{X|\theta}[-\log\hat p_r] = H(X|\theta) + \mathrm{KL}(p\|\hat p_r)\)

and averaging over \(\rho\) yields the boxed identity with const \(=
\mathbb{E}_\rho[H(X|\theta)]\), which is genuinely independent of
\(r\). The strict-propriety conclusion (minimum exactly on
\(\mathcal{M}_\mathcal{F}\)) follows from KL≥0 with equality iff
densities agree a.e.

Two presentational gaps: (1) The "valid density" caveat is only
mentioned parenthetically at end ("assumed to satisfy (R2) so \(\hat
p_r\) is a valid density"). The identity literally *requires* this —
without (R2), \(\hat p_r\) need not normalize and the KL identity
fails. Worth a one-line lead with the assumption rather than a
parenthetical postscript. (2) Need to assume \(p(\cdot|\theta) \ll \hat
p_r(\cdot|\theta)\) (absolute continuity) for KL to be finite, which
holds automatically when both densities are continuous and \(\hat
p_r\) has the support of \(p\); a sentence on regularity (which the
manuscript does in (R1)/(R2) via diffeomorphism onto image) would
close this.

**Suggested fix:** Add one explicit line above the identity: "Under
(R2), \(\hat p_r(\cdot|\theta)\) is a probability density on
\(\mathcal{X}\) (integrates to 1), so the standard identity \(\mathbb{
E}_{X|\theta}[-\log\hat p_r] = H(X|\theta) + \mathrm{KL}(p \| \hat p_r)\)
holds; averaging over \(\rho\) gives the boxed equation."

### C-3.2-strictprop
**Verdict:** ✓
**Quote:** "NF-MLE is strictly proper for the calibration manifold
within the architectural class. No auxiliary independence or sharpness
penalties are needed."
**Diagnosis:** Follows immediately from C-3.2-KL once that identity
holds. The "no auxiliary penalties needed" is then a corollary, not a
separate claim. The reasoning is sound: a single per-sample
log-density term enforces *both* per-θ calibration (Φ(r)|θ ~ U) and the
correct \(|\det \partial r/\partial X|\), which together pin down the
full conditional density.

### C-3.3-manifold-large
**Verdict:** ⚠ under-justified
**Quote:** "In 1D, any composition of a monotone transport \(X|\theta
\to \mathcal{N}(0,1)\) with a measure-preserving bijection of
\(\mathcal{N}(0,1)\) to itself remains in the manifold; the manifold is
infinite-dimensional."
**Diagnosis:** The 1D infinite-dimensionality claim is correct: any
measure-preserving bijection \(\psi: \mathbb{R}\to\mathbb{R}\) (not
just continuous) of N(0,1) composed with a calibrated r yields another
calibrated r, and these form an infinite-dimensional group. The
multivariate "rotation-ambiguous" remark is also correct since
\(\mathcal{N}(0,I_d)\) is O(d)-invariant. *However*, the text doesn't
note that without (R1)/(R2) constraints, even the C¹ requirement
doesn't kill the 1D ambiguity entirely — there exist non-monotone C¹
diffeomorphisms preserving N(0,1) measure only if they fold (which
breaks (R2)). The argument is essentially that *unconstrained* M is
huge, and (R1)+(R2)+(C¹) shrinks it. That's fine but the dimension
reduction step could be tighter.

**Suggested fix:** Clarify that "the manifold over the unconstrained
class" refers to dropping (R1) and (R2); within C¹ pivots satisfying
(R1), (R2), the 1D singleton result (Theorem A) is what carries the
identification.

### C-3.4-SNL
**Verdict:** ✓
**Quote:** "The objective \(L(r)\) is *identical* to the loss used by
Sequential Neural Likelihood (SNL; Papamakarios et al. 2019)."
**Diagnosis:** SNL trains a conditional normalizing flow \(q_\phi(x|
\theta)\) by maximum likelihood on \((\theta, x)\) pairs from the
proposal, i.e. minimizes \(\mathbb{E}_{(\theta,x)\sim\rho\otimes
P_\theta}[-\log q_\phi(x|\theta)]\). This is exactly the NF-MLE loss
when the flow is parameterized as \(r:\mathcal{X}\to\mathbb{R}^d\) with
base \(\phi_d\) via the change-of-variables formula. The "loss is
identical" claim is therefore correct *at the level of the objective
function*. The architectural and inference-time differences are
correctly itemized in the longtable (SNL uses MCMC on \(\hat p\cdot
\pi\); CD-SBI uses pivot inversion). Minor wrinkle: SNL is *sequential*
— it refits on simulator outputs near the current posterior — whereas
"NF-MLE" as defined here uses a fixed proposal \(\rho\). At the
single-round / amortized limit the losses coincide; the manuscript is
implicitly comparing to *non-sequential* NLE. This is a fine
abstraction (the loss is per-round identical), but worth flagging.

**Suggested fix:** In §3.4, add: "In its single-round (non-sequential)
variant — and per-round in the sequential version — SNL's training
objective is exactly \(L(r)\)."

### C-3.5-loss-below
**Verdict:** ⚠ under-justified (substantively correct, mechanism in
§3.5 itself is informal)
**Quote:** "Because the folded \(\hat p_r\) is not normalized in \(X\)
at fixed \(\theta\), the loss \(-\mathbb{E}_X[\log\hat p_r]\) has no
lower bound at the conditional entropy, so it can fall *below* the loss
attained by the true \(r^*\)."
**Diagnosis:** The statement is true and the §2.4 version is more
explicit, but the §3.5 sentence is terse. The precise mechanism: if
\(\hat p_r\) doesn't integrate to 1 (call its integral \(Z(\theta)\)),
then

\(\mathbb{E}_{X|\theta}[-\log\hat p_r] = \mathrm{KL}(p \| \hat p_r/Z)
+ H(X|\theta) - \log Z(\theta).\)

KL≥0 still holds (KL is between *probability* densities \(p\) and
\(\hat p_r/Z\)), but \(-\log Z\) can be arbitrarily negative when
\(Z>1\). Folding inflates local \(|\det \partial r/\partial X|\) so
that \(\hat p_r\) sums to more than 1 over preimages, driving
\(Z(\theta)>1\) and pulling the loss below \(H(X|\theta)\). The
asymmetry "KL≥0 only for proper densities" is implicit but not stated.
This same logic appears in §2.4 — fine to keep §3.5 brief if §3.5
cross-references §2.4 explicitly. It currently does not.

**Suggested fix:** Either (a) add the one-line algebraic statement
above with the \(-\log Z(\theta)\) term made explicit, or (b)
cross-reference §2.4 directly: "see §2.4 for the precise mechanism: the
folded density's integral \(Z(\theta)\) exceeds 1, contributing a
\(-\log Z(\theta) < 0\) shift that breaks the conditional-entropy lower
bound."

### C-3.7-Class1
**Verdict:** ⚠ under-justified (true claim, no counterexample given)
**Quote:** "marginal \(\Phi(r) \sim U(0,1)\) does not imply conditional
\(\Phi(r) | \theta \sim U(0,1)\) — the Hermans trust-crisis pattern
lives precisely here, with marginal PIT passing while conditional
fails."
**Diagnosis:** The implication is genuinely one-way: marginal calibration
on the joint training distribution = E_θ[Law(Φ(r)|θ)] = U(0,1) doesn't
pin down Law(Φ(r)|θ) for each θ. A standard counterexample: take r
that is biased high at one half of the θ-range and biased low at the
other, in a way that averages out marginally. This is well-known in
the calibration literature (cf. conditional vs marginal coverage in
prediction interval theory), and the Hermans 2022 "trust crisis"
empirical observation is consistent with it. The claim is correct, but
the manuscript asserts rather than demonstrates. For a load-bearing-
adjacent claim used to motivate why Class 1 is insufficient and why
NF-MLE is needed, a one-line constructive counterexample would
strengthen it. (Even a sketch: "e.g., if r(θ,X) = (X-θ)·sign(θ) on
\(\theta\in\{-1,+1\}\) with \(\rho\) uniform on \(\{\pm 1\}\), then
marginally \(\Phi(r)\sim U(0,1)\) but conditional fails at θ=-1.")

**Suggested fix:** Add a one-sentence explicit construction in §3.7
showing marginal-without-conditional. Or footnote pointing to the
folklore counterexample in conditional-coverage literature.

### C-3.7-Class2
**Verdict:** ⚠ under-justified (one near-tautological reason)
**Quote:** "empirically fragile in three ways: (i) the HSIC term has a
weak gradient signal far from the optimum compared to a per-sample
density target; (ii) the marginal-calibration term and the
independence term must be balanced via hyperparameters, with no
principled choice; (iii) the marginal and conditional terms can
decouple in finite samples, reproducing the Class-1 pathology."
**Diagnosis:** (i) and (ii) are substantive practical claims — (i) is
a standard observation about kernel-based statistics having flat
gradients relative to log-density losses (cited e.g. in the
MMD-vs-MLE training literature); (ii) is the well-known difficulty of
multi-term losses without principled weights. (iii) is closer to
tautological: it essentially says "if HSIC's finite-sample estimator
isn't tight enough, you don't get conditional calibration." This *is*
true but reduces to "the population-level argument doesn't transfer to
finite samples," which is a generic critique applicable to any
estimator. As a "fragility" claim, (iii) is empirical not structural,
and the manuscript doesn't distinguish.

Also: the manuscript says Class 2 is "strictly proper for M" — true at
the *population* level when HSIC=0 ⇔ independence. The fragility
listed is finite-sample. This distinction is correct but worth
flagging.

**Suggested fix:** Sharpen (iii) to something like "finite-sample HSIC
estimators have variance that scales unfavorably with the dimension of
\(\theta\), so achieving independence in the empirical loss does not
ensure independence in the underlying distribution." Or drop (iii) and
keep (i) and (ii) which are the substantive points.

### C-3.7-Class3
**Verdict:** ✓
**Quote:** "Discretize \(\Theta\) into bins; within each bin, apply a
two-sample distance ... requires \(K^d\) bins in \(d\)-dimensional
\(\Theta\), defeating the entire motivation for amortized inference."
**Diagnosis:** Correct scaling argument. The K^d count assumes K bins
per coordinate and uniform tensor-product binning; smarter adaptive
schemes (e.g. KD-trees) might do better, but the fundamental issue —
that stratified estimators require effective sample sizes per bin and
the bin count scales exponentially with \(d\) — stands.

### C-3.7-CRPS
**Verdict:** ⚠ under-justified (algebra correct, integrals divergent
without regularization)
**Quote:** "\(\mathbb{E}_{Y\sim G}[\mathrm{CRPS}(F,Y)] = \int F(z)^2 dz
+ \int G(z)(1-2F(z))dz\) ... For \(F=\Phi\), the coefficient
\(1-2\Phi(z)\) is positive on \(z<0\) and negative on \(z>0\), so the
pointwise-optimal \(G\) satisfies \(G(z)=0\) for \(z<0\) and \(G(z)=1\)
for \(z>0\) — i.e., \(G=\delta_0\)."
**Diagnosis:** Algebraic expansion is correct:

\(\mathrm{CRPS}(F,y) = \int(F(z)-\mathbf{1}\{y\le z\})^2 dz = \int
F^2 dz - 2\int F(z)\mathbf{1}\{y\le z\}dz + \int\mathbf{1}\{y\le z\}dz\)

(using \(\mathbf{1}^2 = \mathbf{1}\)). Taking \(\mathbb{E}_{Y\sim G}\),
\(\mathbb{E}[\mathbf{1}\{Y\le z\}] = G(z)\), giving \(\int F^2 dz +
\int G(z)(1-2F(z))dz\). ✓

The minimization argument: G is a CDF (monotone, [0,1]-valued, →0 at
−∞ and →1 at +∞). To minimize \(\int G(z)(1-2\Phi(z))dz\), we want
G(z) small where (1-2Φ(z))>0 (z<0) and large where (1-2Φ(z))<0 (z>0).
The pointwise lower envelope subject to "G is a valid CDF" gives G(z)
= 0 for z<0 and G(z) = 1 for z>0, i.e. the step function for
\(\delta_0\). ✓

*Caveat the paper doesn't flag*: both individual integrals \(\int
G(z)(1-2\Phi(z))dz\) and \(\int F(z)^2 dz\) are infinite as written
(\(\int F^2\) diverges since \(F\to 1\) at +∞; \(\int G dz\) diverges
similarly). The standard fix is to use the regularized form \(\int
(F-\mathbf{1}\{z\ge 0\})^2 dz - \int(G(z)-\mathbf{1}\{z\ge 0\})\cdot
(1-2F(z))dz\) — or equivalently, work with the dual form
\(\mathrm{CRPS}(F,y) = \mathbb{E}_{Z\sim F}|Z-y| - \tfrac12
\mathbb{E}_{Z,Z'\sim F}|Z-Z'|\). The pointwise-optimum conclusion is
robust to this regularization (the comparison G=δ_0 vs G=N(0,1) yields
a finite *difference*), but as written the derivation looks like a
divergent-minus-divergent. A reader doing the algebra carefully will
worry about this.

Direction-of-propriety claim: CRPS is strictly proper in the "fix Y∼G*,
minimize over F" direction. Here Class 4 fixes F=N(0,I_d) as the
*reference* and minimizes over the law G of \(r(\theta,X)\) — this is
indeed the *opposite* direction, and the manuscript's analysis showing
this collapses to δ_0 is correct.

LF2I parenthetical: "this is why LF2I uses pinball-style losses in the
*opposite* direction — fixing the test statistic and learning critical
values — and is not a counterexample." This is roughly correct in
spirit: LF2I (Dalmasso et al. 2024) learns critical-value functions
\(c_\alpha(\theta)\) by quantile regression of a fixed test statistic
\(T(X;\theta)\). In that setup the *target distribution* (the
distribution of T) is fixed by the data-generating process, and the
*forecast* (the quantile function being learned) varies — which is the
strictly-proper direction for pinball loss. So LF2I "saves" itself by
applying pinball loss in the standard direction (fix sample, vary
forecast) rather than the inverted direction Class 4 would correspond
to.

**Suggested fix:** (1) Either work with the dual form
\(\mathrm{CRPS}(F,y) = \mathbb{E}|Z-y|-\tfrac12\mathbb{E}|Z-Z'|\) and
note that under \(Y\sim G\), \(\mathbb{E}_{Y\sim G}\mathbb{E}_{Z\sim
F}|Z-Y|\) is the energy distance contribution which is minimized at
G=δ_E[Z] when F is symmetric — or regularize the integrals explicitly.
(2) The LF2I sentence is brief; either expand to one sentence
clarifying "LF2I applies pinball in the standard direction (sample
fixed by simulator, quantile-function forecast learned)" or omit. As
written it requires reader familiarity with LF2I.

### C-3.7-Class5
**Verdict:** ✓
**Quote:** "Apply KL on the conditional density: \(L(r) =
\mathbb{E}_{(\theta,X)}[-\log\hat p_r(X|\theta)]\) ... Strictly proper
for \(\mathcal{M}_\mathcal{F}\) by §3.2. Scales pointwise to any
dimension. Identical to the SNL loss. The only subtlety is the
architectural monotonicity-in-X requirement (§3.5)."
**Diagnosis:** Follows from C-3.2-KL and C-3.5-loss-below. Internally
consistent.

### C-3.7-positioning
**Verdict:** ✓ holds (modulo cross-Part citation correctness)
**Quote:** "NPE/SNPE ... minimize per-sample log-density of \(\theta |
X\) — a Class-5 loss on the *posterior* ... NLE/SNL ... minimize
per-sample log-density of \(X | \theta\) — Class-5 on the *likelihood*
... NRE ... targets the ratio via binary classification — a Class-5-
flavored loss on a different object ..."
**Diagnosis:** Taxonomy is internally consistent and matches the
standard SBI literature framing. NPE = density estimation of p(θ|X);
NLE/SNL = density estimation of p(X|θ); NRE = classification-based
ratio estimation. The "Class-5-flavored" hedge for NRE is appropriate
since NRE's objective (binary cross-entropy) isn't literally a
log-density of a normalized model but is per-sample and density-ratio
based. The LF2I/WALDO/Box CD positioning ("test-statistic CDFs ...
calibration target is CDF of test statistic under each θ, not CDF of
parameter under data") is correct and matches the paper's own §10.
Round 2 will verify the citation attributions; on its face the
taxonomy is sound.

## Cross-Part findings

- **[Part I, C-2.4-folding]**: This is the upstream of C-3.5-loss-below
  and provides the precise mechanism (Z(θ)>1 ⇒ loss<entropy) that §3.5
  references. §3.5 would be strengthened by a forward reference. The
  §2.4 statement itself reads slightly informally — "the loss can dip
  arbitrarily far below the true model's NF-MLE value" — without the
  \(-\log Z(\theta)\) algebra; consider adding it once in §2.4 and
  cross-referencing from §3.5.

- **[Part V, E-8.4-ablation / C-8.4-ablation-interp]**: §8.4's
  empirical observation that the ablated (autograd-Jacobian) model
  achieves loss 0.56 *below* the analytical truth's loss of 0.88 is
  the empirical realization of C-3.5-loss-below. The §3.5 → §8.4 link
  is important; the manuscript does state "the empirical demonstration
  is in §8.4" but doesn't explicitly note that the *gap* (0.56 vs
  0.88) is exactly the \(-\log Z(\theta)\) shift predicted by the
  theory. Linking the numerical gap to the theory in §3.5 or §8.4
  commentary would strengthen the theory-empirics bridge.
