# Round 3 Final Review — Astronomer Reader

## Summary

The manuscript is, on the whole, surprisingly readable for the audience it
targets. The high-level architecture of the argument is genuinely clear:
I came away understanding what a confidence distribution is, why the
authors are not estimating a posterior, what a pivot does for them,
what NF-MLE is, and roughly how a triangular autoregressive flow works.
The opening (Abstract through §2) and Part V (experiments) land
particularly well. The biggest residual obstacle is not vocabulary —
the glossary-style insertions are doing real work — but rather the
density of formal-statistics machinery in Parts III–IV: monotone
rearrangement, Lemma 4.2's pushforward-density blow-up argument,
the (R3)/(R3_U)/(R4) zoo around the discrete case, the
$\sigma$-algebra-equivalence step in the multivariate proof, and the
"strict propriety" abstraction. These are technically correct and
mostly motivated, but several land as walls of definitions or as
machinery whose role is only fully clear in retrospect. A secondary
issue is some genuinely oracle-style moves in §3.7 (Class 4 collapse
argument), the (R4) Bin(3, θ) counterexample, and parts of Theorem A-d's
inductive proof.

## Where I got stuck (terms and concepts under-explained)

- **§2.2 "calibration manifold" (the term itself).** The object is
  defined cleanly, but the word *manifold* never gets justified. As an
  astronomer I expect "manifold" to mean a smooth $k$-dimensional thing
  living inside something bigger. Here it's "the set of $r$'s satisfying
  (C1)," which could be any subset. A one-line aside ("we use 'manifold'
  loosely — it is just a set of functions; the geometric word reflects
  that in regular cases it carries a natural parameterization") would
  help. The same word later carries weight in §3.3 ("the manifold is
  infinite-dimensional"), and I wasn't sure whether that statement was
  literal or rhetorical.

- **§2.4 / §3.5 "(R2) folding" — why $|\det \partial r/\partial X|$
  blows up where $r$ folds.** The text says "the regions of $X$-space
  where $r$ folds are precisely where $|\det \partial r/\partial X|$ is
  large." For me this is the heart of the failure mode, but the
  intuition is given in a parenthetical. A quick concrete 1D picture
  ("imagine $r$ being a parabola in $X$: at the fold the inverse has two
  branches, and around the critical point $\partial_X r$ is small, so
  $1/|\partial_X r|$ — the pushforward density factor — blows up, which
  autograd reads as 'good fit'") would lock this in.

- **§3.2 "strict propriety."** The definition is buried after the
  theorem statement and proof. By the time I reached the formal
  definition I had been told twice that NF-MLE "is strictly proper" and
  was supposed to know what that meant. Moving the one-sentence
  scoring-rule definition before the theorem (or even before the
  paragraph "What we are about to prove") would solve this.

- **§3.7 Class 4 (CRPS collapse).** This is the most oracle-style moment
  in the manuscript (see also next section). The conclusion is asserted
  and *then* derived via two parallel calculations, neither of which I
  could follow without re-reading. In particular, "we are doing the
  opposite — we have fixed the forecast and are varying what the data
  look like" is a sentence that should be unpacked with a tiny diagram
  or table. A reader who has just been introduced to scoring rules a
  paragraph earlier is unlikely to internalize the direction-reversal
  argument the first time through.

- **§4.2 Lemma 4.2 proof (the order-$2m$ critical-point argument).**
  This is where the manuscript becomes a graduate analysis paper. The
  sentence "Let $2m \ge 2$ be the order of the leading nonzero
  derivative of $\psi$ at $X_*$" arrives without warning and the
  fractional-power scaling argument that follows ($|\psi'| \sim
  |v-v_*|^{(2m-1)/(2m)}$) is a textbook computation but not one I do in
  my head. A worked $m=1$ case ("if $\psi$ has a parabolic maximum at
  $X_*$ so $\psi(X) - v_* \approx -c(X-X_*)^2$, then near $X_*$ we have
  $|X-X_*| \sim \sqrt{|v-v_*|}$, hence $|\psi'(X)| \sim \sqrt{|v-v_*|}$,
  hence the pushforward density picks up a $|v-v_*|^{-1/2}$ factor")
  would make the higher-order extension obvious by analogy.

- **§5.7.2 the (R4) Bin(3, θ) counterexample.** The construction
  $V'_\theta$ with the (T=0, T=2, T=1, T=3) tiling appears with no
  motivation: why this permutation and not any other? Why are we
  expecting one to evade (R1)? The "algebraic accident" framing is fine
  in hindsight but doesn't help on first read. Also: I followed the
  individual verifications but never assembled a picture of what
  $V'_\theta$ "looks like" relative to $V^*_\theta$ — a tiny figure
  showing the two tilings of $[0,1]$ side by side would make the whole
  subsection click.

- **§6.3 the $\sigma$-algebra-equivalence step.** "$\sigma(\theta, X_{<k})
  = \sigma(\theta, r_{<k})$" is one line and the conclusion "$r_k \mid
  \theta, X_{<k} \sim \mathcal{N}(0,1)$" is another. For a reader who
  hasn't seen $\sigma$-algebras used as conditioning bookkeeping in this
  way, the move is opaque. I'd suggest a one-sentence translation:
  "Since the previous coordinates $r_{<k}$ are a deterministic bijection
  of $X_{<k}$ at fixed $\theta$, conditioning on either gives the same
  conditional distribution."

- **"Brenier map" (§6.3).** Mentioned in the optimal-transport aside,
  defined as "the unique transport map that is the gradient of a convex
  function." For an astronomer this definition is not actionable — I
  know what a gradient is and what convexity is, but not why those two
  conditions pin a transport map down. A one-liner ("the Brenier map is
  the rotation-invariant $L^2$-optimal version; we don't use it but
  mention it as the natural alternative to KR") is probably sufficient
  given that the manuscript never actually uses Brenier maps.

- **§6.3 Remark 2 ("gauge orbit" of additive flow).** The claim that
  $a_k \mapsto a_k + h$, $b_k \mapsto b_k + h$ leaves $r_k$ unchanged is
  fine, but "the trained $r$ is a finite-network approximation… with
  residual variance scaling like $N^{-1}$" is asserted without support.
  As an astronomer I'd accept the scaling on faith, but I'd want a hint
  at *why* it's $N^{-1}$ and not $N^{-1/2}$ (parametric rate vs. KS-test
  scale).

- **§7.3 "Kolmogorov limiting distribution" / "noise floor."** The
  formula $\Pr(\sqrt{N} \cdot \mathrm{KS} > 1.628) = 0.01$ arrives without
  a sentence explaining where 1.628 comes from. Any astronomer who has
  done a KS test will follow along, but the constant feels mystery-meat
  on first read. A one-line "1.628 is the 0.99 quantile of the
  Kolmogorov distribution, tabulated in any reference on KS tests"
  removes the mystery.

- **§10 "signed-root log-likelihood ratio."** Introduced in one sentence
  at the end of §10 and immediately abandoned. For an astronomer this is
  a phrase that sounds important but cannot be acted on without a
  reference and a worked example. Either expand or remove.

## Where the writing was oracle-style (assertion without motivation)

- **§3.7 Class 4 / CRPS collapse.** "The CRPS-against-$\mathcal{N}(0,1)$
  loss is therefore *minimized by collapsing $r(\theta, X)$ to a
  constant at the median*, the opposite of calibration." This is the
  most striking conclusion in the design-space taxonomy, and the
  manuscript essentially says "we will sketch the calculation," does
  some algebra with the kernel-score form, and arrives at the
  conclusion. The reader is asked to take on faith that "$L^1$
  projection of any distribution onto a point mass collapses to the
  median of the reference." I believe this is true, but the manuscript
  doesn't prove it, and an astronomer reader has no reason to know it.
  A 2-line proof or a citation pointing at the exact result would
  convert this from oracle-style to readable.

- **§5.7.1 / §5.7.2 the randomized PIT setup.** The intuition
  paragraph ("spread each atom of $T(X)$ across an interval of length
  equal to that atom's probability, using an independent uniform $U$")
  is good. But the immediate jump to "Introduce $U \sim U(0,1)$
  independent of $X$" and the formula for $V_\theta(X, U)$ is a leap.
  The "Fact" — that $V_{\theta_0}(X, U) \sim U(0,1)$ exactly — is just
  asserted. For an astronomer this is precisely the kind of moment
  where I want either a one-paragraph proof (it's a 3-line check) or an
  example: take $T \sim$ Bernoulli($\theta$), write down $V_\theta(X, U)$
  in cases $T=0$ and $T=1$, observe that the joint
  distribution is uniform on $[0, 1-\theta] \cup [1-\theta, 1] = [0,1]$.

- **§6.3 Theorem A-d proof, "Calibration $\Rightarrow$ conditional
  Gaussianity."** The step "The full calibration constraint $r(\theta;
  X) \mid \theta \sim \mathcal{N}(0, I_d)$ implies that all components
  are jointly standard normal. Combined with the $\sigma$-algebra
  identity, $r_k \mid \theta, X_{<k} \sim \mathcal{N}(0, 1)$." is
  oracle-style. *Why* does jointly $\mathcal{N}(0, I_d)$ plus
  $\sigma(\theta, X_{<k}) = \sigma(\theta, r_{<k})$ give conditional
  $\mathcal{N}(0,1)$? I had to work it out: because the joint is product
  Gaussian, $r_k$ is independent of $r_{<k}$ given $\theta$, hence the
  conditional law of $r_k$ given $\theta, r_{<k}$ is unconditionally
  $\mathcal{N}(0,1)$, and by the $\sigma$-algebra identity this equals
  the law given $\theta, X_{<k}$. That's not hard, but it's two steps,
  and the manuscript collapses them to "Combined with."

- **§8.4 Step 4 mechanism, "$Z(\theta) \approx e^{0.32} \approx 1.38$
  on average."** The chain is: empirical loss 0.56, truth loss 0.88,
  gap 0.32, exponentiate to get 1.38 inflation in $Z$. This is a
  beautiful diagnostic in retrospect, but the connection "$0.32$ = avg
  $\log Z$, $\exp(0.32) = 1.38$ = avg $Z$ (modulo Jensen, ignored)" is
  not spelled out. An astronomer reader who hasn't internalized the
  $-\log Z$ shift from §3.5 will see the arithmetic but not the logic.

- **§4.3 selecting the increasing vs. decreasing branch.** "For any
  fixed $X$, such a mixture would make $\theta \mapsto r(\theta, X) =
  \pm(\theta - X) + (\mathrm{const})$ decreasing on $A$ and increasing
  on $A^c$, violating the global strict monotonicity required by
  (R1)." This is fine for a careful reader but moves quickly: the
  notation $r(\theta, X) = \pm(\theta - X) + \mathrm{const}$ is doing
  a lot of work (one $X$-dependent constant on each branch, joined
  measurably across $A$). I'd like a sentence saying "explicitly, on
  the increasing branch $r = \theta - X$ and on the decreasing branch
  $r = -\theta + X$, so the $\theta$-derivative flips sign across $A$."

- **§11.4 misspecification / forward-KL projection.** "The
  $\mathrm{argmin}_{r \in \mathcal{F}} \mathbb{E}_\rho[\mathrm{KL}(p(\cdot
  \mid \theta) \,\|\, \hat p_r(\cdot \mid \theta))]$ direction, called
  the $M$-projection in the modern Bregman-geometry sense" is
  vocabulary-dense for a paragraph that is otherwise meant to be a
  pointer to open work. An astronomer wants either "this is what NF-MLE
  converges to when the simulator is wrong, by the same KL argument as
  §3.2, but with $p$ replaced by the true data distribution" or for the
  parenthetical to be cut.

## What landed well (preserve these in any future edits)

- **§1.2 "Astronomer-friendly comparison" and the location-normal
  worked example.** This is precisely the bridge an astronomer needs:
  posterior vs CD framed as a 4-line bullet contrast, then immediately
  the simplest possible model with $r^*(\theta, X) = \theta - X$
  derived in three lines. The PIT explanation right before it is also
  well-pitched — short, complete, no jargon.

- **§1.3 "The proposal $\rho$ is not a prior."** This is the single
  most important conceptual stumble for a Bayesian-trained reader, and
  the manuscript explicitly calls it out, gives the technical reason
  (no averaging over $\theta$ in C1), and ends with a memorable
  one-liner ("you are reading the wrong column of the training
  matrix"). Excellent.

- **§1.5 Roadmap table.** Genuinely useful as a navigation aid. I
  referred back to it twice while reading later sections.

- **§2.2 "Regularity conditions at a glance" table.** With seven
  conditions accumulating across the paper, this lookup table is what
  prevented me from getting completely lost. Keep this.

- **§3.1 "What is a normalizing flow?"** The explanation of $\phi_d(r)
  \cdot |\det \partial r/\partial X|$ as "how dense is the base
  distribution at the image" times "local volume rescaling" is exactly
  the right level of intuition. The follow-up box-equation for the
  NF-MLE loss, with the two terms interpreted, is also clean.

- **§3.4 "NF-MLE is the SNL loss" + comparison table.** A reader
  coming from SBI will immediately want to know how this relates to
  SNL. Two sentences and a 5-row table dispose of the question. This
  is the right way to position a new method against a well-known one.

- **§4.1 "What we are about to prove" paragraph.** The roadmap
  sentence "show that any candidate $r$ is a strictly monotone function
  pushing $\mathcal{N}(\theta_0, 1)$ to $\mathcal{N}(0,1)$, then use
  rigidity of 1D monotone rearrangements to pin it down" gave me a
  hook to hang the three lemmas on. The Theorem A proof structure
  (Step 1 calibration, Step 2 monotonicity, Step 3 form) is also a
  clean pedagogical scaffolding.

- **§6 the "From 1D to multivariate" intro paragraph.** "$r \sim
  \mathcal{N}(0, I_d)$ is invariant under the entire orthogonal group
  $O(d)$" — the manuscript flags up front that the proof strategy has
  to break in $d > 1$ and why. The motivation for autoregressive
  triangular flows is then "we restrict the function class so $O(d)$
  doesn't act on it." This is the right pedagogy: state the obstruction,
  then describe the architectural fix.

- **§6.3 the worked Gaussian example with explicit $\Sigma$, $L$,
  $L^{-1}$ matrices.** Going from the abstract KR statement to "$r_1 =
  \theta_1 - X_1$, $r_2 = 1.155(\theta_2 - X_2) - 0.577(\theta_1 -
  X_1)$" in three lines is what makes Theorem A-d feel real.

- **§7.1 "The idea, in one sentence" for UMNN.** "Take any strictly
  positive function on $\mathbb{R}$, integrate it from a reference
  point to $z$, and the result is a strictly monotone-increasing
  function of $z$." Perfect one-sentence definition. The follow-up
  formula with the bias term is the right next paragraph.

- **§8 the experiment tables and the §8.3 commentary on "why pivot
  RMSE looks bad but isn't."** As an astronomer I'd expect to lean on
  the experiments to sanity-check the theory, and §8.3 explicitly
  addresses the most natural objection (RMSE 0.27 looks huge) by
  pointing out it's 27% of $r_2$'s std, then arguing the right
  multivariate diagnostic is the recovered Jacobian. This is exactly
  the diagnostic literacy I would want.

- **§8.4 the four-step ablation narrative.** "Loss 0.56 is below the
  entropy floor 0.88, therefore the model isn't a valid density,
  therefore (R2) must be architectural" is a tight, falsifiable, and
  memorable argument. Best individual subsection in the paper.

- **§9 the practitioner checklist + "if the truth looks like X, use
  architecture Y" table.** This is what I would actually use if I
  tried to implement the method. Keep it.

- **§10 the comparison table and the two-bullet "what CD-SBI buys you
  vs LF2I/WALDO and vs SNL."** Tight, fair, and answers the question
  "should I use this instead of what I'm already using."

## Specific recommendations (priority-ranked)

1. **Add a worked example for the randomized PIT (§5.7.1).** Take
   $T \sim$ Bernoulli($\theta$), write down $V_\theta(X, U)$ in the two
   cases, draw the partition of $[0,1]$. This single example would do
   more for the discrete-case section than the rest of the prose.
   Currently the formula appears with no motivation and the "Fact" is
   just asserted.

2. **Add a one-paragraph proof or citation for the Class-4 CRPS
   collapse in §3.7.** The "$L^1$ projection of any distribution onto a
   point mass collapses to the median of the reference" claim is the
   load-bearing fact and is currently asserted. Either prove it inline
   (it's two lines) or cite a textbook location.

3. **Add a worked $m=1$ case to Lemma 4.2.** The fractional-power
   scaling argument is opaque on first read; a quadratic-critical-point
   case derived explicitly would make the general statement obvious by
   analogy. This is the single hardest paragraph in the manuscript for
   the target audience.

4. **Add a small figure (or even an ASCII sketch) of the canonical vs
   non-canonical Bin(3, θ) tilings in §5.7.2.** The whole counterexample
   is about which way the four atoms of $T$ are stacked across $[0,1]$.
   A picture would replace half a page of algebra in the reader's
   memory.

5. **Move the "strict propriety" definition (§3.2, post-theorem) to
   *before* Theorem 3.2.** As written, the reader is told the result
   uses strict propriety, then proven the result, then told what strict
   propriety means. Reorder.

6. **Translate the $\sigma$-algebra-equivalence step in the Theorem A-d
   proof into one sentence of plain language.** "Since the previous
   coordinates $r_{<k}$ are a deterministic bijection of $X_{<k}$ at
   fixed $\theta$, conditioning on either gives the same conditional
   distribution." Then keep the formal $\sigma(\theta, X_{<k}) =
   \sigma(\theta, r_{<k})$ as the technical statement.

7. **Add one sentence on why $r_k$ is independent of $r_{<k}$ given
   $\theta$ in the Theorem A-d "Calibration $\Rightarrow$ conditional
   Gaussianity" step.** Right now the conditional Gaussianity is
   asserted "by combining" the joint constraint with the
   $\sigma$-algebra identity, with no mention of the joint
   independence step that's actually doing the work.

8. **Reduce, or label as terminology-only, the "manifold" usage in
   §2.2.** Either commit to a brief geometric justification ("under
   regular models the set carries a smooth parameterization, hence the
   name") or replace with "calibration set" — astronomers will read
   "manifold" as making a geometric promise.

9. **Cut or expand the "signed-root log-likelihood ratio" sentence at
   the end of §10.** As a standalone reference it's not actionable;
   either give a worked example tying it back to $r^*$, or drop it.

10. **Add a one-line gloss on the Kolmogorov constant 1.628 in
    §7.3.** "(0.99 quantile of the Kolmogorov limiting distribution;
    see standard KS-test references)" is enough.
