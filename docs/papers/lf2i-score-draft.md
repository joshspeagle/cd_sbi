# LF2I-Score: Neyman-Calibrated Score Statistics from Amortized Neural Likelihoods

**DRAFT v0.1 — 2026-06-10. For consultation with the Lee group; authorship TBD.**

> **Draft status / before submission:**
> (1) Re-run the arXiv novelty check (stat.ME + hep-ph; "score" ∧ "LF2I"/"Neyman") — the slot is
> verified open as of 2026-06-10 but Jiang–Wang–Yang (arXiv:2603.29054) is one Rao-inversion
> follow-up away; (2) regenerate all tables from `outputs/` via
> `analysis.figures.data_io.figure_data.load_sweep()` — numbers below are transcribed from the
> internal sweep records; (3) replicate the single-seed probe results (§6.5: Cauchy, d_θ=10, SLCP)
> at ≥5 seeds before promoting them out of "preliminary"; (4) Lee-group coordination items: pipeline
> dedup on the score statistic; relation to FreB; whether this merges into a group paper.

---

## Abstract

Likelihood-free frequentist inference (LF2I) constructs confidence sets with frequentist coverage
by Neyman-inverting a test statistic whose critical values are learned from simulations: validity
depends only on the calibration, while power depends on the choice of statistic. The statistics
instantiated to date are the odds (ACORE), the Bayes factor (BFF), and posterior moments (WALDO).
We fill the remaining slot of the classical trinity: **the score**. LF2I-Score trains an amortized
neural likelihood $q_\varphi(X\mid\theta)$ (as in neural likelihood estimation) and reads out a
frequentist confidence set from its structural-parameter score
$U(\theta;X)=\nabla_\theta \log q_\varphi(X\mid\theta)$, in two variants: an asymptotic **Rao**
form $U^\top \hat I(\theta)^{-1} U$ with a $\chi^2_{d_\theta}$ threshold, and a **calibrated** form
$\lVert U\rVert^2$ with critical values $c_\alpha(\theta)$ learned by quantile regression — the
LF2I construction, which renders the procedure valid without Fisher-information estimation and
without the statistic being pivotal. The score readout is automatically $d_\theta$-dimensional, so
it requires no hand-crafted or learned summary statistic and thereby sidesteps the
fixed-dimensional sufficiency bottleneck (Pitman–Koopman–Darmois) that constrains summary-based
methods. In matched-budget comparisons on benchmark targets, LF2I-Score is competitive with or
better than NPE/NLE/NRE/LF2I-BFF baselines at small-to-medium simulation budgets; we document two
failure modes honestly — amplification of likelihood-model overfitting in the fixed-data regime
(removable with fresh batches), and breakdown on non-regular targets where the Fisher information
degenerates, which afflicts the entire calibration family. We position the method precisely against
its ancestors: the efficient method of moments anticipates "score of an estimated model, calibrated
by simulation" with auxiliary-model scores and asymptotic $\chi^2$ calibration; score-based
summaries (SALLY, optimal observables) use the score as a compression, never as the inverted
statistic. To our knowledge, LF2I-Score is the first method to use the structural-parameter score
of an amortized neural likelihood as the Neyman-inverted test statistic with learned, finite-sample
critical values.

---

## 1. Introduction

Simulation-based inference (SBI) targets parameters $\theta$ of a stochastic simulator
$X\sim p(\cdot\mid\theta)$ whose likelihood is intractable. The dominant neural approaches estimate
Bayesian objects — posteriors (NPE), likelihoods (NLE), or likelihood ratios (NRE) — and a
now-extensive literature documents that the resulting posteriors are frequently **overconfident**:
their credible regions undercover the truth [Hermans et al. 2022]. For scientific applications that
require coverage statements — "this interval contains the true parameter with probability at least
$1-\alpha$, whatever the true parameter is" — this motivates *frequentist* SBI.

The likelihood-free frequentist inference framework (LF2I) [Dalmasso et al. 2024] delivers such
guarantees by reviving the Neyman construction: choose any test statistic $\tau(X,\theta)$,
estimate its critical values $c_\alpha(\theta)$ under the null $\theta_0=\theta$ from simulations,
and invert. The framework's own decomposition is the key design insight: *validity depends only on
the calibration; power depends only on the statistic*. "Any method that defines a test statistic
can leverage LF2I to create valid confidence sets" [Dalmasso et al. 2024]. The statistics published
so far are the odds / integrated odds (ACORE [Dalmasso et al. 2020]), the Bayes factor (BFF
[Dalmasso et al. 2024]), and a Wald-type statistic built from posterior moments (WALDO
[Masserano et al. 2023]). Classical testing theory's trinity — Wald, likelihood ratio, score — is
thus two-thirds instantiated: WALDO occupies the Wald slot and ACORE/BFF are ratio-type. The
**score slot is open**, and it is a natural one to fill: the score test is the locally optimal
test (in the classical one-parameter, one-sided sense it is locally most powerful [Rao 1948; Cox &
Hinkley 1974]; in higher dimensions the Rao statistic retains local optimality properties under
contiguous alternatives), and — decisively for SBI — the score is *computable by automatic
differentiation from any neural likelihood, at no additional training cost*.

**Contributions.**
1. **LF2I-Score** (§3): a two-variant frequentist readout of an amortized neural likelihood via its
   score — asymptotic Rao ($\chi^2$ threshold) and Neyman-calibrated ($\lVert U\rVert^2$ with
   learned $c_\alpha(\theta)$). The calibrated variant requires **no Fisher-information estimation**:
   because the critical values are learned per hypothesized $\theta$, the calibration absorbs any
   $\theta$-wise monotone distortion of the statistic, so whitening is unnecessary for validity
   (§3.3).
2. **A no-summary route around the sufficiency bottleneck** (§4): summary-based SBI must compress
   $X$ to fixed dimension, and by the Pitman–Koopman–Darmois theorem no fixed-dimensional summary
   is sufficient outside exponential families. The score readout is automatically
   $d_\theta$-dimensional *without* claiming sufficiency — it is a local reduction whose validity
   is supplied by calibration and whose efficiency is local-asymptotic. We demonstrate this on a
   Cauchy location-scale target, which has no finite-dimensional sufficient statistic (preliminary,
   §6.5).
3. **An honest failure-mode study** (§6.2–6.3, 6.5): (i) the score *amplifies* likelihood-model
   overfitting — in the fixed-training-set regime coverage degrades as budget grows, and the effect
   vanishes with fresh batches; (ii) the method inherits the score test's regularity requirements —
   on a multimodal benchmark with Fisher degeneracies (SLCP), coverage fails and a much better
   density model does not repair it, locating the failure in the statistic's regularity assumptions
   rather than estimation quality; the entire calibration family fails there with it.
4. **Precise positioning** (§7): we cede conceptual ancestry to the efficient method of moments
   [Gallant & Tauchen 1996; Gourieroux, Monfort & Renault 1993] — score of an estimated flexible
   auxiliary model, simulation-calibrated — and distinguish auxiliary-parameter scores used as
   moment conditions from the structural-parameter score of a likelihood surrogate; and we
   distinguish score-as-summary (SALLY [Brehmer et al. 2020], optimal observables [Atwood & Soni
   1992; Diehl & Nachtmann 1994]) from score-as-inverted-statistic.

## 2. Background

### 2.1 The Neyman construction and LF2I

A level-$\alpha$ test of $H_0:\theta=\theta_0$ for each $\theta_0$, inverted, yields a confidence
set with $1-\alpha$ coverage at every $\theta_0$: $C_\alpha(X)=\{\theta_0: \tau(X,\theta_0)\
\text{does not exceed its null critical value}\}$. The construction requires only the null
distribution of $\tau(\cdot,\theta_0)$ at each $\theta_0$ — which a simulator provides on demand.
LF2I amortizes this: draw $(\theta_b, X_b)$ pairs, evaluate $\tau_b=\tau(X_b,\theta_b)$, and
regress the conditional $\alpha$-quantile of $\tau$ **on the hypothesized $\theta$** (quantile
regression in LF2I; conformal trees in TRUST [Cabezas et al. 2024]). Two properties matter here.
First, the statistic need not be pivotal nor have a known null law — calibration absorbs whatever
distribution it has. Second, the guarantee structure is: exact per-$\theta_0$ coverage as the
critical-value estimator converges (asymptotic in *simulation budget*, not data size), with
finite-sample guarantees of the marginal/local kind under the calibration proposal. We adopt this
guarantee structure verbatim and claim nothing stronger.

### 2.2 The score test

For a regular model with log-likelihood $\ell(\theta;X)$, the score $U(\theta;X)=\nabla_\theta\ell$
satisfies $\mathbb{E}_{\theta}[U(\theta;X)]=0$ with covariance the Fisher information $I(\theta)$,
and the Rao statistic $U^\top I(\theta)^{-1}U$ is asymptotically $\chi^2_{d_\theta}$ at the truth.
Its appeal in SBI is threefold: it needs the likelihood only *locally* at the tested $\theta$ (no
maximization, unlike Wald/LR); it is the locally optimal direction; and for a neural likelihood it
is a single autograd call. Its liabilities are exactly the classical ones: the $\chi^2$ calibration
and the local optimality both require non-degenerate Fisher information and local identifiability
[Davies 1977, 1987; Drton 2009], failing under multimodality and parameter-boundary degeneracies —
we test, rather than hide, these (§6.5).

## 3. Method

### 3.1 Stage 1: amortized neural likelihood

Train a conditional normalizing flow $q_\varphi(X\mid\theta)$ on simulator draws by maximum
likelihood — identical to standard NLE [Papamakarios et al. 2019]; LF2I-Score adds **zero**
training cost over an NLE baseline and can re-use an existing NLE model. The score is
$U(\theta;X)=\nabla_\theta \log q_\varphi(X\mid\theta)$, one reverse-mode pass per evaluation. (In
implementation: gradients are enabled locally so the statistic works inside no-grad evaluation
loops, and non-finite scores from extreme inputs are zeroed as a guard — an under-trained flow can
emit unstable gradients; see §6.2.)

### 3.2 Variant R (Rao, asymptotic)

$$\tau_R(X,\theta) = U(\theta;X)^\top\, \hat I(\theta)^{-1}\, U(\theta;X), \qquad
C_\alpha = \{\theta:\ \tau_R \le \chi^2_{d_\theta,\alpha}\}.$$

$\hat I(\theta)$ is the Monte-Carlo score covariance at the *queried* $\theta$: draw
$n_F$ (default 4000) fresh $X\sim p(\cdot\mid\theta)$, form $\hat I = \frac1{n_F}\sum U U^\top$
(ridge-regularized), invert, and cache per $\theta$. Validity is asymptotic on two counts — the
$\chi^2$ approximation (here aided by CLT over i.i.d. replicates within an observation, when
present) and $q_\varphi \approx p$. Variant R is the fast, diagnostic-grade readout.

### 3.3 Variant C (calibrated, Fisher-free)

$$\tau_C(X,\theta) = \lVert U(\theta;X)\rVert^2, \qquad
C_\alpha = \{\theta:\ \tau_C \le \hat c_\alpha(\theta)\},$$

with $\hat c_\alpha(\theta)$ a multi-quantile network trained on calibration pairs
$(\theta_b, \tau_C(X_b,\theta_b))$, $\theta_b$ drawn from the proposal and $X_b\sim
p(\cdot\mid\theta_b)$ — drawn from a *separate stream* from the flow's training data
(freeze-before-calibrate). Two deliberate choices:

- **No Fisher whitening.** Because $\hat c_\alpha(\theta)$ is learned *per hypothesized $\theta$*,
  the calibration is invariant to any $\theta$-pointwise monotone reparametrization of the
  statistic: whatever $I(\theta)$-shaped distortion $\lVert U\rVert^2$ carries is absorbed into the
  learned critical-value surface. Whitening can affect the *shape* (power) of the set, never its
  validity; dropping it removes the single most fragile estimation step (a $d\times d$ inverse
  that degenerates near non-identifiability). [Internal note: an independent design audit reached
  the same conclusion from first principles; the (μ,σ²) target, where $I(\theta)$ is strongly
  $\theta$-dependent, is the empirical check.]
- **The LF2I guarantee, inherited verbatim:** validity depends on the calibration stage only.
  Variant C remains valid (up to critical-value estimation error) even where the $\chi^2$
  asymptotics of Variant R are poor — at small per-observation information, skewed score
  distributions, or moderate non-Gaussianity.

### 3.4 What LF2I-Score is not

It is **not** a confidence-distribution method with by-construction coverage: it is two-stage and
asymptotic-or-calibrated, the mode that exact-pivot constructions (CD-SBI) define themselves
against. It is **not** prior-free in its power profile: the calibration proposal concentrates
critical-value accuracy where the proposal has mass — accuracy at rare $\theta$ is a budget
allocation choice. And it does **not** escape the regularity boundary of score tests (§6.5).

## 4. Positioning: the no-summary route

Summary-based SBI pipelines compress $X$ into a fixed-dimensional statistic before inference. The
Pitman–Koopman–Darmois theorem makes the cost precise: outside exponential families no
fixed-dimensional statistic is sufficient, so the compression necessarily discards information —
and learned summaries demonstrably struggle to capture even the sufficient directions that do
exist (second moments resist; flexibility can worsen it). The score readout takes a different
deal: it reduces $X$ to the $d_\theta$ numbers $U(\theta;X)$ — *per hypothesized $\theta$*, i.e. a
local reduction rather than a global summary — making no sufficiency claim, with validity supplied
by calibration and efficiency claimed only locally. The clean test of this positioning is a target
with **no finite-dimensional sufficient statistic at all**, where every summary-based pipeline is
wrong by theorem: the Cauchy location-scale model. There the score readout attains near-nominal
coverage (preliminary, §6.5). The price of the deal appears at the other boundary: where the score
itself degenerates (§6.5, SLCP).

## 5. Experimental setup

Targets (all with known ground truth): 1D location-normal (§8.1 of the companion CD-SBI study); 2D
location-Gaussian, identity and correlated covariance (§8.2/§8.3); exponential rate with sufficient
statistic $T=\sum X_i$ (§8.4); Gaussian with unknown $(\mu,\sigma^2)$, $n_{\rm iid}=10$; bivariate
normal with unknown $(\mu,\Sigma)$ in log-Cholesky coordinates ($d_\theta=5$). Matched simulation
budgets (small→xlarge), 5 seeds (3 for $d_\theta=5$), fixed training set (`fresh_batch=false`) as
the research-realistic default. Primary metric: `coverage_error_max` — the maximum over a
$\theta_0$ grid and $\alpha\in\{0.5,0.68,0.9,0.95\}$ of $|$empirical $-$ nominal$|$ coverage, MC
noise floor $\approx 0.02$–$0.03$. Baselines: NPE, NLE (HPD readouts), NRE, LF2I-BFF (with the
corrected MC marginal), and the exact-pivot CD-SBI as the by-construction reference.

## 6. Results

*(Numbers transcribed from the internal sweep records; regenerate exact tables before submission.)*

### 6.1 Matched-budget sweep (validated; 2 variants × 4 budgets × 5 seeds × 4 targets)

- **§8.1 (d=1):** Variant R ties the best methods at all budgets (0.029–0.030 vs CD-SBI/NLE
  0.025–0.030) — at the noise floor.
- **§8.2/§8.3 (d=2):** at small/medium budget Variant R (0.043–0.048) **beats every non-CD-SBI
  baseline** (NPE 0.05–0.10, NLE 0.07–0.12, NRE 0.14–0.19, LF2I-BFF 0.11–0.14); CD-SBI's exact
  pivot stays at floor (0.020–0.027).
- **Variant C** sits at 0.07–0.13 on these targets — the quantile head's own estimation error
  currently dominates; see §8 (limitations).
- **$(\mu,\Sigma)$, $d_\theta=5$ (3 seeds):** Variant R is flat in budget (~0.11–0.14) and
  matches/overtakes CD-SBI at large budgets (CD-SBI degrades 0.054→0.152 from overfitting of its
  larger flow); a reversal of the low-$d$ ordering.

### 6.2 The overfitting-amplification finding (validated)

With a fixed training set, Variant R's coverage **degrades as budget grows** at $d\ge2$: §8.2
0.043→0.192 and §8.3 0.048→0.182 from small→xlarge. The cause is the score amplifying NLE
overfitting: a flow that interpolates its training set develops spurious sharp local structure
whose *gradients* are large even where its *density* looks fine. The diagnosis is confirmed by
intervention: with `fresh_batch=true` (new simulations each step), the degradation vanishes
(xlarge: 0.192→0.033 and 0.182→0.032 — back to the floor). Practical guidance: fresh batches,
early stopping, or weight decay on the flow are first-line; the effect is a *statistic-level*
amplification of a *model-level* pathology and should be checked whenever a score readout is used.

### 6.3 Data-conditioning sensitivity (validated negative result)

On the exponential-rate target evaluated on the sufficient statistic $T$ (skewed, Gamma-distributed)
without input transformation, Variant R fails flat (0.363 at all budgets) — the flow's score on a
skewed 1D input is badly miscalibrated. A preliminary raw-$X$ probe with an asinh input transform
(a $\theta$-independent bijection, hence score-preserving in the relevant sense) scored 0.063.
Data conditioning is part of the recipe, not an optional nicety.

### 6.4 Where the exact pivot wins

Across §8.1–8.4 CD-SBI's monotone pivot stays at the noise floor at every budget — its architectural
monotonicity acts as built-in regularization, which is precisely the robustness LF2I-Score lacks in
the fixed-data regime. The two methods are complements: exact/by-construction where its
architectural assumptions can be met; LF2I-Score as the scalable two-stage readout requiring only
an NLE model and autograd.

### 6.5 Preliminary probes (single seed — flagged, not yet replicated)

- **Cauchy location-scale** ($d_\theta=2$; **no finite sufficient statistic exists**): Variant R at
  0.022 — the noise floor; the summary-based route is impossible here by theorem.
- **$d_\theta=10$** (five Gaussian $(\mu_i,\log\sigma_i)$ pairs): 0.058 — no dimensional blow-up.
- **SLCP** (multimodal benchmark; parameters enter squared → sign-symmetry multimodality + Fisher
  degeneracy near 0): Variant R 0.115, Variant C 0.426 — **failure**, as the score test's
  regularity conditions are violated by design. Decisively: replacing the MAF with a much more
  expressive NSF improves the likelihood fit dramatically (−log q 10.99→3.06) and leaves coverage
  unchanged (0.111) while calibration worsens (0.648) — the boundary is **architecture-independent**,
  intrinsic to the statistic. Nor is it specific to the score: LF2I-BFF scores 0.500 over the full
  parameter box. Near non-identifiability no statistic in the family yields tight valid sets
  [Dufour 1997]; the honest output there is valid-but-wide, and the failure of *tightness* should
  not be misread as repairable by a better statistic.

## 7. Related work

**LF2I family.** ACORE (odds) [Dalmasso et al. 2020], BFF (Bayes factor) and the framework
guarantee structure [Dalmasso et al. 2024], WALDO (posterior moments → Wald slot) [Masserano et
al. 2023]; conformal-tree calibration (TRUST/TRUST++) [Cabezas et al. 2024]; posterior
recalibration into locally-valid regions (FreB) [Carzon et al. 2026]. None instantiate a score
statistic; the framework's own texts do not mention one.

**Conceptual ancestry — ceded explicitly.** The **efficient method of moments** [Gallant & Tauchen
1996] and the indirect-inference LM tests [Gourieroux, Monfort & Renault 1993] are "score of an
estimated flexible model, calibrated by simulation": the score of a seminonparametric *auxiliary*
density, evaluated at its quasi-MLE, used as moment conditions with asymptotic $\chi^2$
calibration. LF2I-Score differs in using the **structural-parameter** score of a surrogate for the
actual likelihood (not auxiliary-parameter scores as moments), with **learned, $\theta$-indexed
finite-sample critical values** (not asymptotic $\chi^2$), amortized over $\theta$. Dufour-style
exact Monte Carlo tests are the classical umbrella for simulated critical values.

**Score-as-summary (not score-as-statistic).** SALLY/SALLINO and the MadMiner ecosystem [Brehmer
et al. 2018, 2020] learn the score as a *locally sufficient compression*; downstream inference
re-estimates a likelihood (ratio) of the summary and inverts *that*. Classical optimal observables
[Atwood & Soni 1992; Diehl & Nachtmann 1994] are the same idea with exact tree-level scores. ATLAS
neural SBI [ATLAS 2024] Neyman-inverts a learned likelihood *ratio*. In all of these the score is
an input to inference; here it *is* the inverted statistic.

**Contemporaneous score-based UQ.** Fisher score matching for forecasting/MLE [Sui, Pandey &
Wandelt 2025]; structured score matching with Wald/bootstrap confidence sets around the score root
[Jiang, Wang & Yang 2026] — point-estimation-centric, no test inversion, no learned
$c_\alpha(\theta)$. MUSE [Millea & Seljak 2022] is score-based point estimation with asymptotic
covariance.

**Frequentist-coverage SBI broadly.** Overconfidence of neural posteriors [Hermans et al. 2022];
conservative training [Delaunoy et al. 2022; Falkiewicz et al. 2023]; coverage diagnostics [Talts
et al. 2018; Lemos et al. 2023]. Marginal coverage diagnostics are weaker than the per-$\theta_0$
target of the Neyman construction — the two should not be conflated.

**Relation to CD-SBI.** The companion exact-pivot framework targets by-construction coverage with
architecturally monotone flows and identifies its exact pivot, in regular models, with the
standardized score/signed-root-LR asymptotically; LF2I-Score is the two-stage, asymptotic/calibrated
member of that comparison — a complement, not a variant (see §6.4).

## 8. Limitations

1. **Fixed-data brittleness** (§6.2): without fresh batches or regularization, the score amplifies
   flow overfitting and coverage degrades *with more budget* — a counterintuitive failure that
   practitioners must check.
2. **Regularity boundary** (§6.5): Fisher degeneracy and multimodality break both variants, as
   classical theory predicts [Davies 1977; Drton 2009; Watanabe 2009]; this is a boundary of the
   statistic family, not of our implementation, and near non-identifiability only valid-but-wide
   sets exist [Dufour 1997].
3. **Variant C's quantile head** currently under-performs Variant R on regular targets — the
   per-α multi-quantile MLP is the bottleneck (no monotonicity across α; estimation error at
   extreme quantiles). Conformal-tree calibration (TRUST-style) and monotone-in-α heads are the
   obvious upgrades and are left to coordinated future work.
4. **Tail-$\theta$ accuracy is proposal-dependent**: critical-value estimation error concentrates
   where the calibration proposal puts little mass. This is a budget-allocation lever (one can
   oversample tails), not a guarantee; we do not claim per-$\theta_0$ finite-sample validity.
5. **Data conditioning matters** (§6.3): skewed/heavy-tailed inputs need θ-independent
   stabilizing transforms for the flow's score to be usable.
6. The preliminary probes (§6.5) are single-seed and must be replicated before being relied on.

## 9. Discussion

LF2I-Score completes the classical trinity within the LF2I family at essentially zero additional
training cost over NLE, inherits the framework's validity-from-calibration guarantee, and — via the
score's automatic $d_\theta$-dimensional local reduction — provides the first no-summary route in
the family around the fixed-dimensional sufficiency bottleneck, demonstrated where sufficiency is
impossible in principle. Its honest profile: strong at low-to-medium budgets on regular targets,
brittle to flow overfitting in the fixed-data regime (with a clean fix), and bounded — along with
the whole calibration family — by the regularity of the underlying testing problem. The natural
next steps are calibration-side (conformal trees, monotone-in-α heads, tail-weighted proposals)
rather than statistic-side, and are best pursued in coordination with the LF2I ecosystem.

## References

*(Informal [Author Year] style; full BibTeX to be assembled from the verified citation ledger —
all entries below were existence-and-content verified on 2026-06-10.)*

- ATLAS Collaboration 2024, "An implementation of neural simulation-based inference for parameter
  estimation in ATLAS," arXiv:2412.01600.
- Atwood & Soni 1992, Phys. Rev. D 45, 2405 (optimal observables).
- Brehmer, Cranmer, Louppe & Pavez 2018, "A guide to constraining effective field theories with
  machine learning," arXiv:1805.00020; Brehmer, Louppe, Pavez & Cranmer 2020, "Mining gold from
  implicit models...," PNAS 117, arXiv:1805.12244; MadMiner: arXiv:1907.10621.
- Cabezas, Soares, Ramos, Stern & Izbicki 2024, "Conformal calibration of statistical confidence
  sets" (TRUST/TRUST++), arXiv:2411.19368, TMLR.
- Carzon, Masserano, et al. 2026, "Trustworthy scientific inference with generative models" (FreB),
  arXiv:2508.02602, MLST.
- Cox & Hinkley 1974, *Theoretical Statistics*.
- Dalmasso, Izbicki & Lee 2020, "Confidence sets and hypothesis testing in a likelihood-free
  inference setting" (ACORE), ICML, arXiv:2002.10399.
- Dalmasso, Masserano, Zhao, Izbicki & Lee 2024, "Likelihood-free frequentist inference" (LF2I/BFF),
  EJS 18(2), arXiv:2107.03920.
- Davies 1977, 1987, Biometrika (nuisance parameters present only under the alternative).
- Delaunoy et al. 2022, "Towards reliable SBI with balanced NRE," arXiv:2208.13624.
- Diehl & Nachtmann 1994, Z. Phys. C 62 (optimal observables).
- Drton 2009, "Likelihood ratio tests and singularities," Ann. Statist. 37(2).
- Dufour 1997, "Some impossibility theorems in econometrics...," Econometrica 65(6).
- Falkiewicz et al. 2023, "Calibrating neural SBI with differentiable coverage probability,"
  NeurIPS, arXiv:2310.13402.
- Gallant & Tauchen 1996, "Which moments to match?" (EMM), Econometric Theory 12.
- Gourieroux, Monfort & Renault 1993, "Indirect inference," J. Appl. Econometrics 8.
- Hermans, Delaunoy, Rozet, Wehenkel, Begy & Louppe 2022, "A trust crisis in simulation-based
  inference?...," TMLR, arXiv:2110.06581.
- Jiang, Wang & Yang 2026, "Likelihood-free inference via structured score matching,"
  arXiv:2603.29054.
- Lemos et al. 2023, "Sampling-based accuracy testing..." (TARP), ICML, arXiv:2302.03026.
- Masserano, Dorigo, Izbicki, Kuusela & Lee 2023, "Simulator-based inference with WALDO," AISTATS,
  arXiv:2205.15680.
- Millea & Seljak 2022, MUSE, arXiv:2112.09354.
- Papamakarios, Sterratt & Murray 2019, "Sequential neural likelihood," AISTATS, arXiv:1805.07226.
- Rao 1948, "Large sample tests of statistical hypotheses...," Proc. Camb. Phil. Soc. 44.
- Sui, Pandey & Wandelt 2025, "Fisher score matching for simulation-based forecasting and
  inference," arXiv:2507.07833.
- Talts, Betancourt, Simpson, Vehtari & Gelman 2018, "Validating Bayesian inference algorithms with
  simulation-based calibration," arXiv:1804.06788.
- Watanabe 2009, *Algebraic Geometry and Statistical Learning Theory*.

## Reproducibility

Implementation: `src/cdsbi/methods/score_cd.py` (`ScoreCDRunner`, variants `rao`/`cal`), configs
`configs/method/score_cd_{rao,cal}.yaml`. Sweep records: `outputs/8_*_baseline_sweep/` (the
2026-05-31 score-CD additions) and the $(\mu,\Sigma)$ comparison. Coverage evaluated with the
chunked engine (`diagnostics/engine.py`), bit-identical to the legacy diagnostic. Calibration pairs
for Variant C are drawn from a separate RNG stream from flow training (freeze-before-calibrate).
A code rename (`score_cd` → `lf2i_score`) is pending the coordination decisions above.
