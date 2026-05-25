# Round 1 — Part VI (Practice & position) Review

## Summary
- Total claims vetted: 9
- ✓ holds: 4  |  ⚠ under-justified: 4  |  ✗ error: 1
- Cross-Part findings: 3

## Per-claim findings

### C-9-choose-arch
**Verdict:** ⚠ under-justified
**Quote (from .tex):** "Additive: \(r^* \approx a(\theta) - b(X)\) → Additive flow ... / Multivariate with cross-coupling → Triangular flow, additive form ... / Has multiplicative \(\theta\)–\(X\) interaction → Doubly-monotone construction ... / No prior knowledge → Doubly-monotone construction with autoregressive conditioning" (lines 1700–1707).
**Diagnosis:** The first three rows are essentially restatements of the experimental setups in §8.1–8.4 (location-normal → additive; correlated 2D → triangular-additive; multiplicative \(\theta T\) → doubly-monotone), so they are internally consistent. The fourth row ("no prior knowledge → doubly-monotone with autoregressive conditioning") is not directly supported by any experiment: §8.4 used the doubly-monotone form *with* the sufficient statistic \(T\) wired in (i.e., prior knowledge), and no experiment validates that autoregressive doubly-monotone is a safe default in the *absence* of structural insight. It is also somewhat at odds with §11.5, which acknowledges that scaling autoregressive conditioning to higher \(d\) is non-trivial. The recommendation is reasonable engineering judgment but is over-prescribed as if it followed from the empirical record.
**Suggested fix:** Re-cast the last row as a "default when structure is unknown — supported by §6.1 form-2 architecture but not experimentally tested here" or move it into §11.5 as an open scaling question. Alternatively, soften "No prior knowledge" to "Mild prior knowledge" and add a footnote that the autoregressive doubly-monotone default is informed conjecture, not validated in §8.

### C-9-arch-not-penalty
**Verdict:** ⚠ under-justified
**Quote (from .tex):** "Never rely on autograd to give you a valid Jacobian without architectural support. The §8.4 ablation is documentation of why." (lines 1710–1712).
**Diagnosis:** The mechanism explanation in §3.5 and §8.4 (folded \(r\) → unnormalized \(\hat p_r\) → loss below truth) is theoretically clean and the empirical evidence (single ablation: RMSE 1.56 vs 0.045, conditional PIT failures, loss below truth) is striking. However, "Never" is sweeping: the ablation is *one* problem (Exp rate, single \(d=1\) sufficient statistic, single architecture: joint UMNN in \(\theta\)). The theoretical argument in §3.5 supports the broader claim, but the experimental record alone supports only "in this model family, autograd-Jacobian fails." The wording "documentation of why" overstates how much empirical generalization §8.4 carries.
**Suggested fix:** Either (a) sharpen the framing to "Never rely on autograd-only Jacobians: §3.5 gives the mechanism (no lower bound on the loss when \(r\) folds), §8.4 confirms it empirically in one setup," making clear theory is doing the heavy lifting; or (b) add a sentence acknowledging the single-experiment scope and reframing "Never" as "we treat this as a hard rule because the failure mode is structural — but the experimental basis is one ablation."

### C-9-rho
**Verdict:** ✓ holds
**Quote (from .tex):** "Use a proposal \(\rho\) with full support on the region of interest. The framework is robust to the choice of \(\rho\) in the population limit, but finite-sample performance benefits from \(\rho\) covering the region where inference will be performed. Edge effects (§8.1) appear near the support boundary." (lines 1714–1718).
**Diagnosis:** Coheres with §1.3 (any choice of \(\rho\) with full support yields the same population pivot — pre-requisite C-1.3-rho) and with the §8.1 finding (edge effects near the \([-7, 7]\) boundary — C-8.1-edge). The "robust in the population limit but finite-sample wants coverage" framing is exactly what those upstream claims say.
**Suggested fix:** None.

### C-9-training
**Verdict:** ✓ holds
**Quote (from .tex):** "Train with NF-MLE. No annealing schedule needed. Batch size 256–1024; Adam at \(\sim 3 \times 10^{-3}\); 3,000–10,000 steps depending on problem complexity." (lines 1720–1722).
**Diagnosis:** These are practical hyperparameters that match the §8 experiments' setups (all four run in 2–4 minutes on CPU). The claim is an engineering recipe, not a theoretical statement; treating numbers as correct per round-1 instructions, this is internally consistent. "No annealing" matches §7's "single-stage training" comment (line 1284).
**Suggested fix:** None.

### C-9-validate
**Verdict:** ⚠ under-justified (mischaracterization risk)
**Quote (from .tex):** "If only marginal PIT passes but conditional fails, you have a Hermans-style problem and need to revisit the architecture." (lines 1726–1728).
**Diagnosis:** Hermans et al. (2022) document that mainstream SBI families produce *overconfident posteriors* — empirical coverage below nominal at realistic budgets. The manuscript uses "Hermans-style" in two slightly different senses: at line 174 it cites overconfidence (the original meaning); at line 611 (§3.7 Class-1 discussion) it equates Hermans's pattern with "marginal PIT passing while conditional fails." These are *related but not the same*. Marginal-PIT-passes-while-conditional-fails is the right diagnostic *symptom* a CD-SBI practitioner should care about, but tying the name "Hermans-style" specifically to this dichotomy is the manuscript's own framing, not a direct restatement of Hermans 2022. A skeptical reader might ask whether Hermans 2022 ever framed the issue as marginal-vs-conditional PIT (the paper's primary tools were expected-coverage and TARP-like checks rather than PIT decomposition). The substantive point — that this is a coverage-calibration symptom CD-SBI is designed to avoid — is correct; only the attribution is slightly stretched.
**Suggested fix:** Replace "you have a Hermans-style problem" with "you have the kind of conditional miscalibration documented in Hermans et al. (2022) — marginal calibration is not enough." Or add a parenthetical clarifying that Hermans 2022 documented overconfidence and the marginal/conditional PIT decomposition is the diagnostic lens used in this manuscript to localize it.

### C-9-invert
**Verdict:** ✓ holds
**Quote (from .tex):** "At test time, invert. Given observed \(X_\mathrm{obs}\), compute \(\mathcal{C}_\alpha = \{\theta : \|r(\theta, X_\mathrm{obs})\|^2 \le \chi^2_{d, \alpha}\}\) by 1D root-finding (autoregressively, in \(d > 1\))." (lines 1730–1733).
**Diagnosis:** Restates C-6.4 (inversion algorithm) and is consistent with the triangular-flow Jacobian structure of §6.2 (each \(r_k\) is monotone in \(\theta_k\) at fixed \(\theta_{<k}\), so 1D root-finding per coordinate is well-defined).
**Suggested fix:** None.

### C-10-tbl
**Verdict:** ⚠ under-justified (one row stretches the attribution)
**Quote (from .tex):** Comparison table lines 1758–1776. Per-row check:
- **NPE / SNPE → Bayesian posterior, None built-in (Papamakarios & Murray 2016; Greenberg et al. 2019).** Holds.
- **NLE / SNL → Likelihood, None built-in (Papamakarios et al. 2019).** Holds.
- **NRE / SNRE → Likelihood ratio, None built-in (Hermans et al. 2020; Miller et al. 2022).** Holds — NRE/SNRE indeed target the likelihood ratio.
- **Balanced NRE → Posterior, Conservative regularizer (Delaunoy et al. 2022).** ⚠ Stretched. Balanced NRE adds a balancing regularizer to NRE; mechanically it estimates a *ratio*, not a posterior. The downstream goal is more reliable posteriors, but listing the target as "Posterior" while NRE/SNRE (the parent method) is listed as "Likelihood ratio" creates an inconsistency. Either both should be ratio-target methods, or the table column should be "downstream object" rather than "target."
- **Calibrated NPE → Posterior, Differentiable coverage (Falkiewicz et al. 2023).** Holds.
- **LF2I (ACORE, BFF) → Confidence sets, Neyman construction on learned test stat (Dalmasso et al. 2024).** Holds; matches §1.1's framing. (Note: ACORE/BFF date to 2020–2021; "Dalmasso et al. 2024" presumably refers to the consolidated LF2I paper. Citation-year correctness is round 2's job, but flag for that pass.)
- **WALDO → Confidence sets, Neyman inversion on Wald-style stat (Masserano et al. 2023).** Holds.
- **Box CD → Confidence sets, Depth-based, hyper-rectangles (Bortolato & Ventura 2025).** Holds as far as the manuscript's prior framing (line 182, "depth-based hyper-rectangles") is concerned. This is the author's interpretation of a recent paper; flagged for round-2 verification but not obviously wrong.
- **Variational SBI w/ coverage → Posterior, Conformal post-hoc (Patel et al. 2023).** Holds, with the caveat that "conformal" is a strong characterization — round 2 should confirm Patel et al. 2023 actually uses conformal prediction rather than another coverage-correction mechanism.
**Diagnosis:** Eight of nine rows are factually defensible. The Balanced NRE row is the only one with a real attribution mismatch — labelling its target as "Posterior" while the sibling NRE rows say "Likelihood ratio" misrepresents the method's mechanics.
**Suggested fix:** Change Balanced NRE target column to "Likelihood ratio" with a note "(goal: better posteriors)" — or relabel the "Target" column as "Inferential output" so the NPE/Balanced-NRE/SNL distinction lands consistently. Flag the ACORE/BFF and Patel attributions for round 2.

### C-10-closest
**Verdict:** ⚠ under-justified
**Quote (from .tex):** "The closest cousins are LF2I and WALDO (which target test-statistic CDFs to construct confidence sets via Neyman inversion) and SNL (which uses the same NF-MLE loss but without the monotone-in-\(\theta\) architectural constraint, and without the frequentist inference machinery)." (lines 1779–1783).
**Diagnosis:** The LF2I/WALDO characterization is consistent with the manuscript's earlier statement (lines 680–686). The SNL claim is the more interesting one: §3.4 already establishes that NF-MLE and SNL's loss are *identical* per-sample, so "same NF-MLE loss" is exactly right. However, the contrast is described as missing "the monotone-in-\(\theta\) architectural constraint" — this omits the equally load-bearing monotone-in-\(X\) (R2) constraint that §3.5 and §8.4 demonstrate is *necessary* for the loss to be well-posed. The manuscript's own architectural recipe (§8.5 line 1666–1671) lists *both* monotonicities; restricting the "what makes CD-SBI different from SNL" comparison to monotone-in-\(\theta\) is partial. Similarly, "without the frequentist inference machinery" is true but vague — the concrete distinction is the chi-square pivot inversion of §6 vs. SNL's MCMC-on-surrogate.
**Suggested fix:** Reword to "SNL (same NF-MLE loss, but with a generic flow architecture rather than the monotone-in-\(\theta\) + monotone-in-\(X\) prescription, and with MCMC-on-surrogate inference rather than direct chi-square inversion)." This matches the §3.4 comparison table (lines 538–545) exactly.

### C-10-singlestage
**Verdict:** ✓ holds
**Quote (from .tex):** "Unlike LF2I/WALDO, which require a separately-trained calibration step (the 'critical-values branch'), CD-SBI is single-stage: the same network output gives both the CD and the confidence set, with the calibration enforced in the population limit by the architecture + loss combination rather than by a hold-out calibration sample." (lines 1790–1795).
**Diagnosis:** Consistent with the LF2I/ACORE/BFF literature: those methods learn a test statistic (e.g., a likelihood-ratio classifier) and then estimate critical values \(C_\alpha(\theta)\) by Monte-Carlo from a separate calibration set — what the authors of LF2I do call a "critical-values" estimation step. WALDO likewise constructs Wald-type pivots and inverts via simulated quantiles. The "single-stage" characterization of CD-SBI follows from the population-limit identity \(r(\theta; X) \mid \theta \sim \mathcal{N}(0, I_d)\): the chi-square reference is closed-form and architecturally enforced. No calibration sample needed.
**Suggested fix:** None.

### C-10-IRT
**Verdict:** ✗ error (in the location-normal identification)
**Quote (from .tex):** "the optimal pivot in the location-normal model is the standardized log-likelihood gradient, and in regular exponential families more generally relates to the signed-root log-likelihood ratio (Schweder & Hjort 2016, ch. 5)." (lines 1798–1801).
**Diagnosis:** The location-normal score is \(\partial_\theta \log p(X \mid \theta) = X - \theta\); its standard deviation is 1, so the standardized score is \(X - \theta\). The manuscript's optimal pivot is \(r^*(\theta, X) = \theta - X\) (§4 and §8.1). These differ by a sign — the pivot is the *negative* of the standardized score. That sign is not cosmetic: \(\partial_\theta r^* > 0\) (the R1 constraint) requires \(r^* = \theta - X\), whereas the score \(X - \theta\) is *decreasing* in \(\theta\). So "standardized log-likelihood gradient" is incorrect as stated; the precise identification is "the negative standardized score" or, equivalently in this case, the standardized location of \(\theta\) relative to \(X\). The signed-root log-LR claim in regular exponential families is a standard Barndorff-Nielsen / SH result (the modified signed root \(r^*\) of Barndorff-Nielsen, 1986, with higher-order corrections; SH 2016 ch.~5 develops the connection), so the second half is defensible — but the sign issue in the location-normal example undercuts the rhetorical move.
**Suggested fix:** Either (a) replace "standardized log-likelihood gradient" with "negative standardized score" (or "standardized score, with the sign convention \(\partial_\theta r > 0\)"), or (b) reframe as: "the optimal pivot \(r^* = \theta - X\) is, up to sign, the standardized score; in regular exponential families this connects to the signed-root log-likelihood ratio (Schweder & Hjort 2016, ch. 5)." Option (b) is cleaner and preserves the intended bridge to the implicit-LR tradition.

## Cross-Part findings

- **[Part I → Part VI, C-9-validate]:** The phrase "Hermans-style problem" is used in two slightly different senses across the manuscript (line 174: overconfident posteriors; line 611 / line 1727: marginal PIT passes, conditional fails). Align these — either define "Hermans-style" once early on and use it consistently, or detach the marginal/conditional PIT framing from the Hermans citation.
- **[Part II → Part VI, C-10-closest]:** The "closest cousin" comparison with SNL omits monotone-in-\(X\) (R2), which §3.5 and §8.4 elsewhere argue is *necessary*, not optional. Recommend mirroring the §3.4 table (lines 538–545) in the §10 prose so both monotonicities are named.
- **[Part IV → Part VI, C-10-IRT]:** Sign convention. §4 derives \(r^* = \theta - X\) (positive in \(\theta\) by R1); §10 identifies this with "the standardized log-likelihood gradient" \(X - \theta\), which is the negative. Recommend a sign-convention sentence either at the §4.5 SH bridge or in the §10 IRT paragraph so the identification with the score / signed-root LR is unambiguous.
