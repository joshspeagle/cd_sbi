# Round 2 — Part II (Loss design) Lit-Review

**Reviewer:** Part II agent (Gneiting–Raftery 2007; Wehenkel–Louppe 2019; Gretton et al. 2005)
**Manuscript:** `cd_sbi_v7.tex`, Part II = §3, lines ~437–746.
**Bibliography:** `cd_sbi.bib`.

## Summary
- **Bibkeys verified:** 3 (`GneitingRaftery2007`, `WehenkelLouppe2019`, `GrettonEtAl2005`).
- **Verdicts:** 1 correct (✓), 2 minor issues (⚠), 0 major issues (✗).
- **Citation gaps in §3:** 3 substantive (one each: NF/diffeomorphism background; the "MMD-vs-MLE gradient signal" remark; the CRPS dual / kernel-score form). Several historical-context additions also suggested for round 3.

## Per-citation findings

### `GneitingRaftery2007`
**Status:** ⚠ (minor: definitional stretch)
**Verified:** https://www.tandfonline.com/doi/abs/10.1198/016214506000001437 (DOI 10.1198/016214506000001437; JASA vol. 102, no. 477, March 2007, Review Article, pp. 359–378). Free preprint at https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf and DTIC ADA459827.
**Attribution check:** The .bib entry's metadata (authors, journal, volume, number, pages, year) all match the published article. The manuscript's prose attribution — "(The term comes from the scoring-rule literature; G&R is the standard reference.)" — is accurate at the level of citing G&R as the standard reference. The substantive issue is conceptual, not metadata.
**Notes:**
- **G&R's actual definition** (their §2.1, eq. 1): "S is *proper relative to P* if S(Q, Q) ≥ S(P, Q) for all P, Q ∈ P; *strictly proper* if equality iff P = Q." This is "**forecast P vs. data-generating Q**, both in a class P of probability measures, with the population score uniquely maximized at the truth P = Q."
- **CD-SBI's usage** (§3.2, line ~527): "A loss is strictly proper for a target M if its population-level minimum, over the class of admissible r, is attained exactly on M and nowhere else."
- The CD-SBI form is a *generalization* of G&R: from "minimum is the single true forecast" to "minimum is the manifold M of equivalent calibrated pivots." This is conceptually defensible — all r ∈ M produce the same conditional law p(X | θ), so M plays the role of a "set of correct forecasts" — but it is not literally the G&R definition.
- **Suggested round-3 fix (optional):** add a one-sentence remark in §3.2 that the manuscript's "strictly proper for M" extends the G&R notion from a single-point target to an equivalence-class target, with the equivalence induced by the architectural class's non-identifiability under the calibration constraint. This is a one-line caveat, not a substantive rewrite. Alternatively, cite G&R's Example 3 + §9 (logarithmic rule = ML estimator) more directly, since NF-MLE = log score on the conditional density p(X | θ) is *literally* G&R's framework, and the manifold structure is a consequence of architectural unidentifiability rather than a generalization of propriety itself.
- See `references/GneitingRaftery2007.md`.

### `WehenkelLouppe2019`
**Status:** ⚠ (minor: §7.1 attribution mixes W&L's contribution with CD-SBI's implementation choices)
**Verified:** https://proceedings.neurips.cc/paper_files/paper/2019/file/2a084e55c87b1ebcdaad1f62fdbbac8e-Paper.pdf (NeurIPS 2019, vol. 32); arXiv:1908.05164. Reference implementation: https://github.com/AWehenkel/UMNN. Authors: Antoine Wehenkel, Gilles Louppe (Univ. of Liège). All confirmed against the paper and code.
**Attribution check:**
- **§3.5 (line ~643) attribution: ✓ accurate.** "UMNNs ... represent a scalar monotone function as the integral of a positive function from a free neural net" matches W&L's actual contribution at the abstract level.
- **§7.1 (line ~1566) attribution: ⚠.** The "Definition (UMNN; Wehenkel & Louppe 2019)" presents:
  - integrand = `softplus(MLP(t, c))` — but **W&L use ELU + 1**, not softplus (paper eq. 1: "an ELU activation unit increased by 1"). The reference implementation (`models/UMNN/MonotonicNN.py`, class `IntegrandNN`) uses `nn.ELU()` then `+ 1`.
  - integration scheme = "Gauss–Legendre quadrature (n = 12 nodes by default)" — but **W&L use Clenshaw–Curtis**, not Gauss–Legendre (paper §2 "Forward integration"; reference implementation `models/UMNN/NeuralIntegral.py` has the explicit comment "Clenshaw-Curtis Quadrature Method" and uses cosine-grid CC weights).
- **Why this matters:** the W&L paper's abstract claim is general — "a free-form neural network whose only constraint is for its output to remain strictly positive." Softplus and Gauss–Legendre are both valid instantiations, so CD-SBI's choices are not *wrong* as UMNN-style architecture. But §7.1's framing — `\textbf{Definition (UMNN; \citealp{WehenkelLouppe2019}).}` followed by the specific softplus + Gauss–Legendre construction — reads as if these specifics are part of W&L's definition.
**Notes:**
- **Suggested round-3 fix:** reframe §7.1's definition box. Two clean options:
  - **(a)** State W&L's general definition first ("integral of any strictly positive parametric function, computed by quadrature"), then introduce CD-SBI's deployment choices (softplus + Gauss–Legendre, n = 12) as the specific instantiation used in the experiments. This makes the citation accurate and the deviations explicit.
  - **(b)** Keep current text but add one sentence: "We substitute softplus for W&L's ELU+1 (for smoother gradients near zero) and Gauss–Legendre for W&L's Clenshaw–Curtis (negligible difference at n = 12 on this integrand class)."
- The doubly-monotone UMNN form (§6.3 / §7.1, eq. for `r_k`) — monotone in *both* the integration variable θ_k and the input X_k — is a real CD-SBI contribution that generalizes W&L's "monotone in the integration variable" structure. Worth flagging as such in §6.3 / §11 rather than implicit in the W&L cite.
- The remark on autograd-Jacobian failure (§3.5 line ~621, §2.4 line ~469 "ReLU / UMNN networks with no built-in monotonicity") is *consistent with* but not literally *from* W&L 2019 — W&L's autograd discussion is the Leibniz-rule memory trick (paper eq. 2–3), not a cautionary tale about non-monotone architectures. The CD-SBI §8.4 demonstration is original.
- See `references/WehenkelLouppe2019.md`.

### `GrettonEtAl2005`
**Status:** ✓ (with one-clause caveat about kernel conditions)
**Verified:** https://link.springer.com/chapter/10.1007/11564089_7 (Springer LNCS / LNAI 3734, ALT 2005, pp. 63–77, DOI 10.1007/11564089_7); preprint http://www.gatsby.ucl.ac.uk/~gretton/papers/GreBouSmoSch05.pdf.
**Attribution check:** The manuscript's gloss "HSIC ... is a kernel-based test statistic that is zero iff two random variables are independent" is **correctly attributable to G&L 2005**. The round-2 worry that the iff direction comes from a later paper (Fukumizu et al. 2008 / Sriperumbudur et al. 2011) is *not* warranted: G&L 2005 already prove the iff as Theorem 4 (p. 69), for RKHSs with **universal kernels on compact domains**. Later work refines "universal" to "characteristic" and relaxes "compact" to "locally compact Hausdorff," but the iff itself is 2005.
**Notes:**
- **Minor caveat (not blocking):** the §3.7 glossary entry does not state the kernel-class condition (universal / characteristic). For a one-sentence taxonomy gloss this is acceptable, but a more rigorous formulation would write "HSIC is zero iff independence, provided the kernels are characteristic" and cite `\citep{GrettonEtAl2005, SriperumbudurEtAl2011}` together — making the existing `\nocite{SriperumbudurEtAl2011}` load-bearing instead of "see also."
- Bib metadata is minimal but correct: authors, year, booktitle, pages. A polish item for round 3 is to add `series = {Lecture Notes in Computer Science}, volume = {3734}, editor = {Jain, S. and Simon, H.U. and Tomita, E.}, publisher = {Springer}, doi = {10.1007/11564089_7}` for fuller plainnat rendering.
- See `references/GrettonEtAl2005.md`.

## Citation gaps

These are claims in §3 that currently carry no citation but plausibly should — graded by importance.

1. **§3.1, line ~484 — "A normalizing flow is a parametric family of diffeomorphisms ... that transforms a complicated source density into a simple base density."** This is the defining sentence for normalizing flows in Part II and currently has no inline cite. The `cd_sbi.bib` already has `PapamakariosEtAl2021` (the canonical NF review) and `DurkanEtAl2019` (NSF), cited at §6 introduction. Round-3 fix: add `\citep{PapamakariosEtAl2021}` at first mention of "normalizing flow" in §3.1 — it is a low-cost insertion and the standard reference for any reader who arrives at §3 without prior NF exposure. (Importance: medium — currently the term is defined inline so it's not ambiguous, but the cite anchors the formalism.)

2. **§3.7 Class 2, line ~693 — "(i) the HSIC term has a weak gradient signal far from the optimum compared to a per-sample density target (a standard observation in the MMD-versus-MLE training literature)."** "A standard observation in the literature" is the only attribution. The likely intended references are Dziugaite et al. (2015) "Training generative neural networks via Maximum Mean Discrepancy optimization" and/or Bińkowski et al. (2018) "Demystifying MMD GANs," both of which document the gradient-signal-vs-MLE issue. Round-3 fix: add at least one citation here. (Importance: medium — the claim is plausible and matches folklore, but as written it is unsupported.)

3. **§3.7 Class 4, line ~720 — the dual form of CRPS** "CRPS(F, y) = E_{Z~F}|Z − y| − ½ E_{Z,Z'~F}|Z − Z'|." This is the **kernel-score / energy-distance representation of CRPS**, due originally to Gneiting & Raftery 2007 (their Theorem 9 / Section 5 on the energy score) and Székely & Rizzo (2004) on the energy distance. Since `GneitingRaftery2007` is already in the bib, round-3 could cite it here as the source for the dual form — strengthening the §3.7 Class-4 argument (which depends crucially on this representation) and making the propriety-error analysis traceable to a primary source rather than a folkloric formula. (Importance: medium-high — the entire Class-4 collapse argument hinges on this form being correct and well-attributed.)

Minor additional notes (not gaps per se):

- **§3.4, line ~570 — "identical to the per-round training loss of Sequential Neural Likelihood (SNL)."** Cite `\citep{PapamakariosEtAl2019}` is present. ✓
- **§3.5, line ~639 — the empirical demonstration that autograd-Jacobian ablation attains loss 0.56 < 0.88.** Internal forward-reference to §8.4, no cite needed.
- **§3.7 Class-5 "Position in the SBI literature" paragraph** — all primary SBI methods (NPE/SNPE, NLE/SNL, NRE, Balanced NRE, Calibrated NPE, LF2I, WALDO, Box CD) are correctly cited inline. ✓

## Historical-context additions

Not required for citation correctness, but each would *strengthen* §3 if added in round 3:

- **§3.2 — the log-score = ML equivalence.** G&R §9 ("optimum score estimation") makes explicit that maximum likelihood is the M-estimator under the strictly proper logarithmic scoring rule. The CD-SBI claim "NF-MLE is strictly proper for the calibration manifold within the architectural class" is *exactly* this construction applied to the conditional density p(X | θ), with M arising from architectural non-identifiability. A round-3 §3.2 could cite G&R Example 3 + §9 directly, making the strict-propriety derivation a one-line invocation of the G&R log-score result rather than a fresh KL-non-negativity argument. The current proof in §3.2 (KL ≥ 0 ⇒ minimum iff conditional law matches) is correct and self-contained, but the literature connection is currently invisible.

- **§3.5 — UMNN's actual implementation choices.** Worth mentioning W&L's original ELU+1 + Clenshaw–Curtis design (per `references/WehenkelLouppe2019.md`), since the manuscript's choices differ. The §11 open-problems section discusses higher-d scaling (§11.5), where quadrature cost matters; flagging the design space (CC vs GL vs adaptive Gauss-Kronrod, etc.) explicitly in §11 would be useful.

- **§3.7 Class 2 — HSIC's kernel-class requirement.** The iff direction requires characteristic kernels. The current gloss elides this. A one-clause addition ("HSIC under characteristic kernels is zero iff independence") + a pair-cite `\citep{GrettonEtAl2005, SriperumbudurEtAl2011}` would make the existing `\nocite` substantive. Combined with citation gap (2) above, this could also pre-empt a referee remark on kernel-method rigor.

- **§3.7 Class 4 — the LF2I parenthetical (line ~743).** Good cross-reference to `DalmassoEtAl2024`. ✓ Round-3 could expand this to one sentence about the direction-of-propriety distinction more generally, since it is the key conceptual subtlety of the Class-4 argument.
