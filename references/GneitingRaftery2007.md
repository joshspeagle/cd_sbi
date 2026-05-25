# Gneiting & Raftery, 2007 — Strictly Proper Scoring Rules, Prediction, and Estimation

**Authors:** Tilmann Gneiting, Adrian E. Raftery
**Year:** 2007
**Venue:** Journal of the American Statistical Association, vol. 102, no. 477, pp. 359–378 (Review Article)
**arXiv:** n/a (also issued as Univ. of Washington Stat. Dept. Tech. Report no. 463R, and DTIC ADA459827)
**DOI:** 10.1198/016214506000001437

## One-paragraph summary
A review article and theoretical synthesis of *proper scoring rules* — functions S(P, ω) that score a probabilistic forecast P against a realized outcome ω. The central definition (eq. 1, p. 360): given a convex class P of probability measures, a scoring rule S is **proper relative to P** if S(Q, Q) ≥ S(P, Q) for all P, Q ∈ P, and **strictly proper relative to P** if equality holds iff P = Q. The paper proves a rigorous version of the Savage representation for categorical variables (Theorem 2, McCarthy/Savage); links scoring rules to convex/entropy/Bregman-divergence structure (Theorem 1); systematically reviews logarithmic, quadratic (Brier), spherical/pseudospherical, CRPS, energy, and kernel scores; and proposes a novel form of cross-validation (random-fold CV). The estimation Section §9 ("optimum score estimation") frames maximum likelihood as a special case of optimum score estimation under the strictly proper logarithmic rule, with M-estimation as a further generalization.

## Why CD-SBI cites it
§3.2 (line ~530): the standard reference for the terminology "strictly proper." CD-SBI's claim is that the NF-MLE loss is *strictly proper for the calibration manifold M within the architectural class F* — i.e., the population minimizer set of L(r) over F equals M ∩ F. The G&R reference grounds the use of the term in established literature.

## Specific anchors
- **Definition (proper / strictly proper):** §2.1, eq. (1), p. 360. "S(Q, Q) ≥ S(P, Q) for all P, Q ∈ P," strict iff equality ⟺ P = Q. Note: defined for **forecasts P vs. true measure Q**, both in a class P.
- **Optimum score estimation:** §9 — uses strictly proper scoring rules to define M-estimators; ML = optimum score estimation under logarithmic rule.
- **Logarithmic rule:** Example 3, p. 363. S(p, i) = log p_i; associated Bregman distance is KL — the direct conceptual bridge to the NF-MLE loss in CD-SBI §3.2.

## Notes
- **Attribution check (definitional stretch, ⚠).** G&R's definition is "strictly proper relative to a class P of probability measures," where P is the forecast class and propriety is measured against an underlying truth Q ∈ P, with the minimizer being a *single point* (P = Q). CD-SBI's usage in §3.2 ("strictly proper for a target M ... minimum attained exactly on M and nowhere else") generalizes this from a singleton truth to a *manifold* M of equivalent r-parameterizations that all produce the correct calibrated pushforward. This is conceptually sound — any r ∈ M plays the role of a "correct" forecast, since they all encode the same conditional law — but it is not literally G&R's definition. Worth either (i) a one-line remark that the manuscript extends the standard notion from "unique truth" to "manifold of equivalent solutions," or (ii) framing the loss as strictly proper *for the conditional law* p(X | θ) in G&R's exact sense, with the M structure as a consequence of architectural non-identifiability rather than a generalization of propriety itself.
- **Logarithmic-rule connection (potential ✓-enhancement).** The fact that NF-MLE = log score = strictly proper scoring rule for conditional densities (in G&R's sense, eq. (1), with the target distribution being p(· | θ) at each θ) is exactly Example 3 + §9 of G&R. The current §3.2 cites G&R only for the *terminology* "strictly proper"; it could equivalently cite G&R for the deeper fact that log score is strictly proper, which is the substance of the manuscript's claim. This would make the citation load-bearing rather than terminological.
- The paper is co-authored by Raftery (UW Statistics, where Gneiting was a visitor); the published-version DOI resolves to Taylor & Francis. The UW tech report ("Strictly Proper Scoring Rules, Prediction, and Estimation (Revised)") is a slightly later revision; both are commonly cited under year 2007.
- Cox–Cox–Pfanzagl trail: the "optimum score estimator" terminology in G&R §9 references Pfanzagl 1969 and Birgé–Massart 1993 ("minimum contrast estimation"); the equivalence ML = optimum log score is folklore but G&R is where it is cleanly stated in modern form.
