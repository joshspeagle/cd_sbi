# Lemos et al., 2023 — Sampling-Based Accuracy Testing of Posterior Estimators for General Inference (TARP)

**Authors:** Pablo Lemos, Adam Coogan, Yashar Hezaveh, Laurence Perreault-Levasseur
**Year:** 2023
**Venue:** International Conference on Machine Learning (ICML 2023), PMLR vol. 202
**arXiv:** 2302.03026
**PMLR:** https://proceedings.mlr.press/v202/lemos23a.html

## One-paragraph summary
TARP — Tests of Accuracy with Random Points — is a sample-only coverage
diagnostic for generative posterior estimators (NPE, NRE, MCMC,
likelihood-based, …). For each simulated `(θ₀, X)` pair with posterior
samples `{θ̂ⱼ} ∼ π̂(θ | X)`, TARP draws a random reference point
`θ_ref` (e.g., uniformly from a region containing the support), computes
the distance `dⱼ = ‖θ̂ⱼ − θ_ref‖`, and asks whether the true `θ₀` falls
inside the `α`-credible "spherical" region around `θ_ref` — i.e.,
inside the smallest ball containing an `α`-fraction of the posterior
samples around `θ_ref`. Aggregating across simulations, the expected
empirical coverage at nominal level `α` should equal `α` if and only if
`π̂ = π` almost everywhere. The paper proves this is a **necessary and
sufficient** condition for posterior correctness (their main theorem),
in contrast to SBC-style rank uniformity, which is only necessary. TARP
needs only samples (no density evaluations) and works in high dimensions
where direct credible-region computation is intractable.

## Why CD-SBI cites it
Cited at §7.3 as one of three named "related diagnostic frameworks"
(SBC, TARP, LF2I), positioning CD-SBI's coverage diagnostic (item 5 in
§7.3) — `P_{X | θ₀}(θ₀ ∈ C_α(X))` — relative to existing SBI calibration
tools. The manuscript then states CD-SBI's coverage check is "closest
in spirit to the LF2I local-coverage check," distinguishing it from
TARP.

## Specific anchors
- §1–2 of the paper for the TARP construction and the
  necessary-and-sufficient theorem.
- §3 (synthetic experiments) and §4 (gravitational lensing application)
  for empirical demonstrations on high-dim problems where SBC's marginal
  rank uniformity passes but TARP correctly rejects.
- The TARP coverage statistic is **marginal over the prior / proposal**
  in the same way SBC is — the expectation is over draws of `θ₀ ∼ π(θ)`.
  It is therefore not pointwise/conditional in `θ₀`, which is the
  distinction the manuscript correctly draws when it says CD-SBI's
  coverage check is closer to LF2I (which is explicitly local) than to
  TARP.

## Notes
- Metadata: BibTeX entry has authors with initials only
  (`P. Lemos, A. Coogan, Y. Hezaveh, L. Perreault-Levasseur`). For
  natbib `plainnat`, initials are fine; if the BibTeX style demands
  full first names elsewhere in the .bib, this entry is consistent.
  Title contains `{TARP}` to preserve casing — good.
- Attribution: the brief mention at §7.3 is accurate but slightly
  understates TARP's strength. Three points worth noting (for possible
  weaving in, but not required):
  1. TARP's main selling point over SBC is the
     necessary-and-sufficient guarantee. SBC's uniformity is only
     necessary (a sampler can have uniform ranks yet miscalibrated
     posterior shape). The manuscript does not make this distinction
     — fine for a brief mention, but if §7.3 were to expand on the
     three diagnostics it would be the natural place.
  2. TARP is **sample-only** (no posterior-density evaluations needed),
     which makes it natively applicable to NPE-style samplers; CD-SBI's
     coverage check shares this property since `C_α(X) = {θ : ‖r(θ; X)‖² ≤ χ²_{d,α}}`
     is computed by inverting the trained pivot on a grid.
  3. The manuscript's positioning sentence is correct: TARP is a
     **marginal** test (averaged over the proposal/prior), whereas
     CD-SBI's coverage diagnostic and LF2I are local in `θ₀`. So
     "closest to LF2I, not TARP" is the right call.
- **No attribution correction needed.** The brief mention is accurate;
  any expansion would be a stylistic choice, not a factual fix.
