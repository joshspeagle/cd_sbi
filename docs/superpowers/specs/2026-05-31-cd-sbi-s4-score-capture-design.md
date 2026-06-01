# CD-SBI S4: non-oracle CD-SBI via learned score-capturing summaries

> **⛔ RETIRED 2026-05-31 — superseded by
> `2026-05-31-cd-sbi-moment-summary-design.md`.** Sequential regression-summary
> training dissolves the collapse problem this spec was built to solve (no Fisher
> term, no §13.3 score bias, no likelihood-free phase), and multimodality turned out
> to be an R1 question, not a bottleneck question (theory note §15–16). Kept for the
> audit trail; do not plan from it.

**Status:** design (validated in brainstorm 2026-05-31). Drives the S4 build.
**Theory spine:** `docs/theory/2026-05-31-conditional-coverage-decomposition.md`
(§11–14). Every design choice below traces to a numbered result there.
**Predecessors:** Stage-B bake-off (M2 / Arm I-A / Arm II-A,
`…/specs/2026-05-29-…-stage-b-device-bakeoff-design.md`); d=5 Bartlett
(`…/specs/2026-05-30-cd-sbi-bivariate-normal-mu-cov-design.md`, verdicts N2/N3).

---

## 1. Goal and why

**Goal.** A *non-oracle* CD-SBI procedure: a **learned** summary `s_φ: 𝒳 → ℝ^d`
plus an injective pivot flow that yields confidence sets which are **valid**
(exact conditional coverage), **uniformly** valid (everywhere, not just on
ρ-average), and **efficient** (set volume → oracle), scaling to high `d`.

**Why now.** The theory pass established (note §1–14) that the framework's
inferential chain is airtight and that the non-oracle high-`d` problem is the
**efficiency** axis, not validity:
- *Validity is already non-oracle* — any `(A)`-regular summary + injective
  sign-correct flow reaching `q≈p` covers exactly (Thm 1, Prop 2). The N3 learned
  summaries already calibrated (`χ²₅≈0.03`); they were *valid but inefficient*.
- *Efficiency ⟺ the summary captures the score* (`I_{s_φ}(θ)=I_X(θ)`, Cor 12).
  This is the one thing the learned-summary arms failed at: II-A (calibration-only)
  **collapsed** scale; I-A (invertibility) recovered σ at d=1 but **failed to route
  the quadratic covariance stats at d=5** (N3).
- The collapse theorem (#3, Thm 7) proves coverage **must** be paired with an
  information/efficiency term; Thm 13 gives the exact objective; §13.3 proves that
  term needs *score-level* control (KL does not control the score); §14.1 proves
  the loss certifies coverage only against the summary's *own* floor.

**S4 is "II-A done right":** an end-to-end learned summary, but with the
Fisher-information / score-capture term that §12–13 proved is the necessary
anti-collapse ingredient II-A lacked — a **dimension-`d`** anti-collapse that does
not require I-A's full invertibility, so it can scale where I-A did not.

## 2. The construction (objective)

Over summary params `φ`, flow params `ψ` (flow injective + sign-correct ⟹ C1):

    L(φ,ψ) = E_{θ∼ρ} KL( p(·|θ) ‖ q_ψ(·|θ;φ) )      [V: validity = NF-MLE]
           + λ · E_eff(φ,ψ),                          [E: efficiency / anti-collapse]

`q_ψ(T|θ;φ)` the flow density of `T=s_φ(X)`. Two realizations of `E_eff` that
**share the population optimum `I_T=I_X`** (Cor 12) but **differ off-optimum** —
that difference (driven by the §13.3 flow-score bias) is exactly what Phase 1
measures; they are *not* interchangeable:

- **B — score-residual** (Cor 12 direct; the §13.3 score control), the **primary**
  training objective: `E_eff = E‖ U_X(θ) − g_ξ(s_φ(X),θ) ‖²` with an auxiliary
  regressor `g_ξ`. Its min over `g_ξ` is `E[Cov(U_X|T)] = I_X − I_T` (Thm 10), so it
  drives `I_T→I_X`. The identity holds only at `g_ξ`'s optimum, so `g_ξ` is trained
  to near-optimality *inside* each S4 step (`k_inner` SGD steps, default ~3, or a
  target-network update) — else the residual over-estimates `I_X−I_T`.
- **A — flow-Fisher log-det** (Thm 13 literal; fully likelihood-free), a
  **secondary probe** (the §13.3 bias test), gated on its unit test passing:
  `E_eff = − E_θ log det( Cov_group(∇_θ log q_ψ(s_φ(X)|θ)) + εI )`. By Thm 11 this
  is `+2·E log-volume` — but **Thm 11 is exact only on Gaussian/location
  directions and `O(1/n)`-asymptotic on the scale direction** (§14.2), which is the
  σ coord that matters most here, so A's log-volume reading is leading-order on σ.
  *Estimator constraints (non-negotiable):* the sample covariance is `d×d` over `m`
  replicates, so it needs `m > d` (use `m ≥ ~4d`) to be non-singular; the `log det`
  is computed via Cholesky of `(Î + εI)` with `ε` reported as a fraction of
  `tr(Î)/d`; A is second-order (`create_graph` through `∇_θ log q_ψ`, the project's
  documented Jacobian-failure regime), so it stays a probe until the unit test
  below confirms `log det Î` matches the analytic Fisher of a known Gaussian
  summary *at the chosen `m`*.

Key facts carried from the theory:
- **Validity ⊥ efficiency decouples** (Prop 15): `V` selects the calibration,
  `E` selects the summary; no population λ-trade-off.
- **Loss is per-summary** (Prop 18, §14.1): compare arms by `L − H̄(φ)`, never raw
  `L`; raw `L` is gameable by entropy collapse (the M2 cheat).
- **`E` needs score control** (§13.3): the flow-score is biased while `q_ψ≠p` and
  KL does not bound it — A probes whether this bias bites in practice; B sidesteps
  it via the true score.

## 3. Architecture & components

| Component | Choice | Notes |
|---|---|---|
| Summary `s_φ` | `DeepSetsConditioner` (reuse), `d_out=d` | same architecture as M2's summary; learned summary absorbs feature orientation |
| Flow | `SingleIndexMonotoneFlow(d, signs=(+,…))` | feature-space monotone (R1+R2); `feat_signs=+`, summary orients itself. **C1 is a *trained* property, not architectural** — see below |
| Runner | **new** `ScoreCaptureCDSBIRunner` | grouped-by-θ₀ batches (`B`×`m`); `efficiency_mode ∈ {none, score_residual, flow_fisher}` |
| Loss (eff) | **new** `losses/score_capture.py` | `ScoreResidualLoss` (B), `FlowFisherLoss` (A) |
| Diagnostic | **new** `FisherRecovery` | `det I_T/det I_X` + eigen-ratios of `I_T I_X^{-1}`; **`I_T`, `I_X` from the analytic/true score `U_X`, never the flow-score** (else circular vs A) |
| Data score `U_X` | autograd of `simulator.log_prob` | analytic for μσ / Bartlett (exp-family) |
| Reweighting (P2) | **new** least-favorable `π_LF` estimator | exponentiated-gradient on a θ-grid (Thm 9, §13.4) — **the estimator is theory-open**, see Phase 2 |

**Grouped batches (all modes).** `E_eff` for A is a per-θ₀ covariance, so the runner
*always* samples `B` groups of `m` replicates (as `EnergyCDSBIRunner` does), for
every mode. **Consequence:** `efficiency_mode=none` is therefore *not bit-identical
to the historical M2* (which used ungrouped i.i.d.-(θ,X) NF-MLE) — with `m`
replicates per group the effective distinct-θ₀ count per step is `B`, not `B·m`.
So `none`-mode is a **fresh, grouped collapse baseline** whose M2-style numbers
(loss-below-floor, σ-Spearman) are **re-measured**, not asserted equal to the old
smoke test. The clean attribution still holds: none/B/A share the identical grouped
codepath and summary/flow, differing only in `E_eff`.

**Sign / C1 with a learned summary (honest version).** The flow guarantees
`∂r_k/∂feat_k>0` *in feature space*; the learned `s_φ` orients its outputs so `+`
is the calibrated direction. But C1 (the theory note's diffeomorphism-onto-`ℝ^d`,
under hypothesis (A): abs-cont, full-support conditionals) is a property of `s_φ`,
which a generic DeepSets map (non-injective, `log_det_contrib=0`) does **not**
guarantee — and the Fisher term forbids only *Fisher* singularity (`det I_T→0`),
not the (A) abs-continuity (Cor 12 is "Fisher, not Shannon"). So **C1 is targeted
and certified post-hoc, not architectural** (mirroring the note's S3a: proven
implication + witnessed once trained): verify via pushforward-KS + the Thm-5
ρ-avg bound, and add an **(A)-regularity guard** to the diagnostics —
condition-number / rank of `Cov(s_φ(X))` (degenerate ⟹ (A) violated). The d>1
autoregressive feature ordering for a *learned* (unordered) summary is handled in
Phase 3 (§4); it is the exact thing N3 failed on, so it is a design question there,
not a forward-pointer.

## 4. Phases

Each phase extends Phase-1 code; no parallel codepaths.

### Phase 1 — (μ,σ²) objective + §13.3 probe
Build `ScoreCaptureCDSBIRunner`, both efficiency losses, `FisherRecovery`. Run the
**4-arm bake-off** on `NormalUnknownMeanVar` (`fresh_batch=true`):

| Arm | Summary | Eff term | Batching |
|---|---|---|---|
| none (fresh collapse baseline) | DeepSets | `none` | grouped |
| S4-B (primary) | DeepSets | `score_residual` (true score) | grouped |
| S4-A (probe) | DeepSets | `flow_fisher` | grouped |
| I-A | invertible | exact-density (existing runner) | ungrouped |

All four share summary/flow; none/B/A differ only in `E_eff` (§3 grouped-batch
note — the baseline is re-measured, not equated with historical M2).

**Success criteria** (primary signals are *true-score / model-free*, to avoid
circularity with A):
- *validity* — all arms, coverage-err ≤ 0.05 (pushforward-KS too);
- *anti-collapse (headline)* — S4-B σ²-Spearman ≥ 0.9 (model-free; vs none ≈ M2's
  ≈0.014) **and** the true-score residual `det I_T/det I_X ≥ 0.8` from
  `FisherRecovery` (eigen-ratios of `I_T I_X^{-1}`, both from the analytic `U_X`);
- *efficiency* — stated as `det I_T/det I_X` (Cor 12), **not** a set-volume ratio:
  the `SetSize` diagnostic uses set *diameter* at d>1, and Thm 11's volume formula
  is only `O(1/n)`-asymptotic on the σ direction, so the det-ratio is the clean
  measurable criterion;
- *§13.3 verdict* — S4-A vs S4-B compared on the *true-score* metrics (σ-Spearman,
  true-score residual); A materially worse ⟹ flow-score bias confirmed. A's own
  `flow_fisher` log-det is a *training* signal only, never the scoring metric.
- *λ selection* — sweep `λ ∈ {0, 0.1, 0.3, 1, 3}` (`0` = the none baseline); pick
  the smallest λ clearing σ-Spearman ≥ 0.9 with coverage-err ≤ 0.05 (Prop 15 says
  no population trade-off; this finds the finite-capacity operating point).

### Phase 2 — S2b uniform-coverage reweighting (Thms 8–9, §13.4) — *implements an open estimator*
**Note:** the theory proves the *principle* (minimax = least-favorable Bayes, `π_LF`
on the regret-extremes) but the *tractable estimator + convergence on a real flow
class is explicitly OPEN* (note §9½/§13.4). Phase 2 builds that open estimator; a
null/negative result is informative, not a failure.

Track **per-θ regret** `R(θ) ≈ CE_θ − H̄(θ)` (the per-θ floor-relative gap, §14.1 —
*not* raw `L_θ`, and not the floor-free `D(θ)` which Thm 7 shows is unsafe to
optimize jointly with a learned summary); reweight the training θ-distribution
toward high-regret θ via exponentiated-gradient (→ `π_LF`, Thm 9). The efficiency
term (separate) handles anti-collapse; reweighting only redistributes calibration
effort across θ. **Success:** `sup_θ` coverage error drops vs the ρ-average S4 on
(μ,σ²); the single-scale closed form (Thm 8: minimax = log-mean of the variance
range) reproduced on a controlled case. Fills `tests/theory/…::test_s2_minimax_trained`.

### Phase 3 — d=5 Bartlett non-oracle replacement (headline) — *P2-independent*
**Primary run uses ρ-average S4 (Phase-1 machinery, NO P2)** — the headline result
must not be coupled to P2's open estimator. P2 reweighting is an *add-on* applied
only after the ρ-average result is in (it addresses the μ₂-style tail, a refinement,
not a prerequisite). Full S4 (learned `s_φ:ℝ²⁰→ℝ⁵` + efficiency term) on
`NormalBivariateUnknownCov`, head-to-head vs oracle `BartlettSummaryConditioner`.

*Feature-ordering design (the N3 failure point, resolved here not deferred):* the
learned `s_φ` emits 5 *unordered* coordinates, but `SingleIndexMonotoneFlow` imposes
a triangular order and needs fixed length-5 `theta_signs`/`feat_signs` at
construction. Plan: `feat_signs=+` uniformly (summary orients), `theta_signs` from
the simulator's known KR θ-ordering (diagonals→off-diag→means); the learned summary
must align its coordinate `k` with pivot coord `k`. Whether a *permutation-free*
learned summary can discover this alignment — or whether a light structural prior
(e.g. ordering by a learned scalar score) is needed — is the concrete Phase-3
question.

**Decisive test:** does score-capture route the **quadratic covariance** sufficient
stats that I-A failed to route (N3: `A₂₂` never routed)? Theory predicts yes (the
term targets `I_T→I_X` in every direction). **Success:** coverage + per-direction
eigen-ratios of `I_T I_X^{-1} → 1` across all 5 directions (esp. the covariance
ones), approaching oracle Bartlett. A negative result is itself a sharp finding.

### Phase 4 — likelihood-free scoring (generalization)
Replace the analytic `U_X` in B with an *estimated* score. Real options: (ii) a
score-net via denoising / sliced score-matching on simulations; (iii) the NLE
density's score. **Not a peer option:** (i) reusing the flow-score A — that
reintroduces the exact §13.3 bias B was built to sidestep, so it is at best a
known-biased fallback to measure against, not the target. **Success:**
likelihood-free S4 (via ii/iii) matches analytic-score S4 on (μ,σ²) and d=5 —
deployable on arbitrary simulators.

## 5. File layout

```
src/cdsbi/
  losses/score_capture.py          # ScoreResidualLoss (B), FlowFisherLoss (A)
  methods/cd_sbi_score_capture.py  # ScoreCaptureCDSBIRunner (P1; extended P2,P4)
  diagnostics/fisher_recovery.py   # FisherRecovery (det I_T/I_X, per-direction)
  reweighting/least_favorable.py   # P2: π_LF exponentiated-gradient estimator
configs/
  method/cd_sbi_score_capture.yaml # + efficiency_mode / lambda / group params
  experiment/s4_*_bakeoff.yaml     # P1 (μσ 4-arm), P3 (d=5 vs oracle)
tests/
  unit/test_score_capture_losses.py
  unit/test_fisher_recovery.py
  integration/test_score_capture_runner.py   # none-mode = fresh collapse baseline; trains; (A)-guard
  intensive/test_replicate_mu_sigma_s4.py     # P1 4-arm head-to-head
  intensive/test_replicate_mu_cov_s4.py       # P3 d=5 vs oracle
```
`run.py::_build_method` dispatch extended for the new runner/loss.

## 6. Testing strategy (TDD)

- **Unit (drive the math):** B is minimized at a *known* sufficient summary on a
  synthetic Gaussian (residual → `I_X−I_T`); A's `log det Î` matches the analytic
  Fisher of a known Gaussian summary **at the chosen `m ≥ ~4d`** (this test gates
  A's promotion from probe to objective); `FisherRecovery` (true-score) returns
  ratio 1 for a sufficient summary, `<1` for a lossy one.
- **Integration:** `efficiency_mode=none` (grouped) **re-measures** the collapse
  (loss-below-its-own-floor `H̄(φ)`, σ-Spearman ≈ M2's) — it is a *fresh* baseline,
  not asserted bit-equal to `test_mu_sigma_stage_b_smoke`; the runner trains and
  yields a valid `PivotBasedProcedure`; the (A)-regularity guard flags a degenerate
  `Cov(s_φ(X))`.
- **Intensive:** P1 4-arm bake-off and P3 d=5-vs-oracle, against the Section-4
  thresholds; seed-averaged.

## 7. Open risks (surface, don't hide)

- **λ tuning** — validity↔efficiency frontier; start with a short sweep / mild
  anneal (Prop 15 says they don't conflict at the population level, but finite
  capacity couples them via the score bias §13.3 and `ε*(φ)` §9½).
- **A's covariance estimator is identifiability-constrained**, not just costly:
  the `d×d` sample covariance over `m` replicates needs `m > d` (use `m ≥ ~4d`) or
  `log det` is `εI`-dominated — at d=5 this *biases the estimator itself*, so "A
  worse than B" could be variance, not the §13.3 bias. Pin `m`; A stays a probe
  until its unit test passes at that `m`. Plus the second-order `create_graph`
  cost (the project's documented Jacobian-failure regime).
- **§13.3 bias** — A may mis-estimate/collapse; *measured* against true-score
  metrics (not A's own log-det), not assumed (Phase 1's secondary purpose).
- **C1 is not guaranteed for a learned `s_φ`** — hypothesis (A) (abs-cont,
  full-support) can fail for a degenerate DeepSets map, and the Fisher term forbids
  only *Fisher* singularity, not (A). Mitigated by the (A)-regularity guard +
  post-hoc certification (§3), not by architecture.
- **Phase 2 implements a theory-OPEN estimator** (§9½: `π_LF` tractable estimator +
  convergence unproven). It is decoupled from the Phase-3 headline (which runs
  ρ-average) precisely so the headline isn't hostage to it.
- **Phase 3 is the real unknown** — whether a learned summary routes quadratic
  stats is exactly what I-A failed and the theory predicts S4 fixes; a negative
  result is itself a publishable finding (sharpens the bespoke-summary-dead-end
  thesis). The learned-summary feature-ordering (§4 Phase 3) is the concrete risk.
- **`g_ξ` (B)** — under-capacity *or* under-training over-estimates the residual
  (conservative); give it an expressive MLP and `k_inner` inner steps per S4 step.

## 8. Out of scope (named, deferred)

R1-realizability beyond MLR (S5, §13.5 — shape not coverage); the SLCP/Two-Moons
validity-win experiment (separate track); image/sequence targets (v8). These do not
block S4.
