# LF2I-Score hardening — experiment-rigor checklist (2026-06-10)

Closing out the rigor issues the three audits found, so the LF2I-Score comparison is
publishable. **No shortcuts** — each item is implemented, tested, and (where it affects a number)
re-run. Env: `/home/user/venv` (clean venv; full stack installs; `334 passed` baseline).

Order is dependency-driven: fix what could invalidate a headline number first, then build the
missing experiments, then the accounting/eval upgrades, then re-run.

| # | Item | Audit | Verdict | Status | Acceptance |
|---|---|---|---|---|---|
| 1 | **Score-CD `nan_to_num` bias** | calib | FIX-FIRST (blocking) | **DONE** | non-finite score → REJECT (+inf, conservative); Fisher rows dropped; counted in `procedure.nonfinite_diagnostics`; unit-tested + quantified on real runs |
| 2 | NLE/NRE `ℓ_max` grid at d>1 | calib | FIX-FIRST | **DONE** | gradient-ascent refinement from top-k grid starts (derivative-free fallback); closed-form-exact in tests; real-NLE quantified |
| 3 | Score-CD-cal `n_params` undercount | match | UPGRADE+rerun | **DONE** | head counted (mirrors LF2I-BFF); budget-width head is <15% of backbone (no retune needed — configs already interpolate `${budget.quantile_hidden}`) |
| 4 | Arch logging (built flow ≠ `cfg.flow.name`) | match | UPGRADE+verify | **DONE** | index gains ground-truth `flow_class_built`; `_build_flow` warns LOUDLY at the exact fall-through; §8.4 verification folded into the re-run (old outputs absent here) |
| 5 | **Cauchy loc-scale simulator (NEW)** | targets | NEW | **DONE** | `CauchyLocScale` committed + config; exactly-calibrated closed-form oracle pivot (PIT→normal-scores→orthonormal projections; Gram-Schmidt, any n_iid); oracle at the measured floor through the engine (<0.03 over a 3×3 box grid) |
| 6 | Eval upgrade: oracle floor row + grid + metric | targets | UPGRADE | **DONE** | `eval_thetas_edge` made LIVE (fused grid, all diagnostics); engine + index report mean/p90/max; measured `r_star` oracle floor row on the identical grid/n, saved as `oracle_coverage.parquet` + index columns |
| 7 | Sim-cost accounting + relabel | match | UPGRADE | **DONE** | per-run `sim_calls_{train,calibration,inference,total_method}` in the index (live Fisher counter on Score-CD-rao; d-aware BFF marginal mirrored); budget log relabeled "Parameter budget … matched quantity is parameters" |
| 8 | `fresh_batch` 2-regime sweep | match | REDO | **DONE (configs)** | POC experiment configs sweep `training.fresh_batch: false,true`; execution = the re-run step |
| 9 | (optional) SLCP simulator | targets | NEW | deferred | only if the named benchmark is wanted beyond `SignNormal1D` |

**Then:** re-run the clean POC suite (regular ladder {1D-loc, exp-rate, 2D-corr} + no-suff-stat
{Cauchy} + non-regular {SignNormal1D}) with baselines {NLE same-flow, LF2I-BFF, CD-SBI, oracle},
both fresh_batch regimes, the upgraded eval — and rewrite the draft's §6 from the regenerated
tables.

## Findings log

**Item 1 (nan-guard), 2026-06-10.** Fix landed (commit on branch). **Empirical quantification**
(env `/home/user/venv`, real Score-CD-rao fits on `LocationGaussian2D_iid`, CPU, ~30s/fit):
- Representative worst regime (xlarge MAF hidden=156, `fresh_batch=False`, seed 0):
  `stat_nonfinite=0`, `coverage_error_max=0.0395`. The guard was **latent** — that old number was
  not inflated by it.
- Heavy-overfit stress (`n_train=500`, extreme θ₀=±6): still `stat_nonfinite=0`. MAF + grad-clip
  keeps autograd scores finite even when badly overfit and queried far from support.
- **Conclusion:** the bias was a *latent* (silent-landmine) defect, not an active one in MAF regimes
  — so the headline Score-CD numbers were almost certainly **not** inflated by it. The fix is still
  required: it is the correct conservative, *counted* handling, and non-finite scores are most likely
  on the upcoming stress arms (Cauchy heavy tails, `SignNormal` Fisher degeneracy, NSF flows) where
  `nonfinite_diagnostics` will now flag them rather than silently flatter coverage.
- **Side finding (vindicates "re-run, don't cite the notes"):** the notes' §8.2 xlarge degradation
  (`0.192`) did **not** reproduce — this cell measured `0.0395`. The existing Score-CD sweep numbers
  must be regenerated, not reused.

**Item 2 (ℓ_max refinement + the _BatchCache identity bug), 2026-06-10.**
- Fix: `LikelihoodBasedProcedure` now refines ll_max by Adam ascent from the top-3 grid
  argmax points (elementwise-max with the grid value; clamped; non-finite-ignored), with a
  deterministic shrinking-local-search fallback for non-differentiable log-likelihoods.
  Wired into contains_batch (cached), the 1D and d>1 confidence_set paths; NRE inherits via
  its wrapper. `refine_ll_max=False` preserves the legacy path for ablation.
- **Quantified on a real NLE fit** (2D iid, medium MAF, fresh_batch=False, same flow + eval):
  legacy grid-only `coverage_error_max=0.0700`, mean bias **+0.0211** (over-coverage, the
  predicted direction); refined `0.0395`, mean −0.0153. The published NLE/NRE d>1 baselines
  carry a ~+0.02 grid artifact and must be regenerated.
- **Collateral root-cause fix — `_BatchCache` id-recycling (silent corruption class).** The
  cache keyed on `id(x_obs_batch)`; the chunked engine frees each x-slice before creating the
  next, so recycled ids handed later chunks an earlier chunk's cached values (loud crash on the
  512-vs-464 shape mismatch that exposed it; SILENT corruption when shapes matched). All 8 call
  sites (NPE samples, LF2I/Score-CD t_obs/t_grid, NLE ll_grid) now pass a weakref-validated
  `anchor`: a hit requires the keyed batch to be the same live object. Published numbers were
  likely unaffected (the legacy Coverage path kept one tensor alive across its α-loop), but any
  engine-chunked NPE/NLE/NRE eval — i.e. exactly our upcoming re-runs — would have been corrupted.
  Regression test reproduces the crash scenario end-to-end through the engine.

**Items 3+4, 2026-06-10.**
- #3: `ScoreCDRunner.n_params` now counts the cal variant's MultiQuantileMLP head
  (`calibration_stage`, kind `two_stage_score`), with the run.py dispatch passing
  `d_theta`/`alpha_grid_len` — identical accounting to LF2I-BFF. Softening of the audit's
  worst-case: the method yamls already interpolate `quantile_hidden: ${budget.quantile_hidden}`,
  so swept runs used the budget-tuned 16-wide head (~388 params at d=2, <9% of a medium
  backbone) — the defect was the *reporting*, not an oversized head. No width retune needed.
- #4: `_build_flow` now WARNS at the exact dispatch-trap fall-through (method.flow wins over an
  experiment `/flow:` override; "maf" methods exempt by design), and the index records the
  ground-truth `flow_class_built` from `arch_metadata` next to the intent-only `flow` column —
  the parquet can no longer claim an override took when it didn't. §8.4 arch verification folds
  into the re-run (the original `outputs/` are not present in this workspace).

**Item 5 (CauchyLocScale), 2026-06-10.** The no-sufficient-statistic stress target promoted from
throwaway probe script to committed simulator (`simulators/cauchy_loc_scale.py` +
`configs/target/cauchy_loc_scale.yaml`). θ=(log γ, x₀), n_iid=10, U-box proposal. Oracle pivot:
per-replicate Cauchy-PIT → normal scores → two fixed orthonormal projections — EXACTLY N(0,I₂) at
truth at every θ₀ (docstring states plainly: *a* calibrated pivot for the floor row, NOT an
efficient one — none of fixed dimension can be sufficient here, by PKD). Verified: scipy log_prob
match; per-coord/joint-χ²₂ KS < 0.02 incl. box corners; coords independent for odd n_iid
(Gram-Schmidt); **oracle at the measured floor through `evaluate_coverage` (max|Δ| < 0.03,
3×3 grid × 4 levels, n_per_theta=4000)** — the floor row for the POC table now exists as
reproducible code.

**Items 6+7+8, 2026-06-10.**
- #6: `eval_thetas_edge` was dead config — now fused into the grid every diagnostic consumes
  (`_eval_theta_grid`); engine + index report `coverage_error_{max,mean,p90}` (max is a
  max-order-statistic — inflates with grid size; mean/p90 + the per-(θ₀,α) table are the honest
  companions); and every run with a closed-form `r_star` now writes a MEASURED oracle floor row
  (`oracle_coverage.parquet` + `oracle_coverage_error_*` index columns) on the identical grid/n/α —
  "at the floor" is a verified, grid-matched claim.
- #7: per-run simulator-call accounting in the index: train (fixed-set n_train vs fresh
  steps×batch), calibration (quantile sets; d-aware BFF marginal), inference (live counter on
  Score-CD-rao's per-θ Fisher draws — e.g. 11 grid pts × 4000 = 44k uncounted calls now visible).
  Budget messages relabeled: parameters are the matched quantity, sims are logged.
- #8: POC experiment configs landed (`poc_loc_normal_1d`, `poc_gauss_2d_corr`, `poc_cauchy`,
  `poc_sign_normal` + `configs/target/sign_normal_1d.yaml`): upgraded grids (dense interior + live
  edges; 5×5 product + box corners at d=2), n_per_theta=5000, both `fresh_batch` regimes in the
  sweeper. All four compose-validated; every target exposes `r_star` so the oracle row is automatic.
  Remaining: EXECUTE the POC sweep and rewrite the draft's §6 from the regenerated tables.
  (poc_exp_rate deferred to the run step — it needs the §8.4 on-T reduction wiring decision.)

**Reuse (verified sound, no change):** shared training loop / optimizer / capacity matching;
coverage-engine math (bit-validated); CD-SBI χ²-pivot construction; NPE for Gaussian-posterior
targets; regular simulators' closed-form `r_star`.

## Pilot execution (item 8 EXECUTE), 2026-06-10

**Pilot A — `poc_loc_normal_1d`** (5 methods × {fixed, fresh} × seed 0; ~45 s/run; index
columns label fresh by `sim_calls_total_method` ≈ 2.05M vs 10–46k fixed — the 200× sim cost
of `fresh_batch=true` is now a visible per-row number, not a footnote):

| method | max (fixed) | max (fresh) | mean (both) | note |
|---|---|---|---|---|
| oracle floor | 0.011–0.017 | — | 0.003–0.005 | measured, grid-matched |
| cd_sbi | 0.018–0.024 | 0.013–0.014 | 0.004–0.007 | at floor |
| score_cd_rao | 0.076–0.113 | **0.019–0.030** | 0.006–0.020 | fresh ≈ floor |
| score_cd_cal | 0.073–0.136 | 0.049–0.064 | 0.012–0.032 | |
| lf2i_bff | 0.071–0.073 | 0.083–0.100 | 0.017–0.020 | |
| nle | **0.240–0.264** | 0.243–0.259 | 0.026–0.028 | regime-independent |

Two findings the old 5-pt interior grid could not see:
1. **NLE collapses at the prior boundary** (θ₀ = ±7 live-edge points: ~0.22–0.26 error,
   max-α), in BOTH regimes — a constrained-MLE artifact of the Wilks statistic near the
   support edge, not overfitting. Interior NLE stays at 0.006–0.023. `coverage_error_mean`
   (0.026) vs `_max` (0.26) quantifies how much the max-order-statistic vs the field differ.
2. **Score-CD-rao is robust exactly where NLE breaks** (no constrained MLE in the statistic;
   the score at the edge is still well-defined) but pays interior overfitting wiggles in the
   fixed regime (0.076–0.113 → 0.019–0.030 with fresh batches) — the draft's
   finite-data-brittleness story, now measured on the upgraded grid.

**Perf pathologies found during execution — one shared class: x-invariant quantities
recomputed inside per-θ statistic closures.**
- Score-CD-rao recomputed the Fisher MC (`fisher_n×n_iid` sims) at every distinct θ probe of
  set construction (26 misses/x = 104k sims): fixed by `_FisherGrid` — fit-time Î(θ) on a
  per-dim grid + multilinear interp (`fisher_grid_per_dim=9` default, 0 = exact fallback,
  auto-fallback d>3). §-replication: 69 min → 4.6 min.
- LF2I-BFF recomputed the x-only marginal `log m(x)` at every θ probe: fixed by an
  anchor-validated `_BatchCache` on `_log_marginal` + a one-X-many-θ early branch.
  confidence_set: 25-min-class → 0.046 s/x. Counter `procedure.bff_marginal_evals`.

**Pilot B — `poc_cauchy` raw-vs-asinh (8 runs).** Take-1 killed (pre-fix slowness). Take-2
(22:06) postmortem — three independent failures, all now fixed + committed:
1. **Run-dir collision**: the hydra sweep `subdir` had no `target=` token, so raw and asinh
   runs of the same method silently overwrote each other's directory (two runs interleaved in
   one `run.log`). `target=${target.name}` now leads the subdir pattern.
2. **Chunk misalignment crash** (NLE asinh, SetSize): `_confidence_set_batch_d_gt_1` inferred
   the ll_max broadcast from `theta_flat.shape[0] // B`, but `_chunked_inside` slices at
   arbitrary 4096-row boundaries (B=50 × 200 rays → 4096-row chunk → 50·⌊4096/50⌋ = 4050 ≠
   4096). Crash when non-divisible; SILENT ll_max misalignment when divisible — same bug class
   as the `_BatchCache` id-recycling. Fix: ll_max rides as an extra x-column (aligned by
   construction); the path also gained chunked mesh eval + the `refine_ll_max` polish for
   consistency with the coverage path. Regression: `tests/unit/test_set_batch_chunk_alignment.py`.
3. The background job died silently (~22:21, container-level; not OOM — no kernel kill, 15G
   free), leaving a stale `STATUS=RUNNING`. Lesson: judge progress by run-dir artifacts, not
   STATUS files.
Take-3 relaunched 23:30 post-fix; asinh A/B verdict to be appended.
