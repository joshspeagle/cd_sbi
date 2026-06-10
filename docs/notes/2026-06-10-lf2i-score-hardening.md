# LF2I-Score hardening — experiment-rigor checklist (2026-06-10)

Closing out the rigor issues the three audits found, so the LF2I-Score comparison is
publishable. **No shortcuts** — each item is implemented, tested, and (where it affects a number)
re-run. Env: `/home/user/venv` (clean venv; full stack installs; `334 passed` baseline).

Order is dependency-driven: fix what could invalidate a headline number first, then build the
missing experiments, then the accounting/eval upgrades, then re-run.

| # | Item | Audit | Verdict | Status | Acceptance |
|---|---|---|---|---|---|
| 1 | **Score-CD `nan_to_num` bias** | calib | FIX-FIRST (blocking) | **DONE** | non-finite score → REJECT (+inf, conservative); Fisher rows dropped; counted in `procedure.nonfinite_diagnostics`; unit-tested + quantified on real runs |
| 2 | NLE/NRE `ℓ_max` grid at d>1 | calib | FIX-FIRST | todo | d>1 `ℓ_max` not under-estimated vs a dense reference; deterministic-enough |
| 3 | Score-CD-cal `n_params` undercount | match | UPGRADE+rerun | todo | quantile head counted in `total`; budget re-validated (retune width if needed) |
| 4 | Arch logging (built flow ≠ `cfg.flow.name`) | match | UPGRADE+verify | todo | index logs the *built* `arch_metadata.flow_class`; §8.4 cd_sbi flow verified |
| 5 | **Cauchy loc-scale simulator (NEW)** | targets | NEW | todo | committed `Simulator` + closed-form oracle pivot `r_star`; oracle at floor |
| 6 | Eval upgrade: oracle floor row + grid + metric | targets | UPGRADE | todo | dense interior+edge grid; n_per_theta=5000; mean/p90/max + per-θ₀ vector; measured `r_star` floor row |
| 7 | Sim-cost accounting + relabel | match | UPGRADE | todo | `simulator_calls_total` logged (train+calib+inference Fisher/marginal); headline = "matched *parameter* budget" |
| 8 | `fresh_batch` 2-regime sweep | match | REDO | todo | suite runs at fresh_batch ∈ {false, true}; both reported |
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

**Reuse (verified sound, no change):** shared training loop / optimizer / capacity matching;
coverage-engine math (bit-validated); CD-SBI χ²-pivot construction; NPE for Gaussian-posterior
targets; regular simulators' closed-form `r_star`.
