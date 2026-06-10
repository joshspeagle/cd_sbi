# LF2I-Score hardening — experiment-rigor checklist (2026-06-10)

Closing out the rigor issues the three audits found, so the LF2I-Score comparison is
publishable. **No shortcuts** — each item is implemented, tested, and (where it affects a number)
re-run. Env: `/home/user/venv` (clean venv; full stack installs; `334 passed` baseline).

Order is dependency-driven: fix what could invalidate a headline number first, then build the
missing experiments, then the accounting/eval upgrades, then re-run.

| # | Item | Audit | Verdict | Status | Acceptance |
|---|---|---|---|---|---|
| 1 | **Score-CD `nan_to_num` bias** | calib | FIX-FIRST (blocking) | **in progress** | non-finite score → REJECT (conservative), not accept; counted/exposed; quantified on a real run |
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

**Reuse (verified sound, no change):** shared training loop / optimizer / capacity matching;
coverage-engine math (bit-validated); CD-SBI χ²-pivot construction; NPE for Gaussian-posterior
targets; regular simulators' closed-form `r_star`.
