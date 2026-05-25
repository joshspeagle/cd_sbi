# CD-SBI experiment infrastructure — design spec

**Date:** 2026-05-25
**Status:** Brainstorming output, awaiting user review before plan generation.
**Manuscript baseline:** `cd_sbi_v7.tex` (post-round-3, 47 pages).

---

## 1. Purpose and context

`cd_sbi_v7.tex` develops a calibrated confidence-distribution framework
(CD-SBI) for simulation-based inference and reports four toy experiments
(§8.1–§8.4) whose numbers were produced by ad-hoc code now gone from the
repository. This spec designs the infrastructure that will (i) reproduce
those four experiments under controlled, replicable conditions, (ii)
compare CD-SBI against matched-parameter baselines from the established
SBI literature (NPE, NLE, NRE) and one coverage-targeting alternative
(LF2I), and (iii) provide the forward hooks needed to extend later to
higher-dimensional, real-data, and image/sequence settings without
redesign.

The deliverable of this spec is a layered Python package that supports
the v0 milestone — full §8.1 replication plus a five-method,
four-budget, five-seed baseline sweep — and whose interfaces are
explicitly shaped to absorb later milestones (§8.2–§8.4 replications,
SBI benchmark suite, astronomy inference, image observations,
masked-attention autoregressive flows) as additions rather than
rewrites.

## 2. Scope and non-goals

**In scope for the design (this spec):** the package layout, interface
contracts at every layer boundary, the experiment configuration model,
the on-disk run-directory layout, the testing strategy, the
reproducibility model, and the milestone roadmap with forward-hook
commitments.

**In scope for the v0 build (the first implementation milestone):** §8.1
replication, the CD-SBI / NPE / NLE / NRE / LF2I baseline sweep at
matched parameter budget, and every cross-cutting piece of
infrastructure they require (configs, seeding, diagnostics, run dirs,
GPU/CPU handling, the full test categories).

**Not in scope for v0:** multivariate flows, doubly-monotone form,
ablation flow, image/CNN/attention conditioners, alternative-class
losses from §3.7, sequential variants, coverage-set visualization
beyond per-α tables. Each has a designed-for hook in v0 but no code.

**Not in scope for this spec at all:** the CD-SBI manuscript itself
(the spec consumes the manuscript as authoritative), publication
artifacts beyond `analysis/`-layer paper-table generators, and the
question of which scientific claims to publish (separate decision).

## 3. Architectural approach

A layered Python package (`cdsbi`) with five core layers plus
experiments and analysis surfaces. Each layer has a narrow interface,
is independently testable with mocked dependencies, and admits new
implementations without changes to other layers.

```
cd_sbi/                          # repo root (existing)
├── pyproject.toml               # pip-installable; deps: torch, sbi, hydra-core,
│                                #   numpy, scipy, pandas, pyarrow, pytest
├── src/cdsbi/
│   ├── simulators/              # (θ, X | θ) generators; analytical r*, log p, entropy
│   ├── flows/                   # neural pivot / density estimator; param-count introspection
│   ├── conditioners/            # X → context vector (identity / MLP / CNN / attention)
│   ├── losses/                  # NF-MLE primary; alt-class objectives later
│   ├── methods/                 # CDSBIRunner + NPE/NLE/NRE/LF2I wrappers
│   ├── diagnostics/             # the 5 diagnostics + KS noise floor + coverage curves
│   ├── reproducibility/         # seeding, env capture, atomic run-dir writes
│   ├── device.py                # GPU-with-CPU-fallback resolution
│   ├── experiments/             # Hydra-driven CLI entrypoint + run logic
│   └── analysis/                # run-dir loaders, paper-table & figure generators
├── configs/                     # Hydra config groups (yaml)
├── tests/                       # unit / integration / diagnostics / ablation / intensive
├── notebooks/                   # exploratory + paper-table notebooks (consumers, not orchestrators)
├── outputs/                     # run dirs (created by Hydra at runtime; gitignored)
└── docs/                        # specs, design notes, plans
```

Alternatives considered and rejected:

- **Method-centric layout** (`cdsbi/{cd_sbi,npe,nle,nre}/...`) duplicates
  simulators, diagnostics, and parameter-counting across method dirs;
  the scientific point is cross-method comparison at matched budget, so
  the integration point must be first-class.
- **Notebook-driven library-light layout.** Lower up-front friction but
  fatal for RNG/config snapshotting discipline, testability, and the
  exploding (target × method × architecture × budget × seed) matrix.
  Notebooks belong in `analysis/` as consumers of the framework, not as
  the framework itself.

## 4. Key interfaces

Every cross-layer boundary is a typed Python protocol. The contracts
listed here are the contracts everything else is built around;
implementation classes obey them.

| Layer | Interface | v0 implementation(s) |
|---|---|---|
| `Simulator` | `sample(n, rng) → (θ, X)`; optional `r_star(θ, X)`, `log_prob(X\|θ)`, `entropy_lower_bound()` | `LocationNormal1D` |
| `Flow` | `forward(θ, context) → (r, log_det_jac_X)`; `n_params()`; `monotonicity_guarantees: frozenset[{R1, R2}]` | `AdditiveFlow1D` (= α_a·UMNN(θ) − α_b·UMNN(X)) |
| `Conditioner` | `encode(X) → context_vec` | `Identity` |
| `Loss` | `__call__(flow_out, θ, X) → scalar`; `required_guarantees: frozenset[{R1, R2}]`; `population_lower_bound(simulator) → Optional[float]` | `NFMLELoss` |
| `Method.Runner` | `fit(simulator, config, seed) → TrainedModel`; `n_params() → dict` | `CDSBIRunner`, `NPERunner`, `NLERunner`, `NRERunner`, `LF2IRunner` |
| `Diagnostic` | `__call__(trained, simulator, eval_data) → DiagnosticResult` | All 5 of §7.3 |

Three load-bearing properties of these interfaces:

1. **`Flow.monotonicity_guarantees` is a first-class typed attribute.**
   The training loop checks `loss.required_guarantees ⊆
   flow.monotonicity_guarantees` before fitting. `NFMLELoss` requires
   `{R1, R2}`; the `JointUMNNFlow` ablation case advertises `{R1}` only
   and is refused unless `allow_ablation=True` is set explicitly. This
   turns the §8.4 (R2)-failure mode from a footgun into a deliberate,
   testable property.
2. **`Simulator.r_star` and `Simulator.entropy_lower_bound` are
   optional.** When absent, Diagnostic 1 (pivot RMSE) no-ops and the
   ablation regression test cannot run; this is the correct behavior
   for real-data simulators.
3. **`Method.Runner.n_params()` returns a structured dict**, not an int:
   `{"backbone": ..., "head": ..., "calibration_stage": ..., "total":
   ..., "kind": "flow"|"classifier"|"two_stage"}`. The budget-sweep
   harness drives off `total`; per-experiment paper tables report the
   structured breakdown so that NRE's classifier-vs-flow scaling
   asymmetry and LF2I's two-stage parameter split are honest by
   construction.

## 5. Component design per layer (v0 contents vs forward hooks)

### `simulators/`

- **v0:** `LocationNormal1D` (θ ~ U[−7,7], X ~ N(θ,1), `r*(θ,X) = θ − X`,
  analytic `log_prob`, `entropy_lower_bound = ½ log(2πe)`).
- **Forward hooks (not built v0):** `LocationGaussian2D{iid, correlated}`
  (v1, v2), `ExponentialRate` (v3), `SBIBenchmarkAdapter` wrapping
  `sbibm` (v4), `AstroSimulator` (v7), `ImageSimulator` (v8).
- **Why the v0 protocol is enough:** the `(θ, X)` tensor-shape contract
  and the `r_star` / `log_prob` optionality cover every later case.
  Higher-dimensional `X` is absorbed by the conditioner layer; no
  simulator-protocol change is needed even when `X` is an image.

### `flows/`

- **v0:** `UMNNBlock` (scalar UMNN with softplus + 12-node
  Gauss–Legendre per §7.1), `AdditiveFlow1D` (advertises
  `{R1, R2}`).
- **Forward hooks:** `TriangularAdditiveFlow` (v1, §6.1 form 1),
  `DoublyMonotoneUMNN` (v3, §6.1 form 2), `JointUMNNFlow` (v3, advertises
  `{R1}` only — the ablation form), `MAFAdapter` / `NSFAdapter` (when
  needed to swap flow backbones inside `sbi`'s NPE/NLE),
  `MaskedAttentionFlow` (v8, for §11.5 attention-based autoregressive
  ordering).
- **Why now:** the `monotonicity_guarantees` field plus the explicit
  ablation opt-in convert §8.4's cautionary tale into a reproducible
  experiment instead of a footgun. The flow-adapter pattern is what
  lets NPE/NLE share backbones with CD-SBI for matched-parameter
  comparison.

### `conditioners/`

- **v0:** `Identity` (returns X as-is).
- **Forward hooks:** `MLPConditioner`, `CNNConditioner` (image X),
  `AttentionConditioner` (sequence X and §11.5 masked-attention).
- **Why now:** `Flow.forward(θ, context)` always takes a `context`
  rather than `X` directly, so the v8 CNN drops in with zero changes
  to the flow signature.

### `losses/`

- **v0:** `NFMLELoss`. Requires `{R1, R2}`; exposes
  `population_lower_bound(simulator)` returning the conditional-entropy
  floor when the simulator supplies it (this powers the ablation
  regression test).
- **Forward hooks:** `MarginalPushforwardLoss` (Class 1),
  `MarginalPlusHSICLoss` (Class 2), `StratifiedLoss` (Class 3) —
  the §3.7 alternatives, for v5.

### `methods/`

| Runner | What it trains | Param-count `kind` | v0 status |
|---|---|---|---|
| `CDSBIRunner` | Our `Flow` + `NFMLELoss`; refuses to run on a flow missing R2 unless `allow_ablation=True` | `flow` | v0 |
| `NPERunner` | Wraps `sbi.inference.SNPE_C` with our `Flow` as `density_estimator` | `flow` | v0 |
| `NLERunner` | Wraps `sbi.inference.SNLE_A` with our `Flow` as `density_estimator` | `flow` | v0 |
| `NRERunner` | Wraps `sbi.inference.SNRE_B` with a classifier factory we control (MLP on `[θ, X]`) | `classifier` | v0 |
| `LF2IRunner` | Two-stage: (1) a test statistic `T(θ, X) = log p_NLE(X\|θ) − log p_NLE(X\|θ_ref)` built on the same flow backbone as NLE, (2) a critical-values function `c_α(θ)` trained by pinball-loss quantile regression on a held-out simulation set | `two_stage` | v0 |

LF2I rationale: rolling our own (~150 lines on top of existing layers)
gets us a backbone matched to CD-SBI / NPE / NLE, Hydra-native
configuration, and the same `Method.Runner` interface. Sharing the
first-stage backbone with NLE means at matched flow budget, "LF2I vs
NLE" isolates the calibration mechanism (quantile-regressed critical
values vs none) from the density-estimator choice. WALDO and Box CD
become two more runners later — same protocol.

### `diagnostics/`

- **v0 (all five from §7.3):** `PivotRMSE`, `MarginalPIT`,
  `ConditionalPIT`, `JointMahalanobis`, `Coverage`, plus a
  `ks_noise_floor(N, n_bins)` helper. Each returns a `DiagnosticResult`
  dataclass with raw values and pass/fail status against the
  appropriate (marginal or per-bin) noise floor.
- All five in v0 because they cost essentially nothing once written and
  §8.1 exercises four of them — adding §8.2+ inherits the diagnostic
  infrastructure for free.

### `experiments/` + `configs/`

- **v0 config groups:** `target/loc_normal_1d`, `flow/additive_umnn`,
  `conditioner/identity`, `method/{cd_sbi, npe, nle, nre, lf2i}`,
  `training/adam_3e-3_4k_steps`, `budget/{small, medium, large, xlarge}`
  (~1k / 5k / 25k / 100k params), plus composites
  `experiment/8_1_replication` and `experiment/8_1_baseline_sweep`.
- **CLI:** `python -m cdsbi.experiments.run experiment=... [overrides]`,
  with Hydra `--multirun` for sweeps.
- **Forward hook:** the group structure is the matrix axis — adding
  §8.2 = one new `target/` config + one new `flow/` config + one new
  composite, no code changes.

### `analysis/`

- **v0:** `load_run(path) → dict`, `load_runs(glob) → DataFrame`,
  `paper_table_8_1(df) → DataFrame`.
- **Forward hook:** per-experiment table generators land here as each
  §8 experiment is replicated.

## 6. Data flow

End-to-end for a single run:

```
$ python -m cdsbi.experiments.run experiment=8_1_replication method=cd_sbi seed=0
                       │
                       ▼
  Hydra composes config groups, snapshots resolved YAML to run dir,
  seeds all RNGs (torch, numpy, python random, sbi)
                       │       → ${run_dir}/config.yaml, seeds.json
                       ▼
  Build Simulator, Flow, Conditioner, Method.Runner. Verify
  loss.required_guarantees ⊆ flow.monotonicity_guarantees
  (raises MonotonicityMismatchError unless allow_ablation set)
                       │
                       ▼
  Runner.fit(simulator, train_cfg)
                       │       → ${run_dir}/model.pt, metrics.parquet (per-step),
                       │         env.json (git sha, dirty flag, library versions, device)
                       ▼
  Run all 5 Diagnostics on a held-out evaluation simulation (separate
  RNG stream derived deterministically from run seed)
                       │       → ${run_dir}/diagnostics/*.parquet
                       ▼
  Append summary row to outputs/index.parquet (idempotent on
  (experiment, seed, method, budget) key)
```

Multirun sweep: `python -m cdsbi.experiments.run -m
experiment=8_1_baseline_sweep` expands the composite into 5 methods × 4
budgets × 5 seeds = 100 runs.

## 7. Configuration model

Hydra config groups composed declaratively in `configs/`:

```
configs/
├── config.yaml                          # base; defaults list; run_dir template; device: auto
├── target/                              # {loc_normal_1d, …}
├── flow/                                # {additive_umnn, …}
├── conditioner/                         # {identity, …}
├── method/                              # {cd_sbi, npe, nle, nre, lf2i, …}
├── training/                            # {adam_3e-3_4k_steps, …}
├── budget/                              # {small, medium, large, xlarge}
└── experiment/                          # composites (8_1_replication, 8_1_baseline_sweep, …)
```

`device: auto` at the top level resolves to `cuda` if available else
`cpu`; `device: cpu` and `device: cuda` are valid overrides for
reproducing results across machines.

## 8. Run-directory layout (the reproducibility unit)

```
outputs/
└── 8_1_baseline_sweep/
    └── 2026-05-25_19-23-04/
        └── method=cd_sbi,budget=medium,seed=0/
            ├── config.yaml              # fully-resolved Hydra config
            ├── seeds.json               # all RNG seeds + torch deterministic flags
            ├── env.json                 # git sha, dirty-tree flag, python/torch/sbi versions,
            │                            #   device, num threads, wall-clock start/end
            ├── model.pt                 # state_dict + arch metadata sufficient to rebuild
            ├── metrics.parquet          # per-step: loss, grad_norm, lr, wall_clock
            ├── diagnostics/
            │   ├── pivot_rmse.parquet
            │   ├── marginal_pit.parquet
            │   ├── conditional_pit.parquet  # one row per θ_0 bin
            │   ├── joint_mahalanobis.parquet
            │   └── coverage.parquet         # one row per (θ_0, α)
            └── stdout.log
```

Three reproducibility-critical properties:

1. **Eval seed is deterministically derived from run seed**
   (`eval_seed = hash(run_seed, "eval")`); diagnostics are computed on
   a held-out simulation with its own RNG stream, never on training
   data.
2. **`env.json` captures the dirty-tree flag.** Runs from a dirty
   working tree are allowed but flagged so analysis code can filter.
3. **`outputs/index.parquet` is idempotent on `(experiment, seed,
   method, budget)`.** Re-running a configuration updates that row
   rather than appending; "rerun just the failed seeds" is a normal
   operation.

## 9. Testing strategy

Five test categories. The first four run on every push (≤ 30 s total);
the fifth is opt-in.

| Category | Purpose | Examples |
|---|---|---|
| **Unit** | Per-module correctness with mocked deps | UMNN strictly monotone on a grid; `Identity` conditioner shape-preserving; `n_params` matches `torch` count |
| **Integration** | Composed components, tiny budgets, finish in seconds | 200-step CDSBI on N=50 `LocationNormal1D`; runner produces parseable run-dir |
| **Diagnostics** | Diagnostics give *known* answers on analytical-pivot problems | See subsection below |
| **Ablation** | The §8.4 (R2)-violation smoking gun | See subsection below |
| **Intensive** | Full-budget replication; opt-in via `pytest -m intensive` | §8.1 replication, ~3 min; later: §8.2, §8.3, §8.4 |

### Diagnostic-regression tests (the load-bearing ones)

The diagnostics layer is tested against analytical truth, not against
frozen snapshots. For each diagnostic we use an oracle pivot
`r(θ, X) = r*(θ, X)` and verify pass-rate on a known-correct case and
fail-rate on a deliberately miscalibrated case:

| Test | Setup | Expected behavior |
|---|---|---|
| `test_marginal_pit_oracle_passes` | Apply marginal PIT to `r*` on N=5000 `LocationNormal1D` | KS ≤ noise floor (~0.023) in ≥ 99/100 seeds |
| `test_marginal_pit_catches_miscalibration` | Same setup, pivot `0.5·(θ − X)` (under-dispersed ×2) | KS > floor in ≥ 99/100 seeds |
| `test_coverage_oracle_exact` | Coverage on `r*` at α ∈ {0.5, 0.68, 0.9, 0.95} | Empirical coverage within MC error of nominal at every α |
| `test_coverage_catches_mismatch` | Same on under-dispersed pivot | Empirical coverage < nominal by detectable margin |
| `test_joint_mahalanobis_catches_correlation` | 2D pivot, components individually N(0,1), correlation 0.5 | Joint Mahalanobis KS > floor; component-wise KS does not |
| `test_ks_noise_floor_matches_theory` | `1.628/√N` vs `ks_noise_floor()` | Match to 4 decimals |

### The (R2) ablation regression test

The unique test that pins down §8.4's conclusion as a structural property:

```
test_r2_ablation_loss_below_entropy_floor:
    sim = ExponentialRate(theta_range=[0.3, 3.0], n=5)
    flow = JointUMNNFlow(...)        # advertises monotonicity_guarantees = {R1}
    loss = NFMLELoss()                # required_guarantees = {R1, R2}

    # 1. The safety check must fire by default
    with pytest.raises(MonotonicityMismatchError):
        CDSBIRunner(flow, loss).fit(sim, ...)

    # 2. With explicit opt-in, training proceeds
    trained = CDSBIRunner(flow, loss, allow_ablation=True).fit(sim, cfg_short)

    # 3. The smoking gun: final loss < conditional-entropy lower bound
    assert trained.final_loss < sim.entropy_lower_bound() - 0.05
```

If a future refactor accidentally makes `NFMLELoss` tolerant of a
non-monotone flow, this test fails. Equivalently, if a refactor breaks
Theorem 3.2's `loss ≥ H(X|θ)` bound anywhere, the inequality becomes
violable elsewhere and we catch it here.

### Determinism contract

- `pytest tests/test_determinism.py` runs the same `(seed, config)`
  twice and asserts every diagnostic value matches to `float32`
  precision.
- `torch.use_deterministic_algorithms(True)` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` set unconditionally.
- Documented: determinism applies *within a (machine, torch version,
  device) tuple*; cross-device runs differ in float-rounding only.

### What we explicitly don't test

- **Numerical bit-equivalence to manuscript Table values.** The
  intensive category checks tolerance, not bit-identity. The
  manuscript numbers were produced by a different code path; insisting
  on equivalence would fix arbitrary implementation choices into the
  test suite.
- **`sbi` package internals.** We test our wrappers' input/output
  contract; we trust the package's training loop.
- **GPU-specific paths in v0.** CPU is the v0-target hardware; GPU
  determinism gets its own test pass when v6+ activates it
  meaningfully.

## 10. Reproducibility model

A small `cdsbi.reproducibility` module:

- `seed_everything(seed: int) → SeededRNGs` — seeds `random`, `numpy`,
  `torch`, `torch.cuda`, the `sbi` package's internal RNG; returns a
  namespace of named RNG streams (`train`, `eval`, `init`)
  deterministically derived from the master seed via `hashlib`.
- `capture_env() → dict` — git sha, dirty flag, library versions,
  hardware, device. Written to `env.json` on every run.
- `RunDir.write_atomic(path, payload)` — writes via `path.tmp` +
  rename so a mid-write crash leaves the prior version intact and
  never produces a partially-written parquet.

`cdsbi.device.get_device() → torch.device` returns `cuda` if available
else `cpu`, resolved once at run start and recorded in `env.json`.
Hydra `device: cpu` and `device: cuda` overrides force one or the
other. Diagnostics computations stay on whatever device the model is
on, with one `.cpu()` at the parquet-write boundary; KS tests run on
`numpy`/`scipy` post-transfer (small-N operations where GPU is
irrelevant).

## 11. v0 milestone definition

### Deliverable

One CLI invocation produces a publishable comparison:

```
python -m cdsbi.experiments.run -m experiment=8_1_baseline_sweep
```

→ 5 methods × 4 budgets × 5 seeds = 100 runs; analysis-layer call
`analysis.paper_table_8_1(load_runs(...))` returns the comparison
DataFrame.

### Done criteria

1. The CLI above completes deterministically (same seeds → identical
   outputs to `float32`).
2. CD-SBI lands within tolerance of §8.1's manuscript numbers (RMSE
   ~0.03, marginal KS ~0.008, coverage error < 0.01).
3. All five methods report `n_params()['total']` within ±10% of each
   budget target.
4. Full default `pytest` suite (unit + integration + diagnostics +
   ablation) passes in under 30 s.
5. `pytest -m intensive` runs the §8.1 replication and matches
   manuscript numbers within tolerance.

## 12. Milestone roadmap

Illustrative — not part of v0 commitment.

| Milestone | New code | New configs | Adds |
|---|---|---|---|
| **v0** §8.1 sweep | UMNN, AdditiveFlow1D, Identity conditioner, 5 runners, 5 diagnostics, Hydra/run-dir/seeding/device plumbing, all test categories | `target/loc_normal_1d`, `flow/additive_umnn`, `method/{cd_sbi, npe, nle, nre, lf2i}`, `experiment/8_1_*` | First comparison |
| **v1** §8.2 | `TriangularAdditiveFlow` | `target/loc_gauss_2d_iid`, `flow/triangular_additive`, `experiment/8_2_*` | Multivariate triangular flow; joint Mahalanobis becomes load-bearing |
| **v2** §8.3 | (no new flow) | `target/loc_gauss_2d_corr`, `experiment/8_3_*` | KR-uniqueness empirical evidence |
| **v3** §8.4 + ablation | `DoublyMonotoneUMNN`, `JointUMNNFlow` | `target/exp_rate`, `flow/{doubly_monotone, joint_umnn_ablation}`, `experiment/8_4_*` | (R2) smoking-gun as a published result |
| **v4** SBI benchmark | `SBIBenchmarkAdapter` | `target/sbibm_*` | Community leaderboards |
| **v5** §3.7 alt-loss | `MarginalPushforwardLoss`, `MarginalPlusHSICLoss`, `StratifiedLoss` | `loss/*`, `experiment/3_7_*` | Empirical Class 1–4 vs Class 5 demonstration |
| **v6** synthetic high-d | (mostly config; possibly `MLPConditioner`) | `target/loc_gauss_kd`, `experiment/scaling_*` | §11.5 scaling claim |
| **v7** astro inference | `AstroSimulator` (likely wraps a domain code) | `target/cosmo_*` or `target/gw_*` | Real-data deliverable |
| **v8** image / sequence | `CNNConditioner`, `AttentionConditioner`, possibly `MaskedAttentionFlow` | `conditioner/cnn`, `conditioner/attention`, `flow/masked_attention` | §11.7 territory + §11.5 attention variant |

## 13. Forward-hook commitments

These are *not built* in v0 but must be designed-for-in-v0 to avoid
later rework:

1. **The conditioner seam.** Even with `Identity` as the only v0
   implementation, `Flow.forward(θ, context)` always takes `context`
   rather than `X` directly. v8's CNN drops in with zero changes to
   the flow signature.
2. **`monotonicity_guarantees` as a first-class flow attribute.** v0's
   `AdditiveFlow1D` is the only flow that sets it, but the field
   exists on the base class and `NFMLELoss` inspects it. v3's ablation
   flow drops in and behaves correctly (refuses to train without
   `allow_ablation`) with no loss-layer changes.
3. **`Simulator.entropy_lower_bound()` as optional protocol.** Returns
   `None` when unknown; powers the ablation regression test for v3 and
   any future loss-floor diagnostic. Adding it now costs one line on
   `LocationNormal1D` and unblocks v3 cleanly.

## 14. Non-goals (explicit, to prevent scope creep)

- No multivariate flow code in v0 (lands v1)
- No doubly-monotone or ablation flow in v0 (lands v3)
- No image / sequence support in v0 (lands v8)
- No alt-class loss objectives in v0 (lands v5)
- No sequential variants (open problem §11.6; no scheduled milestone)
- No coverage-set visualization beyond per-α empirical-coverage tables
  in v0 (engineering question §11.8; lands as analysis-layer work when
  v3 makes it relevant)
- No cross-machine bit-identity reproducibility (within-machine
  determinism only)
