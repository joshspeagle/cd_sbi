# CD-SBI experiment infrastructure — design spec

**Date:** 2026-05-25 (revised same-day post-review)
**Status:** Brainstorming output, awaiting user review before plan generation.
**Manuscript baseline:** `cd_sbi_v7.tex` (post-round-3, 47 pages).

---

## 1. Purpose and context

`cd_sbi_v7.tex` develops a calibrated confidence-distribution framework
(CD-SBI) for simulation-based inference and reports four toy experiments
(§8.1–§8.4) whose numbers were produced by ad-hoc code *outside this
repository*. This spec designs the infrastructure that will (i)
reproduce those four experiments under controlled, replicable
conditions, (ii) compare CD-SBI against matched-parameter baselines
from the established SBI literature (NPE, NLE, NRE) and one
coverage-targeting alternative (LF2I), and (iii) provide the forward
hooks needed to extend later to higher-dimensional, real-data, and
image/sequence settings without redesign.

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
infrastructure they require (configs, seeding, diagnostics including
1D confidence-set computation, run dirs, GPU/CPU handling, the full
test categories).

**Not in scope for v0:** multivariate flows, doubly-monotone form,
ablation flow at §8.4's exponential-rate scale (the *1D* ablation
analog *is* in v0; see §5 and §9), image/CNN/attention conditioners,
alternative-class losses from §3.7, sequential variants, multivariate
confidence-set computation. Each has a designed-for hook in v0 but no
code.

**Not in scope for this spec at all:** the CD-SBI manuscript itself
(the spec consumes the manuscript as authoritative); publication
artifacts beyond what `analysis/` produces (`analysis/` produces (i)
pandas DataFrames for paper tables, (ii) matplotlib figures saved to
PDF, (iii) optional LaTeX-table emission via
`pandas.DataFrame.to_latex()` — nothing beyond that); and the question
of which scientific claims to publish (separate decision).

## 3. Architectural approach

A layered Python package (`cdsbi`) with six core layers plus
experiments and analysis surfaces. Each layer has a narrow interface,
is independently testable with mocked dependencies, and admits new
implementations without changes to other layers.

```
cd_sbi/                          # repo root (existing)
├── .gitignore                   # adds outputs/** as a v0 setup task
├── pyproject.toml               # pip-installable, src-layout; deps: torch, sbi, nflows,
│                                #   hydra-core, numpy, scipy, pandas, pyarrow, pytest
├── src/cdsbi/
│   ├── simulators/              # (θ, X | θ) generators; analytical r*, log p, entropy
│   ├── flows/                   # neural pivot / density estimator; param-count introspection
│   ├── conditioners/            # X → context vector (identity / MLP / CNN / attention)
│   ├── losses/                  # NF-MLE primary; alt-class objectives later
│   ├── methods/                 # CDSBIRunner + NPE/NLE/NRE/LF2I wrappers; each returns
│   │                            #   a ConfidenceProcedure
│   ├── confidence_set/          # 1D root-finder + HPD extractor; ConfidenceSet datatype
│   ├── diagnostics/             # 4 diagnostics in v0 (1, 2, 3 consume a pivot;
│   │                            #   5 consumes any ConfidenceProcedure);
│   │                            #   JointMahalanobis (diagnostic 4) lands v1
│   ├── reproducibility/         # seeding, env capture, atomic run-dir writes
│   ├── device.py                # GPU-with-CPU-fallback resolution
│   ├── experiments/             # Hydra-driven CLI entrypoint + run logic
│   └── analysis/                # run-dir loaders, paper-table & figure generators
├── configs/                     # Hydra config groups (yaml)
├── tests/                       # unit / integration / diagnostics / intensive (ablation lands v3)
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

Every cross-layer boundary is a typed Python protocol.

### 4.1 Cross-method interface — `ConfidenceProcedure`

The cross-method comparison axis is **coverage** (the inferentially
primary diagnostic per §7.3 of the manuscript). Coverage requires
**confidence sets**. Pivots are one way to produce confidence sets;
LF2I has a different way; NPE/NLE/NRE have yet other ways. The
abstraction that unifies them is `ConfidenceProcedure`:

```python
class ConfidenceProcedure(Protocol):
    """What every runner's fit() returns. Produces a confidence set
       at observed X and confidence level α."""
    def confidence_set(self, X_obs, alpha) -> ConfidenceSet: ...

class PivotBasedProcedure(ConfidenceProcedure):           # CDSBI
    def pivot(self, theta, X) -> Tensor: ...
    # default confidence_set: chi-square inversion + 1D root-find

class CriticalValueProcedure(ConfidenceProcedure):        # LF2I (and later WALDO)
    def test_statistic(self, theta, X) -> Tensor: ...
    def critical_value(self, theta, alpha) -> Tensor: ...
    # default confidence_set: {θ : T(θ,X) ≤ c_α(θ)} via 1D root-find

class PosteriorBasedProcedure(ConfidenceProcedure):       # NPE
    def posterior_samples(self, X_obs, n) -> Tensor: ...
    # default confidence_set: highest-posterior-density (HPD) region

class LikelihoodBasedProcedure(ConfidenceProcedure):      # NLE
    def log_likelihood(self, theta, X) -> Tensor: ...
    # default confidence_set: Wilks/likelihood-ratio inversion + 1D root-find

class RatioBasedProcedure(ConfidenceProcedure):           # NRE
    def log_ratio(self, theta, X) -> Tensor: ...
    # default confidence_set: ratio thresholding + 1D root-find
```

The `confidence_set/` module provides the 1D root-finder and HPD
extractor (~150 lines). In 1D these are trivial (bisection on a
continuous monotone function — and pivots are monotone by
construction; likelihood/ratio surfaces in regular problems are too).
Multivariate `confidence_set` lands when v1 introduces multivariate
flows.

### 4.2 Per-layer interfaces

| Layer | Interface | v0 implementation(s) |
|---|---|---|
| `Simulator` | `sample(n, rng) → (θ, X)`; optional `r_star(θ, X)`, `log_prob(X\|θ)`, `entropy_lower_bound()` | `LocationNormal1D` (with closed-form entropy `½ log(2πe)`) |
| `Flow` | `forward(θ, context) → (r, log_det_jac_input)`; `n_params()`; `monotonicity_guarantees: frozenset[{R1, R2}]` | `AdditiveFlow1D` (advertises `{R1, R2}`), `MAFAdapter` (advertises `frozenset()`, used by NPE/NLE/LF2I baselines) |
| `Conditioner` | `encode(X) → (context_vec, log_det_jac_input_contribution)` | `Identity` (returns `(X, 0)`) |
| `Loss` | `__call__(flow_out, θ, X) → scalar`; `required_guarantees: frozenset[{R1, R2}]`; `population_lower_bound(simulator) → Optional[float]` | `NFMLELoss` (requires `{R1, R2}`) |
| `Method.Runner` | `__init__(..., allow_ablation: bool = False)`; `fit(simulator, config, seed) → TrainedModel`; `n_params() → dict` | `CDSBIRunner`, `NPERunner`, `NLERunner`, `NRERunner`, `LF2IRunner` |
| `Diagnostic` | `__call__(trained, simulator, eval_data) → DiagnosticResult` | All 5 of §7.3 (Diagnostics 1–4 consume a pivot; Diagnostic 5 consumes any `ConfidenceProcedure`) |

Three load-bearing properties of these interfaces:

1. **`Flow.monotonicity_guarantees` is a first-class typed attribute.**
   The training loop checks `loss.required_guarantees ⊆
   flow.monotonicity_guarantees` before fitting. `NFMLELoss` requires
   `{R1, R2}`. Any flow that doesn't advertise `R2` (e.g., the
   v3-landing `JointUMNNFlow`) is refused unless `allow_ablation=True`
   is set on the runner constructor (also exposed in
   `method/cd_sbi.yaml`'s `allow_ablation` field, defaults `false`).
   `MAFAdapter` advertises `frozenset()`; NPE/NLE/LF2I runners do not
   use `NFMLELoss` and the check passes vacuously. In v0 no flow
   triggers the refusal path (the only `NFMLELoss`-using flow,
   `AdditiveFlow1D`, advertises both guarantees) — the code path is
   in place but unexercised by tests until v3.
2. **`Simulator.r_star` and `Simulator.entropy_lower_bound` are
   optional.** When absent, Diagnostic 1 (pivot RMSE) and the ablation
   regression test no-op gracefully; this is the correct behavior for
   real-data simulators.
3. **`Method.Runner.n_params()` returns a structured dict**, not an
   int: `{"backbone": ..., "head": ..., "calibration_stage": ...,
   "total": ..., "kind": "flow"|"classifier"|"two_stage"}`. Convention,
   for every `kind`, **every field is always populated** (zero when not
   applicable):
   - `kind='flow'` (CDSBI, NPE, NLE): `backbone` = the trained flow
     weights; `head` = scalar scale/embedding parameters (e.g.,
     CDSBI's two learnable `α` scalars); `calibration_stage = 0`.
   - `kind='classifier'` (NRE): `head` = the classifier MLP;
     `backbone = 0`; `calibration_stage = 0`.
   - `kind='two_stage'` (LF2I): `backbone` = test-statistic flow;
     `calibration_stage` = quantile-regression head; `head = 0`.

### 4.3 Return-type schemas

```python
@dataclass
class TrainedModel:
    procedure: ConfidenceProcedure        # the cross-method handle
    state_dict: dict
    final_loss: float
    n_steps: int
    wall_clock_sec: float
    arch_metadata: dict                   # enough to rebuild via factory

@dataclass
class DiagnosticResult:
    name: str
    value: float | pd.Series              # scalar or per-bin
    passed: bool
    noise_floor: float
    n_samples: int
    meta: dict                            # diagnostic-specific extras

@dataclass
class ConfidenceSet:
    contains: Callable[[theta], bool]
    boundary_repr: Tensor                 # 1D: shape (2,); higher d: boundary samples
    alpha: float
```

### 4.4 Matched-budget rule

The budget sweep is the central confounder control. The rule is:

A `budget/*.yaml` config sets a single integer `target_params`. Each
runner exposes `build_from_budget(target_params)` which enumerates a
discrete set of width candidates (e.g., `widths = [4, 8, 12, 16, …,
256]`), counts `n_params()` for each (no NN training involved), and
picks the candidate whose backbone params are closest to
`target_params`. Soft criterion: within ±10% is "matched";
between ±10% and ±15% is accepted with a warning logged into the
run dir; outside ±15% raises `BudgetUnreachableError`. Enumeration
is more robust than bisection on an integer-domain step function,
deterministic, and runs in <1 ms.

| Method | What `target_params` matches | What gets reported separately |
|---|---|---|
| CDSBI | `n_params['backbone']` (the flow weights only) | `n_params['head']` (the two `α` scalars) — reported as its own column for symmetry with LF2I |
| NPE / NLE | `n_params['backbone']` (the flow weights only) | `n_params['head']` (MAF scale/embedding params, typically a few dozen) |
| NRE | `n_params['head']` (the classifier) — same target so NRE has "equal total params, spent in classifier shape" | — |
| LF2I | `n_params['backbone']` (the first-stage flow, matched to NLE) | `n_params['calibration_stage']` (quantile-regression head) reported separately as its own column |

The rule is: **all flow-kind methods match on `backbone` only**, with
their non-backbone parameters (CDSBI `α` scalars; MAF scale/embed)
reported separately so they're never hidden in the matched-budget
total. This is symmetric with LF2I's calibration head.

Paper-table footnote convention:

> Budget matched on flow-backbone (CDSBI, NPE, NLE, LF2I-stage-1) or
> classifier (NRE); non-backbone parameters reported separately per
> method.

## 5. Component design per layer (v0 contents vs forward hooks)

### `simulators/`

- **v0:** `LocationNormal1D` (θ ~ U[−7,7], X ~ N(θ,1), `r*(θ,X) = θ − X`,
  analytic `log_prob`, `entropy_lower_bound = ½ log(2πe) ≈ 1.4189`).
- **Forward hooks (not built v0):** `LocationGaussian2D{iid, correlated}`
  (v1, v2), `ExponentialRate` (v3), `SBIBenchmarkAdapter` wrapping
  `sbibm` (v4), `AstroSimulator` (v7), `ImageSimulator` (v8).

### `flows/`

- **v0:**
  - `UMNNBlock` — scalar UMNN with softplus + 12-node Gauss–Legendre
    per §7.1
  - `AdditiveFlow1D` — `α_a · UMNN(θ) − α_b · UMNN(X)`; advertises
    `{R1, R2}`
  - `MAFAdapter` — wraps `nflows.flows.MaskedAutoregressiveFlow`
    behind the `Flow` protocol; advertises `frozenset()`; used as the
    natural backbone for NPE, NLE, and LF2I stage 1
- **Forward hooks:** `TriangularAdditiveFlow` (v1, §6.1 form 1),
  `DoublyMonotoneUMNN` (v3, §6.1 form 2), `JointUMNNFlow` (v3 — the
  §8.4 sufficient-statistic ablation form; powers the v3 trained
  (R2)-ablation experiment), `JointUMNN1DFlow` (v3 — a 1D analog
  used in a direct-construction mechanism test alongside the §8.4
  trained test), `NSFAdapter` (v1+ if NSF becomes desirable as a
  baseline backbone), `MaskedAttentionFlow` (v8, for §11.5
  attention-based autoregressive ordering).

### `conditioners/`

- **v0:** `Identity` (returns `(X, log_det_jac_input_contribution=0)`).
- **Forward hooks:** `MLPConditioner`, `CNNConditioner` (image X),
  `AttentionConditioner` (sequence X and §11.5 masked-attention).
- **v3 specifically:** when the doubly-monotone form lands, its
  conditioner does an X→T (sufficient-statistic) reduction and
  contributes `log|∂T/∂X|` through `log_det_jac_input_contribution`.
  v0's `Identity` returns 0; the `Flow.forward` signature is already
  named `log_det_jac_input` to accommodate this without later rename.

### `losses/`

- **v0:** `NFMLELoss`. Requires `{R1, R2}`; exposes
  `population_lower_bound(simulator)` returning the conditional-entropy
  floor when the simulator supplies it.
- **Forward hooks:** `MarginalPushforwardLoss` (Class 1),
  `MarginalPlusHSICLoss` (Class 2), `StratifiedLoss` (Class 3) — the
  §3.7 alternatives, for v5.

### `methods/`

| Runner | What it trains | `procedure` returned | Param-count `kind` | v0 status |
|---|---|---|---|---|
| `CDSBIRunner` | `AdditiveFlow1D` + `NFMLELoss` (refuses to run on flow missing R2 unless `allow_ablation=True`) | `PivotBasedProcedure` | `flow` | v0 |
| `NPERunner` | Wraps `sbi.inference.SNPE_C` with `num_rounds=1` (amortized) and `MAFAdapter` as `density_estimator` | `PosteriorBasedProcedure` | `flow` | v0 |
| `NLERunner` | Wraps `sbi.inference.SNLE_A` with `num_rounds=1` and `MAFAdapter` as `density_estimator` | `LikelihoodBasedProcedure` | `flow` | v0 |
| `NRERunner` | Wraps `sbi.inference.SNRE_B` with `num_rounds=1` and a classifier factory we control (MLP on `[θ, X]`) | `RatioBasedProcedure` | `classifier` | v0 |
| `LF2IRunner` | Two-stage: (1) test statistic `T(θ, X) = log p_NLE(X\|θ) − log p_NLE(X\|θ_ref)` built on `MAFAdapter` (same backbone as NLE), with `θ_ref` defaulting to prior median (configurable via `method/lf2i.yaml`'s `theta_ref` field); (2) critical-values `c_α(θ)` trained by pinball-loss quantile regression on a held-out simulation set | `CriticalValueProcedure` | `two_stage` | v0 |

Notes:

- `sbi.inference.SNPE_C / SNLE_A / SNRE_B` are the *sequential*
  variants; v0 passes `num_rounds=1` to make them effectively
  amortized NPE/NLE/NRE. Documented loudly in each runner module.
- LF2I rationale: rolling our own (~150 lines on top of existing layers)
  gets us a backbone matched to CDSBI/NPE/NLE, Hydra-native
  configuration, and the same `Method.Runner` interface. Sharing the
  first-stage backbone with NLE means at matched flow budget, "LF2I vs
  NLE" isolates the calibration mechanism (quantile-regressed critical
  values vs none) from the density-estimator choice. WALDO and Box CD
  become two more runners later — same protocol.

### `confidence_set/`

- **v0:** 1D bisection root-finder, 1D HPD extractor (`alpha`-quantile
  thresholding of sampled posteriors), `ConfidenceSet` datatype. Used
  by the `confidence_set()` default implementations on every
  `ConfidenceProcedure` subtype.
- **Forward hooks:** multivariate root-finding via the §11.8 approaches
  (autoregressive boundary inversion, HMC/SMC, pushforward sampling) —
  lands in v1+ as multivariate flows arrive.

### `diagnostics/`

- **v0 (four of five from §7.3):**
  - **Diagnostics 1–3** (`PivotRMSE`, `MarginalPIT`, `ConditionalPIT`)
    — consume a *pivot*; apply only to `PivotBasedProcedure` (CDSBI).
    Skip gracefully on other procedures. The manuscript labels these
    "interpretability checks" — they diagnose *where* a calibration
    failure lives inside CDSBI.
  - **Diagnostic 5** (`Coverage`) — consumes any `ConfidenceProcedure`;
    the cross-method comparison axis.
  - `ks_noise_floor(N, n_bins)` helper. `ConditionalPIT(n_bins=k)`
    takes `k` explicitly and computes its per-bin floor as
    `1.628/√(N/k)`.
- **Forward hook (lands v1):** `JointMahalanobis` (Diagnostic 4).
  Degenerate in 1D (reduces to `r²` ~ χ²_1, same as the squared
  marginal-PIT residual); only meaningful when v1 introduces
  multivariate flows. Shipped together with its regression test
  (`test_joint_mahalanobis_catches_correlation`) at v1.

### `experiments/` + `configs/`

- **v0 config groups:** `target/loc_normal_1d`, `flow/{additive_umnn,
  maf}`, `conditioner/identity`, `method/{cd_sbi, npe, nle, nre,
  lf2i}`, `training/adam_3e-3_4k_steps`, `budget/{small, medium,
  large, xlarge}` (~1k / 5k / 25k / 100k `target_params`), plus
  composites `experiment/8_1_replication` and
  `experiment/8_1_baseline_sweep`.
- **CLI:** `python -m cdsbi.experiments.run experiment=...
  [overrides]`, with Hydra `--multirun` for sweeps.
- **`batch_size` lives in `training/`** (training hyperparameter);
  **`n_eval` and `eval_thetas` live in `experiment/`** (eval-data
  design is per-experiment).

### `analysis/`

- **v0:** `load_run(path) → dict`, `load_runs(glob) → DataFrame` (globs
  per-run parquet rows and concatenates), `paper_table_8_1(df) →
  DataFrame`.
- **Outputs:** pandas DataFrames for paper tables, matplotlib figures
  to PDF, optional LaTeX-table emission via
  `pandas.DataFrame.to_latex()`.

## 6. Data flow

End-to-end for a single run:

```
$ python -m cdsbi.experiments.run experiment=8_1_replication method=cd_sbi seed=0
                       │
                       ▼
  Hydra composes config groups, snapshots resolved YAML to run dir,
  computes config_hash = sha256(canonical_yaml(resolved_config)),
  seeds all RNGs (torch, numpy, python random; sbi gets explicit
  per-call seeds — no reliance on sbi global state)
                       │       → ${run_dir}/config.yaml, seeds.json
                       │       → ${run_dir}/STATUS = "RUNNING"
                       ▼
  Build Simulator, Flow, Conditioner, Method.Runner via
  build_from_budget(target_params). Verify
  loss.required_guarantees ⊆ flow.monotonicity_guarantees
  (raises MonotonicityMismatchError unless allow_ablation set)
                       │
                       ▼
  Runner.fit(simulator, train_cfg) → TrainedModel
                       │       → ${run_dir}/model.pt, metrics.parquet (per-step),
                       │         env.json (git sha, dirty flag, library versions, device)
                       ▼
  Run diagnostics: pivot-based (1, 2, 3 in v0; 4 lands v1) only on PivotBasedProcedure;
  Coverage (5) on the procedure regardless of kind. Held-out
  evaluation simulation, separate RNG stream derived deterministically
  from run seed.
                       │       → ${run_dir}/diagnostics/*.parquet
                       ▼
  Write per-run summary row to ${run_dir}/index_row.parquet
  (one parquet per run; analysis layer globs and concatenates at
  read time — no central index file, no parallel-write race)
                       │       → ${run_dir}/STATUS = "OK"
                       ▼
  (on exception): STATUS = "FAILED", stack trace appended to stdout.log
```

Multirun sweep: `python -m cdsbi.experiments.run -m
experiment=8_1_baseline_sweep` expands the composite into 5 methods × 4
budgets × 5 seeds = 100 runs. Rerun behavior: `--multirun` skips run
dirs with `STATUS=OK`; partial dirs (`STATUS=RUNNING` or `FAILED`) are
recomputed.

## 7. Configuration model

Hydra config groups composed declaratively in `configs/`:

```
configs/
├── config.yaml                          # base; defaults list; run_dir template; device: auto
├── target/                              # {loc_normal_1d, …}
├── flow/                                # {additive_umnn, maf, …}
├── conditioner/                         # {identity, …}
├── method/                              # {cd_sbi, npe, nle, nre, lf2i, …}
├── training/                            # {adam_3e-3_4k_steps, …}; batch_size lives here
├── budget/                              # {small, medium, large, xlarge}; sets target_params
└── experiment/                          # composites; n_eval and eval_thetas live here
```

`device: auto` at the top level resolves to `cuda` if available else
`cpu`; `device: cpu` and `device: cuda` are valid overrides.

## 8. Run-directory layout (the reproducibility unit)

### 8.1 Layout

```
outputs/
└── 8_1_baseline_sweep/
    └── 2026-05-25_19-23-04/
        └── method=cd_sbi,budget=medium,seed=0/
            ├── STATUS                   # RUNNING | OK | FAILED — written atomically
            ├── config.yaml              # fully-resolved Hydra config
            ├── seeds.json               # all RNG seeds + torch deterministic flags
            ├── env.json                 # git sha, dirty-tree flag, python/torch/sbi versions,
            │                            #   device, num threads, wall-clock start/end
            ├── model.pt                 # state_dict + arch_metadata sufficient to rebuild
            ├── metrics.parquet          # per-step: loss, grad_norm, lr, wall_clock
            ├── diagnostics/
            │   ├── pivot_rmse.parquet           # only present for PivotBasedProcedure runs
            │   ├── marginal_pit.parquet         # ditto
            │   ├── conditional_pit.parquet      # ditto; one row per θ_0 bin
            │   ├── joint_mahalanobis.parquet    # ditto
            │   └── coverage.parquet             # always present; one row per (θ_0, α)
            ├── index_row.parquet        # one-row parquet: this run's entry; aggregated
            │                            #   at read time by analysis.load_runs()
            └── stdout.log
```

### 8.2 Parquet schemas

Validated at write time by a tiny `schema_check(df, schema)` helper.

`metrics.parquet`:

| column | dtype | description |
|---|---|---|
| `step` | int64 | training step index |
| `loss` | float64 | per-step loss |
| `grad_norm` | float64 | post-clipping gradient norm |
| `lr` | float64 | current learning rate |
| `wall_clock_sec` | float64 | seconds since training start |

`diagnostics/coverage.parquet`:

| column | dtype |
|---|---|
| `theta_0_0` | float64 |
| `alpha` | float64 |
| `nominal` | float64 |
| `empirical` | float64 |
| `n_eval` | int64 |
| `passed` | bool |
| `tolerance` | float64 |

In v1+, additional coordinate columns are added as `theta_0_1`,
`theta_0_2`, … — one column per parameter dimension. This keeps the
schema readable across milestones (no dtype switching on a single
column).

`diagnostics/conditional_pit.parquet`:

| column | dtype |
|---|---|
| `theta_0_bin` | int64 |
| `theta_0_center_0` | float64 (v1+ adds `theta_0_center_1`, …) |
| `ks` | float64 |
| `per_bin_noise_floor` | float64 |
| `n_per_bin` | int64 |
| `passed` | bool |

Similar minimal schemas for `marginal_pit.parquet` and
`pivot_rmse.parquet`. (`joint_mahalanobis.parquet` lands at v1 with
its diagnostic.)

`index_row.parquet` — one-row summary; aggregated to a DataFrame by
`analysis.load_runs()`:

| column | dtype | description |
|---|---|---|
| `config_hash` | str | sha256 of resolved YAML; **primary key** |
| `experiment` | str | composite name |
| `method` | str | runner name |
| `flow` | str | flow class name |
| `target` | str | simulator name |
| `budget_name` | str | budget config name |
| `target_params` | int64 | budget target |
| `actual_params_total` | int64 | runner.n_params()['total'] |
| `actual_params_kind` | str | flow / classifier / two_stage |
| `seed` | int64 | run seed |
| `device` | str | cpu / cuda |
| `git_sha` | str | repo HEAD at run start |
| `dirty_tree` | bool | working-tree dirty at run start |
| `final_loss` | float64 | |
| `wall_clock_sec` | float64 | |
| `coverage_error_max` | float64 | max abs error across (θ_0, α) — quick comparison column |
| `marginal_ks` | float64 | |
| `pivot_rmse` | float64 (nullable) | only for PivotBasedProcedure |

### 8.3 Reproducibility-critical properties

1. **Eval seed is deterministically derived from run seed**
   (`eval_seed = hash(run_seed, "eval")`); diagnostics are computed on
   a held-out simulation with its own RNG stream, never on training
   data.
2. **`env.json` captures the dirty-tree flag.** Runs from a dirty
   working tree are allowed but flagged so analysis code can filter.
3. **Primary key for "this exact run" is `config_hash`**, not the
   four named axes. Any change to any config value (training lr, flow
   width, eval N) produces a new hash → new run dir → new row. No
   silent collisions.

## 9. Testing strategy

Four test categories in v0. The first three run on every push
(≤ 30 s total); the fourth is opt-in via `pytest -m intensive`.

| Category | Purpose | v0 examples |
|---|---|---|
| **Unit** | Per-module correctness with mocked deps | UMNN strictly monotone on a grid; `Identity` conditioner shape-preserving; `n_params` matches `torch` count; `build_from_budget` enumeration lands within ±10%, raises within ±15%–outside |
| **Integration** | Composed components, tiny budgets, finish in seconds | 200-step CDSBI on N=50 `LocationNormal1D`; runner produces parseable run-dir; `STATUS=OK` written |
| **Diagnostics** | Diagnostics give *known* answers on analytical-pivot problems | See §9.1 |
| **Intensive** | Full-budget replication; opt-in via `pytest -m intensive` | §8.1 replication, ~3 min; later: §8.2, §8.3, §8.4 |

A fifth **Ablation** category activates at **v3**, when the §8.4
exponential-rate setup (`ExponentialRate` + `JointUMNNFlow`) provides
a target on which the §3.5 mechanism actually fires under training.
The category covers both (i) the runner's safety-check test
(`MonotonicityMismatchError` fires on a non-R2 flow unless
`allow_ablation=True`) and (ii) the trained-folding empirical test
(`final_loss < entropy_lower_bound − margin`) with the empirical
noise-floor margin. The safety-check *code path* exists in v0 (the
runner does perform the check); only its regression test waits for v3
when there's a non-R2 flow in the codebase to point it at.

### 9.1 Diagnostic-regression tests

Tested against analytical truth, not against frozen snapshots. For
each diagnostic we use an oracle pivot `r(θ, X) = r*(θ, X)` and verify
pass-rate on a known-correct case and fail-rate on a deliberately
miscalibrated case:

| Test | Setup | Expected behavior |
|---|---|---|
| `test_marginal_pit_oracle_passes` | Apply to `r*` on N=5000 `LocationNormal1D` | KS ≤ floor (~0.023) in ≥ 99/100 seeds |
| `test_marginal_pit_catches_miscalibration` | Same setup, pivot `0.5·(θ − X)` (under-dispersed ×2) | KS > floor in ≥ 99/100 seeds |
| `test_coverage_oracle_exact` | `Coverage` on `PivotBasedProcedure(r*)` at α ∈ {0.5, 0.68, 0.9, 0.95} | Empirical coverage within MC error of nominal at every α |
| `test_coverage_catches_mismatch` | Same on under-dispersed pivot | Empirical coverage < nominal by detectable margin |
| `test_per_bin_floor_calculation` | `ks_noise_floor(N=5000, n_bins=5)` and similar | Matches analytic `1.628/√(N/k)` to 4 decimals |

`test_joint_mahalanobis_catches_correlation` ships at v1 alongside
the `JointMahalanobis` diagnostic itself.

### 9.2 (R2) ablation tests — deferred to v3

The §3.5 mechanism (NF-MLE + autograd Jacobian + non-monotone-in-X
flow ⇒ `Z(θ) > 1` ⇒ loss < entropy floor) requires a target where
the truth is not easily approximated by the monotone-in-X class —
otherwise training simply finds the monotone truth and folding never
happens (the §3.5 mechanism *permits* folding to lower the loss but
does not *force* it). `LocationNormal1D` (truth `r* = θ − X`, linear
in X) is in the realizable monotone class, so a trained 1D ablation
test on this target is not reliable.

The full ablation category — both the safety-check test and the
trained-folding empirical test (with empirical noise-floor margin) —
lands at **v3** alongside `ExponentialRate` + `JointUMNNFlow`, where
the §8.4 setup provides a target on which the mechanism actually
fires.

The runner's safety check (`MonotonicityMismatchError` raised when
`loss.required_guarantees` is not a subset of
`flow.monotonicity_guarantees` and `allow_ablation=False`) is
implemented in v0 code — only its regression test waits for v3 when a
non-R2 flow exists in the codebase to point it at.

### 9.3 Determinism contract

- `pytest tests/test_determinism.py` runs the same `(seed, config)`
  twice and asserts every diagnostic value matches to `float32`
  precision.
- `torch.use_deterministic_algorithms(True)` set unconditionally;
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` set unconditionally (harmless on
  CPU paths, ensures determinism on any CUDA path).
- Documented: determinism applies *within a (machine, torch version,
  device) tuple*; cross-device runs differ in float-rounding only.

### 9.4 What we explicitly don't test

- **Bit-equivalence to manuscript Table values.** The intensive
  category checks tolerance bands derived from the diagnostic's noise
  floor, not bit-identity. The manuscript numbers were produced by a
  different code path; insisting on numerical equivalence would fix
  arbitrary implementation choices into the test suite.
- **`sbi` package internals.** We test our wrappers' input/output
  contract; we trust the package's training loop. `sbi`-based runners
  pass explicit seeds per-call to `sbi.inference.*` constructors and
  `.train()` methods rather than relying on global state.
- **GPU-specific performance.** v0-scale §8.1 runs may be CPU-faster
  than GPU due to kernel-launch overhead; the framework picks GPU by
  default but per-experiment configs override to `device=cpu` for
  small-scale workloads. Documented in `tests/intensive/README.md`.

## 10. Reproducibility model

A small `cdsbi.reproducibility` module:

- `seed_everything(seed: int) → SeededRNGs` — seeds `random`, `numpy`,
  `torch`, `torch.cuda`; returns a namespace of named RNG streams
  (`train`, `eval`, `init`) deterministically derived from the master
  seed via `hashlib`. **Does not assert seeding of `sbi` global
  state**; `sbi`-based runners pass explicit per-call seeds to each
  `sbi.inference.*` constructor and `.train()` invocation.
- `capture_env() → dict` — git sha, dirty flag, library versions,
  hardware, device. Written to `env.json` on every run.
- `RunDir.write_atomic(path, payload)` — writes via `path.tmp` +
  rename so a mid-write crash leaves the prior version intact and
  never produces a partially-written parquet.
- `RunDir.set_status(state)` — atomic `STATUS` file update with
  states `RUNNING` / `OK` / `FAILED`.

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
2. **CD-SBI's seed-averaged results land within these falsifiable
   tolerance bands on `LocationNormal1D`**, with "interior" defined
   operationally as **`|θ_0| ≤ 5`** (excludes the outer ~2 units of
   the [−7, 7] proposal where edge effects live, per manuscript §8.1
   "interior is uniformly clean"):

   | Diagnostic | Manuscript value | v0 tolerance band | Justification |
   |---|---|---|---|
   | Pivot RMSE (interior) | 0.030 | ≤ 0.05 | ~1.5× manuscript; well above optimization noise, well below 1σ of pivot's marginal scale |
   | Marginal PIT KS | 0.008 | ≤ 0.023 | Equal to marginal noise floor at N=5000 (`1.628/√5000`) — "passes the test" |
   | Conditional PIT KS (interior) | ~0.01 (bulk; manuscript §8.1) | ≤ 0.052 | Per-bin noise floor at N=5000 with k=5 bins of N/k=1000 each (`1.628/√1000`) — "passes the test" |
   | Coverage error (interior) | < 0.01 | ≤ 0.02 | ~2× manuscript; tight enough to catch real regressions, loose enough to absorb seed-to-seed variance |

3. All five methods land their backbone parameter count within ±10%
   of each budget target via the §4.4 closest-candidate enumeration
   (±15% accepted with logged warning; outside ±15% raises
   `BudgetUnreachableError`).
4. Full default `pytest` suite (unit + integration + diagnostics)
   passes in under 30 s. (Ablation category lands v3.)
5. `pytest -m intensive` runs the §8.1 replication and matches the
   tolerance bands above.

## 12. Milestone roadmap

Illustrative — not part of v0 commitment.

| Milestone | New code | New configs | Adds |
|---|---|---|---|
| **v0** §8.1 sweep | UMNN, AdditiveFlow1D, MAFAdapter, Identity conditioner, 5 runners (each returning a `ConfidenceProcedure`), 1D root-finder / HPD extractor, 4 diagnostics (1, 2, 3, 5), Hydra/run-dir/seeding/device plumbing, unit + integration + diagnostics + intensive test categories | `target/loc_normal_1d`, `flow/{additive_umnn, maf}`, `method/{cd_sbi, npe, nle, nre, lf2i}`, `budget/{small, medium, large, xlarge}`, `experiment/8_1_*` | First cross-method comparison at matched budget |
| **v1** §8.2 | `TriangularAdditiveFlow`; multivariate `confidence_set`; `JointMahalanobis` diagnostic + its regression test | `target/loc_gauss_2d_iid`, `flow/triangular_additive`, `experiment/8_2_*` | Multivariate triangular flow; joint Mahalanobis becomes load-bearing |
| **v2** §8.3 | (no new flow) | `target/loc_gauss_2d_corr`, `experiment/8_3_*` | KR-uniqueness empirical evidence |
| **v3** §8.4 + (R2) ablation | `DoublyMonotoneUMNN`, `JointUMNNFlow` (§8.4 form with sufficient-statistic reduction), `JointUMNN1DFlow` (1D analog for the direct-construction mechanism test), `MLPConditioner` doing the X→T reduction with non-zero `log_det_jac_input_contribution`; new **Ablation** test category (safety-check + trained-folding empirical test with empirical noise-floor margin + 1D direct-construction mechanism test) | `target/exp_rate`, `flow/{doubly_monotone, joint_umnn, joint_umnn_1d}`, `experiment/8_4_*` | The §8.4 (R2)-failure as a published result; (R2) safety property regression-tested |
| **v4** SBI benchmark | `SBIBenchmarkAdapter` | `target/sbibm_*` | Community leaderboards |
| **v5** §3.7 alt-loss | `MarginalPushforwardLoss`, `MarginalPlusHSICLoss`, `StratifiedLoss` | `loss/*`, `experiment/3_7_*` | Empirical Class 1–4 vs Class 5 demonstration |
| **v6** synthetic high-d | (mostly config; possibly `MLPConditioner` for parameter encoding) | `target/loc_gauss_kd`, `experiment/scaling_*` | §11.5 scaling claim |
| **v7** astro inference | `AstroSimulator` (likely wraps a domain code) | `target/cosmo_*` or `target/gw_*` | Real-data deliverable |
| **v8** image / sequence | `CNNConditioner`, `AttentionConditioner`, possibly `MaskedAttentionFlow`, `NSFAdapter` | `conditioner/{cnn, attention}`, `flow/{masked_attention, nsf}` | §11.7 territory + §11.5 attention variant |

## 13. Forward-hook commitments

These are *not built* in v0 but must be designed-for-in-v0:

1. **The conditioner seam.** `Flow.forward(θ, context)` always takes
   `context` rather than `X`. v8's CNN drops in with zero changes to
   the flow signature.
2. **`monotonicity_guarantees` as a first-class flow attribute.** v0
   has two flows that exercise this field (`AdditiveFlow1D` =
   `{R1, R2}`, `MAFAdapter` = `frozenset()`); the runner's safety
   check (`MonotonicityMismatchError`) is implemented in v0 and waits
   for v3's `JointUMNNFlow` (= `{R1}` only) to be exercised by tests.
   The protocol is stable across the activation gap.
3. **`log_det_jac_input` naming + conditioner's
   `log_det_jac_input_contribution`.** v0's identity conditioner
   contributes 0; v3's sufficient-statistic conditioner contributes
   `log|∂T/∂X|`. No flow-signature change between v0 and v3.
4. **`Simulator.entropy_lower_bound()` as optional protocol.** Defined
   on `LocationNormal1D` in v0 (= `½ log(2πe)`) but unused until v3,
   when the ablation regression test consumes it via
   `loss.population_lower_bound(sim)`. Returns `None` when unknown
   (real-data simulators).
5. **`ConfidenceProcedure` as the cross-method interface.** v0
   instantiates all five subtypes; v1+ adds WALDO and Box CD as
   additional `CriticalValueProcedure` instances without touching the
   diagnostics or runner protocols.

## 14. Non-goals (explicit, to prevent scope creep)

- No multivariate flow code in v0 (lands v1)
- No `JointMahalanobis` diagnostic in v0 (degenerate in 1D; lands v1
  with multivariate flows)
- No (R2) ablation tests in v0 (the §3.5 mechanism doesn't reliably
  fire on `LocationNormal1D` because the truth is in the realizable
  monotone-in-X class; full Ablation test category — safety-check,
  1D direct-construction mechanism test, §8.4 trained-folding test —
  lands at v3 alongside `ExponentialRate` + `JointUMNNFlow`). The
  runner's safety check **code path** is implemented in v0.
- No multivariate confidence-set computation in v0 (1D root-find is in
  v0; multivariate is v1+)
- No image / sequence support in v0 (lands v8)
- No alt-class loss objectives in v0 (lands v5)
- No sequential variants (open problem §11.6; no scheduled milestone)
- No cross-machine bit-identity reproducibility (within-machine
  determinism only)
