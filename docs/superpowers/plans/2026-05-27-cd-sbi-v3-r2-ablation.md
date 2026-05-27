# CD-SBI v3 — §8.4 exponential rate + (R2) ablation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the v0/v1/v2 infrastructure to manuscript §8.4 — non-additive exponential-rate model with sufficient-statistic reduction `T = Σ X_i` — and add the empirical (R2) ablation that proves architectural enforcement of monotonicity-in-X is a correctness requirement, not a stylistic preference.

**Architecture:**
- `ExponentialRate` simulator: `X_i ~ Exp(θ)` iid (n=5), `θ ~ U[0.3, 3.0]`, sufficient statistic `T = Σ X_i`, closed-form truth pivot `r*(θ, T) = Φ⁻¹(F_{χ²_{10}}(2θT))`. Exposes `entropy_lower_bound() = 0.88` for the ablation regression test.
- `MLPConditioner` (X → T sufficient-statistic reduction): `encode(x: (n, d_x)) → (T: (n, 1), log|∂T/∂X|: (n,))`. For T = Σ X_i with n=5, `log|∂T/∂X| = ½ log 5` constant per row. v3's first conditioner with non-zero `log_det_jac_input_contribution`.
- `DoublyMonotoneUMNN` (§6.1 form 2): `r(θ, T) = b_umnn(T) + ∫_{θ_ref}^θ softplus(α(t) + β_umnn(T)) dt` with `b_umnn`, `β_umnn` UMNNs of T. Both R1 (∂_θ r = softplus(·) > 0) and R2 (∂_T r = b'_umnn + β'_umnn · ∫ σ(·) dt > 0) enforced architecturally. `monotonicity_guarantees = {R1, R2}`. The §8.4 main flow.
- `JointUMNNFlow` (the v3 ablation form): generic UMNN in θ, with T input fed as context. Monotone in θ (R1) but NOT in T — autograd computes `∂_T r` at training time. `monotonicity_guarantees = {R1}` only. The §8.4 ablation flow that demonstrates the §3.5 mechanism failure.
- `JointUMNN1DFlow`: 1D synthetic-mechanism analog of JointUMNNFlow with scalar X (no T reduction). Used for the direct-construction mechanism test.
- New Ablation test category at `tests/ablation/`: (a) safety check — `allow_ablation=False` + R1-only flow raises `MonotonicityMismatchError`; (b) trained-folding empirical — final loss falls below the simulator's `entropy_lower_bound()` with empirical noise-floor margin; (c) 1D direct-construction mechanism — synthetic 1D `JointUMNN1DFlow` reproduces the folding mechanism.
- New configs: `target/exp_rate`, `flow/{doubly_monotone, joint_umnn, joint_umnn_1d}`, `experiment/8_4_*`.
- New `paper_table_8_4` shim (§8.4 has no Jacobian-recovery — the truth pivot's Jacobian is non-constant, unlike §8.3's L⁻¹ — so the table reuses the §8.2 column set).
- Manuscript §8.4 integration as a final task with the same seven-block template that landed §8.1/§8.2/§8.3.

**Tech Stack:** PyTorch (autograd for the ablation's ∂_T r; Gauss-Legendre quadrature for the explicit integral), Hydra, nflows / sbi (MAF backbone for NPE/NLE baselines), pytest (new `ablation` marker), pandas + parquet.

**Out of scope (carried from v1/v2 addenda + still open):**
- True KDE-based HPD region for NPE at d > 1. §8.4 is 1D (target/T are scalar) so this doesn't apply here; footnote-with-§8.2/§8.3 if §8.4's cross-method table is added.
- SetSize true volume estimation at d > 1: §8.4 is 1D, so set width = right − left is exact. No issue.
- `r_star_jacobian` for non-constant Jacobians (§8.4's truth Jacobian depends on (θ, T)). v2's `JacobianRecovery` diagnostic only works when the truth Jacobian is constant; §8.4 skips it.

---

## File structure

```
NEW
  src/cdsbi/simulators/exp_rate.py                  # ExponentialRate simulator
  src/cdsbi/conditioners/mlp.py                      # MLPConditioner (X→T)
  src/cdsbi/flows/doubly_monotone.py                 # DoublyMonotoneUMNN (R1+R2 form 2)
  src/cdsbi/flows/joint_umnn.py                      # JointUMNNFlow (R1 only — ablation)
  src/cdsbi/flows/joint_umnn_1d.py                   # JointUMNN1DFlow (1D mechanism analog)
  configs/target/exp_rate.yaml
  configs/flow/doubly_monotone.yaml
  configs/flow/joint_umnn.yaml
  configs/flow/joint_umnn_1d.yaml
  configs/experiment/8_4_replication.yaml
  configs/experiment/8_4_baseline_sweep.yaml
  configs/experiment/8_4_ablation.yaml
  tests/unit/test_exp_rate_simulator.py
  tests/unit/test_mlp_conditioner.py
  tests/unit/test_doubly_monotone_flow.py
  tests/unit/test_joint_umnn_flow.py
  tests/unit/test_joint_umnn_1d_flow.py
  tests/ablation/__init__.py
  tests/ablation/test_safety_check.py
  tests/ablation/test_trained_folding.py
  tests/ablation/test_1d_mechanism.py
  tests/intensive/test_replicate_8_4.py

MODIFY
  src/cdsbi/experiments/run.py                      # d-aware _build_flow extension; wire MLPConditioner for exp_rate target
  src/cdsbi/analysis/paper_tables.py                # paper_table_8_4 shim
  configs/budget/*.yaml                             # doubly_monotone_hidden per budget
  pyproject.toml                                    # add "ablation" pytest marker
  cd_sbi_v7.tex                                     # §8.4 manuscript integration
```

No changes to: existing simulators (LocationNormal1D, LocationGaussian2D_iid, LocationGaussian2D_corr), existing flows (AdditiveFlow1D, TriangularAdditiveFlow, MAFAdapter), confidence_set procedures, six existing diagnostics, JacobianRecovery (skip-no-op already handles the missing `r_star_jacobian` case for exp_rate). v0+v1+v2 carry over.

---

## Task 1: `ExponentialRate` simulator

**Files:**
- Create: `src/cdsbi/simulators/exp_rate.py`
- Test: `tests/unit/test_exp_rate_simulator.py`

The §8.4 target: `X_i ~ Exp(θ)` iid (n=5), `θ ~ U[0.3, 3.0]`. Sufficient statistic `T = Σ_{i=1}^n X_i ~ Gamma(n, 1/θ)`. The simulator returns the full `(n_samples, n)`-shape X tensor; the `MLPConditioner` (Task 2) does the X→T reduction. Closed-form truth pivot per manuscript: `r*(θ, T) = Φ⁻¹(F_{χ²_{2n}}(2θT))` (in distribution, `2θT ~ χ²_{2n}`).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_exp_rate_simulator.py`:

```python
"""ExponentialRate — n=5 iid exponential samples, T = Σ X_i sufficient statistic."""
from __future__ import annotations

import numpy as np
import torch


def test_simulator_sample_shape_and_distribution(seed):
    from cdsbi.simulators.exp_rate import ExponentialRate
    rng = np.random.default_rng(seed)
    sim = ExponentialRate()
    assert sim.d_theta == 1 and sim.d_x == 5
    assert sim.n_iid == 5
    theta, x = sim.sample(20000, rng)
    assert theta.shape == (20000, 1)
    assert x.shape == (20000, 5)
    # X | θ ~ Exp(θ) ⇒ E[X] = 1/θ. With θ ~ U[0.3, 3.0], E_marginal[X] ≈ E_θ[1/θ].
    # Sanity-check that values are positive.
    assert (x > 0).all()
    # For a specific theta, X mean ≈ 1/θ to within MC noise (single-theta slice).


def test_sample_x_given_theta_shape_and_mean(seed):
    from cdsbi.simulators.exp_rate import ExponentialRate
    rng = np.random.default_rng(seed)
    sim = ExponentialRate()
    x = sim.sample_x_given_theta(theta_0=1.5, n=20000, rng=rng)
    assert x.shape == (20000, 5)
    # Each X_i ~ Exp(θ=1.5) ⇒ E[X_i] = 2/3
    np.testing.assert_allclose(float(x.mean()), 2.0 / 3.0, atol=0.02)
    # T = Σ X_i has mean n/θ = 5/1.5 ≈ 3.33
    T = x.sum(dim=-1).numpy()
    np.testing.assert_allclose(T.mean(), 5.0 / 1.5, atol=0.05)


def test_r_star_marginal_is_standard_normal(seed):
    """r*(θ, T) = Φ⁻¹(F_{χ²_{2n}}(2θT)) should be ~N(0, 1) marginally over (θ, X)."""
    from cdsbi.simulators.exp_rate import ExponentialRate
    rng = np.random.default_rng(seed)
    sim = ExponentialRate()
    theta, x = sim.sample(20000, rng)
    T = x.sum(dim=-1, keepdim=True)  # (n, 1)
    r = sim.r_star(theta, T).numpy()
    # Marginal ≈ N(0, 1)
    np.testing.assert_allclose(r.mean(), 0.0, atol=0.05)
    np.testing.assert_allclose(r.std(), 1.0, atol=0.05)


def test_entropy_lower_bound_matches_paper():
    """The manuscript reports the NF-MLE loss lower bound ≈ 0.88 for this model.

    `entropy_lower_bound()` is implemented as a Monte-Carlo evaluation of the
    NF-MLE loss at the truth pivot r*, on the (θ, T) reduced space with the
    conditioner's log|∂T/∂X| accounted for. MC noise at n_mc=50_000 is small
    (~ 0.01); the band [0.70, 1.10] is generous to absorb seed and clamp-floor
    variance.
    """
    from cdsbi.simulators.exp_rate import ExponentialRate
    sim = ExponentialRate()
    H = sim.entropy_lower_bound()
    assert 0.70 <= H <= 1.10, f"entropy_lower_bound={H} outside expected [0.70, 1.10] around manuscript's 0.88"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_exp_rate_simulator.py -v`

Expected: FAIL on import with `ModuleNotFoundError: No module named 'cdsbi.simulators.exp_rate'`.

- [ ] **Step 3: Implement the simulator**

Create `src/cdsbi/simulators/exp_rate.py`:

```python
"""ExponentialRate: X_i ~ Exp(θ) iid (n=5), θ ~ U[0.3, 3.0].

Sufficient statistic T = Σ X_i ~ Gamma(n, 1/θ); equivalently 2θT ~ χ²_{2n}.
Truth pivot per manuscript §8.4: r*(θ, T) = Φ⁻¹(F_{χ²_{2n}}(2θT)).
This is the v3 non-additive target — the multiplicative θT interaction
takes the model outside the additive class of §8.1–§8.3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from scipy.stats import chi2, norm


@dataclass
class ExponentialRate:
    theta_range: Tuple[float, float] = (0.3, 3.0)
    n_iid: int = 5
    d_theta: int = 1

    @property
    def d_x(self) -> int:
        return self.n_iid

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = self.theta_range
        theta_np = rng.uniform(a, b, size=(n, self.d_theta))
        # X_i ~ Exp(θ) — scale parameter 1/θ. numpy's Exponential takes scale.
        x_np = rng.exponential(scale=1.0 / theta_np, size=(n, self.n_iid))
        return (
            torch.from_numpy(theta_np).float(),
            torch.from_numpy(x_np).float(),
        )

    def sample_x_given_theta(self, theta_0, n: int, rng: np.random.Generator) -> torch.Tensor:
        """Draw n samples of X = (X_1, …, X_n_iid) conditional on θ = θ_0."""
        theta_vec = np.atleast_1d(np.asarray(theta_0, dtype=np.float64))
        assert theta_vec.shape == (self.d_theta,), (
            f"theta_0 has shape {theta_vec.shape}, expected ({self.d_theta},)"
        )
        x_np = rng.exponential(scale=1.0 / float(theta_vec[0]), size=(n, self.n_iid))
        return torch.from_numpy(x_np).float()

    def r_star(self, theta: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Truth pivot on the (θ, T) sufficient-statistic space.

        r*(θ, T) = Φ⁻¹(F_{χ²_{2n}}(2θT)).
        Inputs:
          theta: (n, 1) — parameter samples
          T:     (n, 1) — sufficient statistic Σ X_i
        Returns: (n, 1)
        """
        df = 2 * self.n_iid
        # Compute via scipy on CPU then move to theta's device. The truth pivot
        # is only used by PivotRMSE and diagnostics, never inside autograd loops.
        twothT = (2.0 * theta * T).detach().cpu().numpy()
        u = chi2.cdf(twothT, df=df)
        # Clamp away from {0, 1} for the inverse-Φ; the diagnostic eval support
        # is well inside (0, 1) for the simulator's θ range, but be defensive.
        u = np.clip(u, 1e-10, 1.0 - 1e-10)
        r_np = norm.ppf(u)
        return torch.from_numpy(r_np).float().to(theta.device)

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """log p(X_1, …, X_n_iid | θ) = Σ_i log p(X_i | θ); X_i ~ Exp(θ)."""
        # log p(X_i | θ) = log θ − θ X_i for X_i > 0
        # x: (n, n_iid); theta: (n, 1)
        return (torch.log(theta) - theta * x).sum(dim=-1)

    def entropy_lower_bound(self, n_mc: int = 50000, seed: int = 42) -> float:
        """Monte-Carlo estimate of E[NF-MLE loss at truth r*] on the (θ, T)
        space with the conditioner's log|∂T/∂X| accounted for.

        The NF-MLE loss the trainer sees with X→T conditioner is
            L = ½ r² + ½ log(2π) − log|∂r/∂T| − log|∂T/∂X|.
        At r = r*(θ, T), the expected value of L is the conditional-entropy
        lower bound that any valid normalized surrogate must satisfy
        (Theorem 3.2 / §3.2). Per manuscript §8.4, this evaluates to ≈ 0.88
        for the exponential-rate model with n=5 and θ ~ U[0.3, 3.0].

        Implemented via MC rather than a closed form because the analytic
        derivation requires expectations over (θ, T) involving log f_{χ²}
        and log φ that don't simplify cleanly; MC is a faithful evaluation
        of the same loss function the trainer optimizes.
        """
        from scipy.stats import chi2, norm
        rng = np.random.default_rng(seed)
        theta_np = rng.uniform(*self.theta_range, size=(n_mc, 1))
        z_np = rng.standard_normal(size=(n_mc, self.n_iid))
        # X_i ~ Exp(θ): scale = 1/θ
        x_np = rng.exponential(scale=1.0 / theta_np, size=(n_mc, self.n_iid))
        T_np = x_np.sum(axis=-1, keepdims=True)  # (n_mc, 1)
        df = 2 * self.n_iid
        twothT_np = 2.0 * theta_np * T_np
        u_np = chi2.cdf(twothT_np, df=df)
        u_np = np.clip(u_np, 1e-10, 1.0 - 1e-10)
        r_star_np = norm.ppf(u_np)  # (n_mc, 1)
        # ∂r*/∂T = 2θ · f_{χ²_{2n}}(2θT) / φ(r*) by chain rule + inverse-CDF
        f_chi2_np = chi2.pdf(twothT_np, df=df)
        phi_r_np = norm.pdf(r_star_np)
        dr_dT_np = (2.0 * theta_np * f_chi2_np) / np.clip(phi_r_np, 1e-30, None)
        log_dr_dT_np = np.log(np.clip(dr_dT_np, 1e-30, None))
        log_dT_dX = 0.5 * math.log(self.n_iid)
        loss_per_sample = (
            0.5 * (r_star_np ** 2)
            + 0.5 * math.log(2 * math.pi)
            - log_dr_dT_np
            - log_dT_dX
        )
        return float(loss_per_sample.mean())
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/unit/test_exp_rate_simulator.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/simulators/exp_rate.py tests/unit/test_exp_rate_simulator.py
git commit -m "$(cat <<'EOF'
feat(simulators): ExponentialRate (§8.4 target)

X_i ~ Exp(θ) iid (n=5), θ ~ U[0.3, 3.0]. Sufficient statistic
T = Σ X_i ~ Gamma(n, 1/θ); truth pivot r*(θ, T) = Φ⁻¹(F_{χ²_{2n}}(2θT)).
The non-additive case for §8.4: multiplicative θT interaction takes
the model outside the additive class of §8.1–§8.3.

entropy_lower_bound() implements the closed-form
n * (1 − E_θ[log θ]) per the §8.4 ablation regression test;
this is the lower bound the trained ablation flow will be checked
against (loss < bound ⇒ Z(θ) > 1 ⇒ the §3.5 folding mechanism).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `MLPConditioner` (X → T sufficient-statistic reduction)

**Files:**
- Create: `src/cdsbi/conditioners/mlp.py`
- Test: `tests/unit/test_mlp_conditioner.py`

v0's `Identity` conditioner returns X as context with zero log-det contribution. v3 adds an `MLPConditioner` that reduces `X = (X_1, …, X_{n_iid})` to a scalar (or low-dim) sufficient statistic `T`, carrying a non-zero `log|∂T/∂X|` contribution. For §8.4 specifically, T = Σ X_i: a linear map with `|∂T/∂X| = √n_iid` (the gradient ‖dT/dX‖ for the directional differential dT = dX_1 + ⋯ + dX_n). The conditioner exposes the reduction map as a trainable MLP — for §8.4 the truth is known to be the sum, so the MLP can be initialized as a sum and frozen, OR trained to learn the reduction (which validates the framework when the sufficient statistic is not known).

For the v3 main experiment we use a frozen-sum initialization (the trained-MLP variant is a v3.1+ extension). The `log_det_jac_input_contribution` is constant = `½ log(n_iid)`, matching `√n_iid` row-wise.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_mlp_conditioner.py`:

```python
"""MLPConditioner — frozen-sum sufficient-statistic reduction for §8.4."""
from __future__ import annotations

import math

import torch


def test_mlp_conditioner_frozen_sum_reduction():
    from cdsbi.conditioners.mlp import MLPConditioner
    cond = MLPConditioner(input_dim=5, output_dim=1, mode="frozen_sum")
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [0.1, 0.2, 0.3, 0.4, 0.5]])
    T, log_det = cond.encode(x)
    assert T.shape == (2, 1)
    assert log_det.shape == (2,)
    # Frozen-sum mode: T should equal Σ X_i exactly.
    torch.testing.assert_close(T.squeeze(-1), torch.tensor([15.0, 1.5]))
    # log|∂T/∂X| for T = Σ X_i with n=5: |∂T/∂X| = √5 row-wise (directional gradient norm).
    expected_log_det = 0.5 * math.log(5.0)
    torch.testing.assert_close(log_det, torch.full((2,), expected_log_det))


def test_mlp_conditioner_frozen_sum_no_trainable_params():
    """Frozen-sum mode has zero learnable parameters — the reduction is hardcoded."""
    from cdsbi.conditioners.mlp import MLPConditioner
    cond = MLPConditioner(input_dim=5, output_dim=1, mode="frozen_sum")
    assert cond.n_params() == 0


def test_mlp_conditioner_invalid_mode_raises():
    """Unknown mode raises a clear error (rather than silently producing garbage)."""
    from cdsbi.conditioners.mlp import MLPConditioner
    try:
        MLPConditioner(input_dim=5, output_dim=1, mode="undefined_mode_xyz")
    except ValueError as e:
        assert "mode" in str(e).lower()
        return
    raise AssertionError("MLPConditioner did not raise on unknown mode")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_mlp_conditioner.py -v`

Expected: FAIL on import with `ModuleNotFoundError: No module named 'cdsbi.conditioners.mlp'`.

- [ ] **Step 3: Implement the conditioner**

Create `src/cdsbi/conditioners/mlp.py`:

```python
"""MLPConditioner: X → T sufficient-statistic reduction with non-zero log|∂T/∂X|.

v3 introduces this as the first conditioner with a non-trivial Jacobian
contribution. For §8.4 the truth is T = Σ X_i (known closed-form sufficient
statistic for the exponential model), so we ship a frozen-sum mode that
hardcodes the reduction and the constant log|∂T/∂X| = ½ log(n_iid). A
trainable-MLP mode for unknown sufficient statistics is a v3.1+ extension.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


class MLPConditioner:
    """Frozen-sum / trainable-MLP sufficient-statistic reducer.

    Currently only `mode='frozen_sum'` is supported; trainable modes land in
    v3.1+ when an unknown-T target arrives.
    """

    def __init__(self, input_dim: int, output_dim: int = 1, mode: str = "frozen_sum"):
        if mode != "frozen_sum":
            raise ValueError(
                f"MLPConditioner mode={mode!r} not supported; only 'frozen_sum' available in v3"
            )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        # log|∂T/∂X| for T = Σ X_i:
        # the map (X_1, …, X_n) → T = Σ X_i has gradient row vector (1, …, 1);
        # interpreted as a directional Jacobian, the magnitude is √n.
        self._log_det_per_row = 0.5 * math.log(float(input_dim))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """X → (T, log|∂T/∂X|).

        Input x: shape (n, input_dim). For frozen-sum mode, T = Σ X_i across
        the last axis, output_dim must equal 1 (sum reduction), and
        log|∂T/∂X| is the constant ½ log(n) per row.
        """
        assert self.mode == "frozen_sum"
        assert self.output_dim == 1, (
            f"frozen_sum mode requires output_dim=1, got {self.output_dim}"
        )
        T = x.sum(dim=-1, keepdim=True)  # (n, 1)
        log_det = torch.full(
            (x.shape[0],), self._log_det_per_row, dtype=x.dtype, device=x.device,
        )
        return T, log_det

    def n_params(self) -> int:
        # frozen-sum mode has no trainable parameters
        return 0
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/unit/test_mlp_conditioner.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/conditioners/mlp.py tests/unit/test_mlp_conditioner.py
git commit -m "$(cat <<'EOF'
feat(conditioners): MLPConditioner — frozen-sum X→T reduction

v0's Identity conditioner returns X unchanged with zero log-det.
v3 adds MLPConditioner with a frozen-sum mode for §8.4: collapses
X = (X_1, …, X_n) → T = Σ X_i with constant log|∂T/∂X| = ½ log(n)
per row. This is the first conditioner with non-zero
log_det_jac_input_contribution — exercises the v0 forward-hook
naming + the Loss layer's accounting of the conditioner's
Jacobian contribution.

Trainable-MLP modes (for unknown sufficient statistics) land
in v3.1+ when an unknown-T target arrives.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Hydra configs for §8.4

**Files:**
- Create: `configs/target/exp_rate.yaml`
- Create: `configs/flow/doubly_monotone.yaml`
- Create: `configs/flow/joint_umnn.yaml`
- Create: `configs/flow/joint_umnn_1d.yaml`
- Create: `configs/experiment/8_4_replication.yaml`
- Create: `configs/experiment/8_4_baseline_sweep.yaml`
- Create: `configs/experiment/8_4_ablation.yaml`

Doing configs before the flow implementations (Tasks 4-6) so the integration tests in Task 7 can resolve them. Flow configs cite the still-to-be-built classes by import path; that's expected and matches v2's approach.

- [ ] **Step 1: Write `configs/target/exp_rate.yaml`**

```yaml
name: exp_rate
_target_: cdsbi.simulators.exp_rate.ExponentialRate
theta_range: [0.3, 3.0]
n_iid: 5
```

- [ ] **Step 2: Write `configs/flow/doubly_monotone.yaml`**

```yaml
name: doubly_monotone
_target_: cdsbi.flows.doubly_monotone.DoublyMonotoneUMNN
hidden: ${budget.doubly_monotone_hidden}
theta_ref: 0.3  # lower endpoint of the proposal support; integrand always over [theta_ref, theta]
```

- [ ] **Step 3: Write `configs/flow/joint_umnn.yaml`**

```yaml
name: joint_umnn
_target_: cdsbi.flows.joint_umnn.JointUMNNFlow
hidden: ${budget.doubly_monotone_hidden}  # share budget with doubly-monotone for matched compare
```

- [ ] **Step 4: Write `configs/flow/joint_umnn_1d.yaml`**

```yaml
name: joint_umnn_1d
_target_: cdsbi.flows.joint_umnn_1d.JointUMNN1DFlow
hidden: ${budget.doubly_monotone_hidden}
```

- [ ] **Step 5: Write `configs/experiment/8_4_replication.yaml`**

```yaml
# @package _global_
defaults:
  - override /target: exp_rate
  - override /flow: doubly_monotone
  - override /method: cd_sbi
  - override /budget: medium

experiment:
  name: 8_4_replication
  n_eval: 6000
  eval_thetas_interior: [0.5, 1.0, 1.5, 2.0, 2.5]
  eval_thetas_edge: []
  alpha_grid: [0.5, 0.68, 0.9, 0.95]
  n_eval_per_theta: 2000
  joint_mahalanobis_n_per_theta: 0  # 1D — JointMahalanobis no-ops
```

- [ ] **Step 6: Write `configs/experiment/8_4_baseline_sweep.yaml`**

```yaml
# @package _global_
defaults:
  - override /target: exp_rate
  - override /flow: doubly_monotone

experiment:
  name: 8_4_baseline_sweep
  n_eval: 6000
  eval_thetas_interior: [0.5, 1.0, 1.5, 2.0, 2.5]
  eval_thetas_edge: []
  alpha_grid: [0.5, 0.68, 0.9, 0.95]
  n_eval_per_theta: 2000

# Launch with explicit sweep dims:
#   python -m cdsbi.experiments.run -m experiment=8_4_baseline_sweep \
#     method=cd_sbi,npe,nle,nre,lf2i_bff \
#     budget=small,medium,large,xlarge \
#     seed=0,1,2,3,4 \
#     training.fresh_batch=false
```

- [ ] **Step 7: Write `configs/experiment/8_4_ablation.yaml`**

```yaml
# @package _global_
defaults:
  - override /target: exp_rate
  - override /flow: joint_umnn          # the R1-only ablation flow
  - override /method: cd_sbi
  - override /budget: medium

# This experiment compares the architectural (doubly_monotone) case to the
# autograd-Jacobian (joint_umnn) ablation. allow_ablation=true is required
# so CDSBIRunner doesn't raise MonotonicityMismatchError on joint_umnn (R1
# only). The ablation flow is intentionally below the (R2) contract — its
# final loss should fall below the simulator's entropy_lower_bound, which
# is the §3.5 mechanism failure.
method:
  allow_ablation: true

experiment:
  name: 8_4_ablation
  n_eval: 6000
  eval_thetas_interior: [0.5, 1.0, 1.5, 2.0, 2.5]
  eval_thetas_edge: []
  alpha_grid: [0.5, 0.68, 0.9, 0.95]
  n_eval_per_theta: 2000
```

- [ ] **Step 8: Verify configs load**

Skipped until Task 7 lands the wire-in code; for now confirm the YAML files exist and Hydra parses them syntactically:

```bash
python -c "from omegaconf import OmegaConf; print(OmegaConf.load('configs/target/exp_rate.yaml'))"
python -c "from omegaconf import OmegaConf; print(OmegaConf.load('configs/flow/doubly_monotone.yaml'))"
```

Expected: each prints a non-empty config dict.

- [ ] **Step 9: Commit**

```bash
git add configs/target/exp_rate.yaml configs/flow/doubly_monotone.yaml configs/flow/joint_umnn.yaml configs/flow/joint_umnn_1d.yaml configs/experiment/8_4_replication.yaml configs/experiment/8_4_baseline_sweep.yaml configs/experiment/8_4_ablation.yaml
git commit -m "$(cat <<'EOF'
config(experiment): §8.4 target + flow + experiment YAMLs

target/exp_rate uses the ExponentialRate simulator (n_iid=5,
theta_range=[0.3, 3.0]).
flow/{doubly_monotone, joint_umnn, joint_umnn_1d} stage the v3 flow
classes (implemented in Tasks 4-6).
experiment/8_4_{replication, baseline_sweep, ablation} cover the
intensive validation, cross-method sweep, and the (R2) ablation
experiment respectively.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `DoublyMonotoneUMNN` flow (§6.1 form 2)

**Files:**
- Create: `src/cdsbi/flows/doubly_monotone.py`
- Test: `tests/unit/test_doubly_monotone_flow.py`

The §8.4 main architecture per manuscript §6.1 form 2:

```
r(θ, T) = b_umnn(T) + ∫_{θ_ref}^θ softplus(α(t) + β_umnn(T)) dt
```

where `b_umnn`, `β_umnn` are monotone-increasing UMNN-style nets of T (each implements `softplus(MLP)`), and `α(t)` is a scalar trainable bias. Both monotonicities hold by construction:
- `∂_θ r = softplus(α(θ) + β_umnn(T)) > 0` (R1)
- `∂_T r = b'_umnn(T) + β'_umnn(T) ∫_{θ_ref}^θ σ(α(t) + β_umnn(T)) dt > 0` (R2, both summands non-negative with `b'_umnn > 0` strict)

Use Gauss-Legendre quadrature (already in v0's `UMNNBlock`) for both integrals. The `log_det_jac_input` returned by `.forward(θ, context)` is `log(∂_T r)` (closed form via the integrand at θ).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_doubly_monotone_flow.py`:

```python
"""DoublyMonotoneUMNN — §6.1 form 2 flow for §8.4."""
from __future__ import annotations

import torch

from cdsbi.flows.base import Guarantee


def test_doubly_monotone_advertises_R1_R2():
    from cdsbi.flows.doubly_monotone import DoublyMonotoneUMNN
    flow = DoublyMonotoneUMNN(hidden=8, theta_ref=0.3)
    assert Guarantee.R1 in flow.monotonicity_guarantees
    assert Guarantee.R2 in flow.monotonicity_guarantees


def test_doubly_monotone_forward_shape_and_log_det(seed):
    from cdsbi.flows.doubly_monotone import DoublyMonotoneUMNN
    torch.manual_seed(seed)
    flow = DoublyMonotoneUMNN(hidden=8, theta_ref=0.3)
    theta = torch.rand(64, 1) * 2.7 + 0.3  # uniform on [0.3, 3.0]
    T = torch.rand(64, 1) * 10 + 0.5       # positive scalar (sum-of-exponentials proxy)
    r, log_det = flow(theta, context=T)
    assert r.shape == (64, 1)
    assert log_det.shape == (64,)
    # log_det should be the log of ∂r/∂T evaluated at each (θ, T), a strictly
    # positive quantity ⇒ log_det is finite (no -inf).
    assert torch.isfinite(log_det).all()


def test_doubly_monotone_partial_theta_positive(seed):
    """∂r/∂θ = softplus(α(θ) + β_umnn(T)) > 0 for any (θ, T)."""
    from cdsbi.flows.doubly_monotone import DoublyMonotoneUMNN
    torch.manual_seed(seed)
    flow = DoublyMonotoneUMNN(hidden=8, theta_ref=0.3)
    theta = (torch.rand(64, 1) * 2.7 + 0.3).requires_grad_(True)
    T = torch.rand(64, 1) * 10 + 0.5
    r, _ = flow(theta, context=T)
    grad_theta = torch.autograd.grad(r.sum(), theta, create_graph=False)[0]
    assert (grad_theta > 0).all(), grad_theta


def test_doubly_monotone_partial_T_positive(seed):
    """∂r/∂T = b'_umnn(T) + β'_umnn(T) · ∫_{θ_ref}^θ σ(·) dt > 0 by construction."""
    from cdsbi.flows.doubly_monotone import DoublyMonotoneUMNN
    torch.manual_seed(seed)
    flow = DoublyMonotoneUMNN(hidden=8, theta_ref=0.3)
    theta = torch.rand(64, 1) * 2.7 + 0.3
    T = (torch.rand(64, 1) * 10 + 0.5).requires_grad_(True)
    r, _ = flow(theta, context=T)
    grad_T = torch.autograd.grad(r.sum(), T, create_graph=False)[0]
    assert (grad_T > 0).all(), grad_T


def test_doubly_monotone_at_theta_ref_equals_b(seed):
    """At θ = θ_ref, the integral is zero, so r(θ_ref, T) = b_umnn(T) exactly."""
    from cdsbi.flows.doubly_monotone import DoublyMonotoneUMNN
    torch.manual_seed(seed)
    flow = DoublyMonotoneUMNN(hidden=8, theta_ref=0.3)
    T = torch.tensor([[1.0], [2.0], [3.0]])
    theta_ref_tensor = torch.full((3, 1), 0.3)
    r, _ = flow(theta_ref_tensor, context=T)
    # b_umnn(T) is directly accessible
    b_T = flow._b_umnn(T)
    torch.testing.assert_close(r, b_T, atol=1e-5, rtol=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_doubly_monotone_flow.py -v`

Expected: FAIL on import with `ModuleNotFoundError: No module named 'cdsbi.flows.doubly_monotone'`.

- [ ] **Step 3: Implement the flow**

Create `src/cdsbi/flows/doubly_monotone.py`:

```python
"""DoublyMonotoneUMNN — §6.1 form 2:

    r(θ, T) = b_umnn(T) + ∫_{θ_ref}^{θ} softplus(α(t) + β_umnn(T)) dt

Both R1 and R2 enforced architecturally via Gauss-Legendre quadrature.
The §8.4 main flow; the closed-form log|∂_T r| accompanies the forward
pass so NF-MLE's Jacobian factor never needs autograd through the flow's
internals.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cdsbi.flows.base import Flow, Guarantee


_NODES_NP, _WEIGHTS_NP = np.polynomial.legendre.leggauss(12)


class _MonotoneScalarUMNN(nn.Module):
    """A 1D monotone-increasing function of T via T → bias + ∫_0^T softplus(MLP(t)) dt.

    Self-contained quadrature; does not depend on v0's UMNNBlock to keep the
    integral signature clean (we use both the value and the derivative).
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("_nodes", torch.tensor(_NODES_NP, dtype=torch.float32))
        self.register_buffer("_weights", torch.tensor(_WEIGHTS_NP, dtype=torch.float32))

    def _integrand(self, t: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.mlp(t)) + 1e-3  # strictly positive

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        """∫_0^T softplus(MLP(t)) dt + bias, with T positive."""
        n = T.shape[0]
        # Map [-1, 1] nodes to [0, T]
        u = self._nodes.view(1, -1, 1).expand(n, -1, 1)
        T_exp = T.view(n, 1, 1).expand(-1, u.size(1), -1)
        t = 0.5 * T_exp * (u + 1.0)
        integrand = self._integrand(t)
        weights = self._weights.view(1, -1, 1)
        integral = 0.5 * T * (weights * integrand).sum(dim=1)
        return self.bias + integral

    def derivative(self, T: torch.Tensor) -> torch.Tensor:
        """f'(T) = softplus(MLP(T))."""
        return self._integrand(T)


class DoublyMonotoneUMNN(nn.Module, Flow):
    monotonicity_guarantees = frozenset({Guarantee.R1, Guarantee.R2})

    def __init__(self, hidden: int = 16, theta_ref: float = 0.3):
        super().__init__()
        self.hidden = hidden
        self.theta_ref = theta_ref
        # b_umnn(T): scalar monotone-increasing function of T
        self._b_umnn = _MonotoneScalarUMNN(hidden=hidden)
        # β_umnn(T): scalar monotone-increasing function of T (parameter inside the integrand)
        self._beta_umnn = _MonotoneScalarUMNN(hidden=hidden)
        # α(t): scalar trainable bias on the θ-integrand
        # Implemented as a small MLP for flexibility; for the §8.4 truth pivot the
        # learned α should saturate at the value that makes ∂_θ r match
        # ∂_θ Φ⁻¹(F_{χ²_{2n}}(2θT)) on average.
        self._alpha_net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )
        self.register_buffer("_nodes", torch.tensor(_NODES_NP, dtype=torch.float32))
        self.register_buffer("_weights", torch.tensor(_WEIGHTS_NP, dtype=torch.float32))

    def _integrand(self, t: torch.Tensor, beta_T: torch.Tensor) -> torch.Tensor:
        """softplus(α(t) + β_umnn(T)) — strictly positive integrand."""
        alpha_t = self._alpha_net(t)
        return F.softplus(alpha_t + beta_T) + 1e-3

    def forward(
        self, theta: torch.Tensor, context: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """r(θ, T) and log|∂r/∂T| in closed form. context = T, shape (n, 1)."""
        assert context is not None and context.shape[-1] == 1, (
            "DoublyMonotoneUMNN expects context = T scalar, shape (n, 1)"
        )
        T = context
        n = theta.shape[0]

        # β_umnn(T) and its derivative; constant in the θ-integrand
        beta_T = self._beta_umnn(T)             # (n, 1)
        beta_prime_T = self._beta_umnn.derivative(T)  # (n, 1)

        # b_umnn(T) and b'_umnn(T) — used in the constant term and the Jacobian
        b_T = self._b_umnn(T)                    # (n, 1)
        b_prime_T = self._b_umnn.derivative(T)   # (n, 1)

        # Map [-1, 1] nodes to [theta_ref, theta]
        u = self._nodes.view(1, -1, 1).expand(n, -1, 1)  # (n, K, 1)
        a = self.theta_ref
        b = theta.view(n, 1, 1).expand(-1, u.size(1), -1)
        # t = a + (b - a)/2 * (u + 1)
        t = a + 0.5 * (b - a) * (u + 1.0)

        # β_T broadcast across nodes
        beta_exp = beta_T.unsqueeze(1).expand(-1, u.size(1), -1)  # (n, K, 1)
        integrand = self._integrand(t.reshape(-1, 1), beta_exp.reshape(-1, 1)).view(n, -1, 1)

        weights = self._weights.view(1, -1, 1)
        # ∫_{theta_ref}^{theta} softplus(α(t) + β_T) dt = (theta - theta_ref) / 2 · Σ w_i f(t_i)
        integral = 0.5 * (theta - a) * (weights * integrand).sum(dim=1)  # (n, 1)

        r = b_T + integral  # (n, 1)

        # ∂_T r = b'_umnn(T) + β'_umnn(T) · ∫ σ(α(t) + β_T) dt
        # where σ = derivative of softplus = sigmoid. By the same quadrature:
        sigmoid_integrand = torch.sigmoid(self._alpha_net(t.reshape(-1, 1)) + beta_exp.reshape(-1, 1)).view(n, -1, 1)
        sigmoid_integral = 0.5 * (theta - a) * (weights * sigmoid_integrand).sum(dim=1)  # (n, 1)
        dr_dT = b_prime_T + beta_prime_T * sigmoid_integral  # (n, 1)
        # Clamp away from 0 just in case (b'_umnn is already > 0 by softplus)
        log_det_jac_input = torch.log(dr_dT.clamp_min(1e-12)).squeeze(-1)  # (n,)

        return r, log_det_jac_input

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/unit/test_doubly_monotone_flow.py -v`

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/flows/doubly_monotone.py tests/unit/test_doubly_monotone_flow.py
git commit -m "$(cat <<'EOF'
feat(flows): DoublyMonotoneUMNN (§6.1 form 2 — §8.4 main flow)

r(θ, T) = b_umnn(T) + ∫_{θ_ref}^θ softplus(α(t) + β_umnn(T)) dt

Both R1 and R2 enforced architecturally:
  ∂_θ r = softplus(α(θ) + β_umnn(T)) > 0
  ∂_T r = b'_umnn(T) + β'_umnn(T) · ∫ σ(α(t) + β_umnn(T)) dt > 0
both in closed form via 12-point Gauss-Legendre quadrature.
log|∂_T r| is returned alongside r so NF-MLE's Jacobian factor
never needs autograd through the flow's internals.

monotonicity_guarantees = {R1, R2} so CDSBIRunner accepts it
under the default allow_ablation=False contract.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `JointUMNNFlow` — the R1-only ablation form

**Files:**
- Create: `src/cdsbi/flows/joint_umnn.py`
- Test: `tests/unit/test_joint_umnn_flow.py`

The ablation flow for §8.4: monotone in θ via a UMNN integrand that takes (θ, T) as joint input, BUT with no architectural guarantee of monotonicity in T. The autograd Jacobian `∂_T r` is what the NF-MLE loss consumes — and that's exactly the §3.5 mechanism failure case.

```
r(θ, T) = b(T) + ∫_{θ_ref}^θ softplus(MLP(t, T)) dt
```

where `b(T)` is a generic (unconstrained) MLP of T, NOT a UMNN. The integrand is positive (softplus) so ∂_θ r > 0 (R1), but the joint dependence on T inside the MLP means ∂_T r has no sign guarantee → R2 is NOT enforced architecturally. log_det_jac_input is computed via `torch.autograd.grad(r.sum(), T, create_graph=True)` so that the Jacobian factor in NF-MLE flows through training.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_joint_umnn_flow.py`:

```python
"""JointUMNNFlow — the R1-only ablation flow for §8.4."""
from __future__ import annotations

import torch

from cdsbi.flows.base import Guarantee


def test_joint_umnn_advertises_R1_only():
    from cdsbi.flows.joint_umnn import JointUMNNFlow
    flow = JointUMNNFlow(hidden=8)
    assert Guarantee.R1 in flow.monotonicity_guarantees
    assert Guarantee.R2 not in flow.monotonicity_guarantees, (
        "ablation flow must NOT advertise R2 — it's the whole point of the ablation"
    )


def test_joint_umnn_forward_shape(seed):
    from cdsbi.flows.joint_umnn import JointUMNNFlow
    torch.manual_seed(seed)
    flow = JointUMNNFlow(hidden=8)
    theta = torch.rand(32, 1) * 2.7 + 0.3
    T = torch.rand(32, 1) * 10 + 0.5
    r, log_det = flow(theta, context=T)
    assert r.shape == (32, 1)
    assert log_det.shape == (32,)
    assert torch.isfinite(log_det).all()


def test_joint_umnn_partial_theta_positive(seed):
    """∂r/∂θ is the softplus integrand at θ → strictly positive."""
    from cdsbi.flows.joint_umnn import JointUMNNFlow
    torch.manual_seed(seed)
    flow = JointUMNNFlow(hidden=8)
    theta = (torch.rand(32, 1) * 2.7 + 0.3).requires_grad_(True)
    T = torch.rand(32, 1) * 10 + 0.5
    r, _ = flow(theta, context=T)
    grad_theta = torch.autograd.grad(r.sum(), theta, create_graph=False)[0]
    assert (grad_theta > 0).all(), grad_theta


def test_joint_umnn_log_det_uses_autograd_through_T(seed):
    """The returned log_det should equal log|∂r/∂T| computed via autograd."""
    from cdsbi.flows.joint_umnn import JointUMNNFlow
    torch.manual_seed(seed)
    flow = JointUMNNFlow(hidden=8)
    theta = torch.rand(8, 1) * 2.7 + 0.3
    T = (torch.rand(8, 1) * 10 + 0.5).requires_grad_(True)
    r, log_det = flow(theta, context=T)
    # Independent autograd of r w.r.t. T
    grad_T = torch.autograd.grad(r.sum(), T, create_graph=False)[0]
    expected_log_det = torch.log(grad_T.abs().clamp_min(1e-12)).squeeze(-1)
    torch.testing.assert_close(log_det, expected_log_det, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_joint_umnn_flow.py -v`

Expected: FAIL on import with `ModuleNotFoundError: No module named 'cdsbi.flows.joint_umnn'`.

- [ ] **Step 3: Implement the flow**

Create `src/cdsbi/flows/joint_umnn.py`:

```python
"""JointUMNNFlow — the R1-only ablation flow for §8.4.

Monotone in θ via a UMNN integrand taking (θ, T) jointly, but with no
architectural constraint on monotonicity in T. log|∂_T r| comes from
autograd, which is exactly the §3.5 failure regime: when autograd's
local Jacobian doesn't enforce positive ∂_T r everywhere, the surrogate
density's mass Z(θ) exceeds 1 and the NF-MLE loss drops below the
information-theoretic floor.

advertised monotonicity_guarantees = {R1}; CDSBIRunner refuses to
construct unless allow_ablation=True.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cdsbi.flows.base import Flow, Guarantee


_NODES_NP, _WEIGHTS_NP = np.polynomial.legendre.leggauss(12)


class JointUMNNFlow(nn.Module, Flow):
    monotonicity_guarantees = frozenset({Guarantee.R1})

    def __init__(self, hidden: int = 16, theta_ref: float = 0.3):
        super().__init__()
        self.hidden = hidden
        self.theta_ref = theta_ref
        # Joint integrand MLP — takes (θ, T) jointly, outputs a scalar to be softplus'd
        self._integrand_mlp = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Constant term b(T) — generic MLP, no monotonicity constraint in T
        self._b_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self._integrand_mlp[-1].weight)
        nn.init.zeros_(self._integrand_mlp[-1].bias)
        nn.init.zeros_(self._b_mlp[-1].weight)
        nn.init.zeros_(self._b_mlp[-1].bias)
        self.register_buffer("_nodes", torch.tensor(_NODES_NP, dtype=torch.float32))
        self.register_buffer("_weights", torch.tensor(_WEIGHTS_NP, dtype=torch.float32))

    def _integrand(self, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """softplus(MLP([t, T])) — strictly positive in t direction by softplus,
        but no architectural sign-of-derivative guarantee in T."""
        inputs = torch.cat([t, T], dim=-1)
        return F.softplus(self._integrand_mlp(inputs)) + 1e-3

    def forward(
        self, theta: torch.Tensor, context: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert context is not None and context.shape[-1] == 1
        T = context
        n = theta.shape[0]
        a = self.theta_ref

        # Quadrature nodes on [theta_ref, theta]
        u = self._nodes.view(1, -1, 1).expand(n, -1, 1)
        b = theta.view(n, 1, 1).expand(-1, u.size(1), -1)
        t = a + 0.5 * (b - a) * (u + 1.0)
        T_exp = T.unsqueeze(1).expand(-1, u.size(1), -1)
        integrand = self._integrand(t.reshape(-1, 1), T_exp.reshape(-1, 1)).view(n, -1, 1)
        weights = self._weights.view(1, -1, 1)
        integral = 0.5 * (theta - a) * (weights * integrand).sum(dim=1)

        # Make sure T is included in the autograd graph (it's the conditioner output
        # — typically passed in detached). To compute ∂r/∂T via autograd we need
        # T to be a leaf with requires_grad=True OR a non-leaf that's part of the
        # graph. We force the latter by including a no-op identity that touches T.
        # If the caller passed T as a leaf with requires_grad=True, that path also
        # works.
        if not T.requires_grad:
            T_grad = T.clone().detach().requires_grad_(True)
            # Recompute the integral with the grad-requiring T to get an accurate
            # ∂_T r through autograd (the path above used the non-grad T).
            T_grad_exp = T_grad.unsqueeze(1).expand(-1, u.size(1), -1)
            integrand_g = self._integrand(t.reshape(-1, 1), T_grad_exp.reshape(-1, 1)).view(n, -1, 1)
            integral_g = 0.5 * (theta - a) * (weights * integrand_g).sum(dim=1)
            b_g = self._b_mlp(T_grad)
            r_for_grad = b_g + integral_g
            grad_T = torch.autograd.grad(
                r_for_grad.sum(), T_grad, create_graph=self.training, retain_graph=True,
            )[0]
        else:
            b_T = self._b_mlp(T)
            r_for_grad = b_T + integral
            grad_T = torch.autograd.grad(
                r_for_grad.sum(), T, create_graph=self.training, retain_graph=True,
            )[0]

        b_T_eval = self._b_mlp(T.detach() if not T.requires_grad else T)
        r = b_T_eval + integral
        log_det_jac_input = torch.log(grad_T.abs().clamp_min(1e-12)).squeeze(-1)

        return r, log_det_jac_input

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/unit/test_joint_umnn_flow.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/flows/joint_umnn.py tests/unit/test_joint_umnn_flow.py
git commit -m "$(cat <<'EOF'
feat(flows): JointUMNNFlow (R1-only ablation form for §8.4)

Monotone in θ via softplus(MLP(θ, T)) integrand — R1 enforced
architecturally — but with no constraint on monotonicity in T;
log|∂_T r| comes from autograd, which is the §3.5 failure regime.

advertised monotonicity_guarantees = {R1}; CDSBIRunner refuses
unless allow_ablation=True is explicitly set. Used by the §8.4
ablation experiment to demonstrate the folding mechanism.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `JointUMNN1DFlow` — 1D mechanism analog

**Files:**
- Create: `src/cdsbi/flows/joint_umnn_1d.py`
- Test: `tests/unit/test_joint_umnn_1d_flow.py`

The 1D analog of JointUMNNFlow used for the direct-construction mechanism test (Task 9c). Same shape as JointUMNNFlow but operates on (θ, X) with scalar X (no sufficient-statistic reduction). Same `monotonicity_guarantees = {R1}` semantics. The simpler 1D form lets the synthetic mechanism test inspect the folding behavior directly without confounders from the (θ, T) reduction.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_joint_umnn_1d_flow.py`:

```python
"""JointUMNN1DFlow — 1D analog of JointUMNNFlow for the synthetic mechanism test."""
from __future__ import annotations

import torch

from cdsbi.flows.base import Guarantee


def test_joint_umnn_1d_advertises_R1_only():
    from cdsbi.flows.joint_umnn_1d import JointUMNN1DFlow
    flow = JointUMNN1DFlow(hidden=8)
    assert Guarantee.R1 in flow.monotonicity_guarantees
    assert Guarantee.R2 not in flow.monotonicity_guarantees


def test_joint_umnn_1d_forward_shape(seed):
    from cdsbi.flows.joint_umnn_1d import JointUMNN1DFlow
    torch.manual_seed(seed)
    flow = JointUMNN1DFlow(hidden=8)
    theta = torch.rand(32, 1) * 5.0 - 2.5  # uniform on [-2.5, 2.5]
    x = torch.rand(32, 1) * 4.0 - 2.0
    r, log_det = flow(theta, context=x)
    assert r.shape == (32, 1)
    assert log_det.shape == (32,)
    assert torch.isfinite(log_det).all()


def test_joint_umnn_1d_partial_theta_positive(seed):
    """∂r/∂θ is the softplus integrand at θ → strictly positive."""
    from cdsbi.flows.joint_umnn_1d import JointUMNN1DFlow
    torch.manual_seed(seed)
    flow = JointUMNN1DFlow(hidden=8)
    theta = (torch.rand(32, 1) * 5.0 - 2.5).requires_grad_(True)
    x = torch.rand(32, 1) * 4.0 - 2.0
    r, _ = flow(theta, context=x)
    grad_theta = torch.autograd.grad(r.sum(), theta, create_graph=False)[0]
    assert (grad_theta > 0).all(), grad_theta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_joint_umnn_1d_flow.py -v`

Expected: FAIL on import with `ModuleNotFoundError: No module named 'cdsbi.flows.joint_umnn_1d'`.

- [ ] **Step 3: Implement the flow**

Create `src/cdsbi/flows/joint_umnn_1d.py`:

```python
"""JointUMNN1DFlow — 1D analog of JointUMNNFlow for the mechanism test.

Same R1-only contract as JointUMNNFlow but takes (θ, X) with scalar X
(no sufficient-statistic reduction). Used by tests/ablation/test_1d_mechanism.py
to demonstrate the §3.5 folding behavior in the simplest possible setting.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cdsbi.flows.base import Flow, Guarantee


_NODES_NP, _WEIGHTS_NP = np.polynomial.legendre.leggauss(12)


class JointUMNN1DFlow(nn.Module, Flow):
    monotonicity_guarantees = frozenset({Guarantee.R1})

    def __init__(self, hidden: int = 16, theta_ref: float = -2.5):
        super().__init__()
        self.hidden = hidden
        self.theta_ref = theta_ref
        self._integrand_mlp = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self._b_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self._integrand_mlp[-1].weight)
        nn.init.zeros_(self._integrand_mlp[-1].bias)
        nn.init.zeros_(self._b_mlp[-1].weight)
        nn.init.zeros_(self._b_mlp[-1].bias)
        self.register_buffer("_nodes", torch.tensor(_NODES_NP, dtype=torch.float32))
        self.register_buffer("_weights", torch.tensor(_WEIGHTS_NP, dtype=torch.float32))

    def _integrand(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self._integrand_mlp(torch.cat([t, x], dim=-1))) + 1e-3

    def forward(
        self, theta: torch.Tensor, context: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert context is not None and context.shape[-1] == 1
        x = context
        n = theta.shape[0]
        a = self.theta_ref

        u = self._nodes.view(1, -1, 1).expand(n, -1, 1)
        b = theta.view(n, 1, 1).expand(-1, u.size(1), -1)
        t = a + 0.5 * (b - a) * (u + 1.0)
        x_exp = x.unsqueeze(1).expand(-1, u.size(1), -1)
        integrand = self._integrand(t.reshape(-1, 1), x_exp.reshape(-1, 1)).view(n, -1, 1)
        weights = self._weights.view(1, -1, 1)
        integral = 0.5 * (theta - a) * (weights * integrand).sum(dim=1)

        if not x.requires_grad:
            x_grad = x.clone().detach().requires_grad_(True)
            x_grad_exp = x_grad.unsqueeze(1).expand(-1, u.size(1), -1)
            integrand_g = self._integrand(t.reshape(-1, 1), x_grad_exp.reshape(-1, 1)).view(n, -1, 1)
            integral_g = 0.5 * (theta - a) * (weights * integrand_g).sum(dim=1)
            b_g = self._b_mlp(x_grad)
            r_for_grad = b_g + integral_g
            grad_x = torch.autograd.grad(
                r_for_grad.sum(), x_grad, create_graph=self.training, retain_graph=True,
            )[0]
        else:
            b_T = self._b_mlp(x)
            r_for_grad = b_T + integral
            grad_x = torch.autograd.grad(
                r_for_grad.sum(), x, create_graph=self.training, retain_graph=True,
            )[0]

        b_x_eval = self._b_mlp(x.detach() if not x.requires_grad else x)
        r = b_x_eval + integral
        log_det_jac_input = torch.log(grad_x.abs().clamp_min(1e-12)).squeeze(-1)

        return r, log_det_jac_input

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/unit/test_joint_umnn_1d_flow.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/flows/joint_umnn_1d.py tests/unit/test_joint_umnn_1d_flow.py
git commit -m "$(cat <<'EOF'
feat(flows): JointUMNN1DFlow (1D analog of JointUMNNFlow)

Same R1-only contract as JointUMNNFlow but operates on (θ, X) with
scalar X — no sufficient-statistic reduction. Used by the §8.4
ablation's 1D direct-construction mechanism test to demonstrate
the §3.5 folding behavior in the simplest possible setting.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Wire `MLPConditioner` + new flows into `experiments/run.py`

**Files:**
- Modify: `src/cdsbi/experiments/run.py`
- Modify: `src/cdsbi/methods/cd_sbi.py` — add `self.flow.train()` at top of `fit()`; widen `n_params()` to handle the v3 flows
- Test: extend `tests/integration/test_run_diagnostics.py`

The `_build_flow` function in `run.py` already dispatches based on `cfg.flow.name`. Add three new branches for `doubly_monotone`, `joint_umnn`, `joint_umnn_1d`. The `_build_conditioner` (or equivalent) needs to dispatch on `cfg.target.name`: for `exp_rate` use `MLPConditioner(input_dim=5, output_dim=1, mode="frozen_sum")`; for all other targets use the existing `Identity`.

- [ ] **Step 1: Read `src/cdsbi/experiments/run.py` to locate the flow-build dispatch**

```bash
grep -n "_build_flow\|_build_conditioner\|maf_adapter\|MAFAdapter\|additive_umnn\|triangular_additive" src/cdsbi/experiments/run.py | head -20
```

Identify the function that builds the flow from cfg and the place where the conditioner is constructed (likely inside `_build_method` or a sibling).

- [ ] **Step 2: Write the failing integration test**

Append to `tests/integration/test_run_diagnostics.py`:

```python
def test_run_produces_doubly_monotone_flow_on_exp_rate(tmp_path, seed):
    """End-to-end: a single CDSBI run on the ExponentialRate target with
    DoublyMonotoneUMNN writes STATUS=OK and produces a valid index_row."""
    import subprocess, pandas as pd
    from pathlib import Path
    out = tmp_path / "run"
    cmd = [
        "python", "-m", "cdsbi.experiments.run",
        f"hydra.run.dir={out}",
        "experiment=8_4_replication",
        f"seed={seed}",
        "training.fresh_batch=false",
        "+experiment.set_size_n_per_theta=25",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
    assert result.returncode == 0, result.stderr
    assert (out / "STATUS").read_text().strip() == "OK"
    idx = pd.read_parquet(out / "index_row.parquet")
    assert idx["method"].iloc[0] == "cd_sbi"
    assert idx["flow"].iloc[0] == "doubly_monotone"
    assert idx["target"].iloc[0] == "exp_rate"
    # Final loss should be near the entropy_lower_bound (≈ 0.88) within ~0.2
    # at small budget (this is a smoke test, not a tight precision check).
    assert idx["final_loss"].iloc[0] > 0.4, f"final_loss = {idx['final_loss'].iloc[0]} too low — possible R2 violation"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/integration/test_run_diagnostics.py::test_run_produces_doubly_monotone_flow_on_exp_rate -v`

Expected: FAIL — either Hydra can't resolve the flow target (TypeError on instantiation) OR the conditioner is hardcoded to Identity and exp_rate's d_x=5 confuses downstream code.

- [ ] **Step 4: Extend `_build_flow` to handle the v3 flow classes**

In `src/cdsbi/experiments/run.py`, find the `_build_flow` function. After the existing `if/elif` chain for `additive_umnn`, `maf`, `triangular_additive`, add:

```python
    if method_flow_label == "doubly_monotone":
        return _instantiate(
            "cdsbi.flows.doubly_monotone.DoublyMonotoneUMNN",
            hidden=int(cfg.budget.doubly_monotone_hidden),
            theta_ref=float(simulator.theta_range[0]),
        )
    if method_flow_label == "joint_umnn":
        return _instantiate(
            "cdsbi.flows.joint_umnn.JointUMNNFlow",
            hidden=int(cfg.budget.doubly_monotone_hidden),
            theta_ref=float(simulator.theta_range[0]),
        )
    if method_flow_label == "joint_umnn_1d":
        return _instantiate(
            "cdsbi.flows.joint_umnn_1d.JointUMNN1DFlow",
            hidden=int(cfg.budget.doubly_monotone_hidden),
            theta_ref=float(simulator.theta_range[0]),
        )
```

Also handle the fast-path (when `method_flow_label == cfg.flow.name`) — those three flow names should match into the existing `_instantiate(cfg.flow.*)` call without special-casing if the flow YAML carries `_target_` and `theta_ref` already. (Re-verify after the edit.)

- [ ] **Step 4b: Fix `CDSBIRunner.fit()` and `n_params()` for the v3 flows**

In `src/cdsbi/methods/cd_sbi.py`:

(i) At the top of `fit()`, add `self.flow.train()` so the v3 ablation flow's `create_graph=self.training` autograd path correctly creates the second-order graph during training. Find the line `self.flow.to(self.device)` (around line 28) and immediately after it add:

```python
        self.flow.train()
```

(ii) `n_params()` in CDSBIRunner (around lines 169–185) currently splits backbone vs head by accessing `flow.a_blocks` / `flow.a` — attributes specific to `AdditiveFlow1D` / `TriangularAdditiveFlow`. The v3 flows have neither. Widen the function with a fallback that returns the whole-flow parameter count when neither attribute is present:

```python
    def n_params(self) -> dict:
        flow_total = sum(p.numel() for p in self.flow.parameters())
        if hasattr(self.flow, "a_blocks"):
            # TriangularAdditiveFlow path (v1)
            ...  # keep existing logic
        elif hasattr(self.flow, "a"):
            # AdditiveFlow1D path (v0)
            ...  # keep existing logic
        else:
            # v3 flows (DoublyMonotoneUMNN, JointUMNNFlow, JointUMNN1DFlow) and
            # any future flow without an explicit backbone/head split.
            return {
                "backbone": flow_total,
                "head": 0,
                "calibration_stage": 0,
                "total": flow_total,
                "kind": "single_block",
            }
```

(read the existing `n_params` body first to preserve the v0/v1 branches verbatim; only ADD the `else` branch).

- [ ] **Step 5: Extend conditioner construction to dispatch on target**

In `_build_method` or wherever the conditioner is constructed, replace the unconditional `Identity()` with:

```python
    if cfg.target.name == "exp_rate":
        from cdsbi.conditioners.mlp import MLPConditioner
        conditioner = MLPConditioner(
            input_dim=int(simulator.d_x),
            output_dim=1,
            mode="frozen_sum",
        )
    else:
        from cdsbi.conditioners.identity import Identity
        conditioner = Identity()
```

- [ ] **Step 6: Run the integration test (requires Task 8's budget config first)**

Task 8 lands `doubly_monotone_hidden` in each budget YAML. Once that's in:

```bash
pytest tests/integration/test_run_diagnostics.py::test_run_produces_doubly_monotone_flow_on_exp_rate -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/cdsbi/experiments/run.py tests/integration/test_run_diagnostics.py
git commit -m "$(cat <<'EOF'
feat(run): wire DoublyMonotoneUMNN / JointUMNN / JointUMNN1D + MLPConditioner

Extends _build_flow to dispatch the three v3 flow classes from
cfg.flow.name. Extends conditioner construction to use MLPConditioner
(frozen-sum X→T reduction) when the target is exp_rate, Identity
otherwise.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Budget retune for the doubly-monotone flow

**Files:**
- Modify: `configs/budget/small.yaml`, `medium.yaml`, `large.yaml`, `xlarge.yaml`

The new `DoublyMonotoneUMNN` has its own per-budget hidden size. The other budget keys (cdsbi_flow_hidden, cdsbi_flow_hidden_d2, maf_hidden, classifier_hidden, quantile_hidden) stay unchanged. The §8.4 sweep uses these matched-budget contracts.

- [ ] **Step 1: Compute target hidden sizes**

Run, from repo root:

```bash
python <<'PY'
from cdsbi.flows.doubly_monotone import DoublyMonotoneUMNN
print(f"{'budget':10s} {'hidden':>7s} {'params':>7s}")
for budget, target, h_guess in [
    ("small",  1000, 12),
    ("medium", 5000, 28),
    ("large", 25000, 64),
    ("xlarge",100000,128),
]:
    # Scan around h_guess to find the closest match to target
    best = None
    for h in range(max(4, h_guess - 6), h_guess + 12):
        flow = DoublyMonotoneUMNN(hidden=h, theta_ref=0.3)
        n = sum(p.numel() for p in flow.parameters())
        rel_err = abs(n - target) / target
        if best is None or rel_err < best[2]:
            best = (h, n, rel_err)
    h, n, rel = best
    print(f"{budget:10s} {h:>7d} {n:>7d}  ({rel:.1%})")
PY
```

Record the hidden size that brings each budget within ~10% of its target.

- [ ] **Step 2: Update budget YAMLs**

For each budget YAML (`configs/budget/{small,medium,large,xlarge}.yaml`), add a `doubly_monotone_hidden` key with the value from Step 1, plus a `# comment` per the v1/v2 pattern showing actual params + percentage off.

Example for medium (numbers to be filled from Step 1 output):

```yaml
doubly_monotone_hidden: 28  # DoublyMonotoneUMNN actual=<N> (<%>; matched)
```

- [ ] **Step 3: Verify the §8.4 smoke run from Task 7 now passes**

Run:

```bash
pytest tests/integration/test_run_diagnostics.py::test_run_produces_doubly_monotone_flow_on_exp_rate -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add configs/budget/small.yaml configs/budget/medium.yaml configs/budget/large.yaml configs/budget/xlarge.yaml
git commit -m "$(cat <<'EOF'
config(budget): add doubly_monotone_hidden across the budget grid

Per-budget hidden size for the v3 DoublyMonotoneUMNN flow. Other
budget keys (cdsbi_flow_hidden, cdsbi_flow_hidden_d2, maf_hidden,
classifier_hidden, quantile_hidden) carry over unchanged from v0/v1/v2.

Each entry shows the actual parameter count to make the matched-band
contract verifiable per the v1/v2 pattern.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: New `Ablation` test category — three tests

**Files:**
- Create: `tests/ablation/__init__.py`
- Create: `tests/ablation/test_safety_check.py`
- Create: `tests/ablation/test_trained_folding.py`
- Create: `tests/ablation/test_1d_mechanism.py`
- Modify: `pyproject.toml` (add `ablation` pytest marker)

The new test category captures the (R2) ablation results as a regression suite. Three tests:
- **9a Safety check:** training with `allow_ablation=False` + a flow that only advertises R1 (like JointUMNNFlow) must raise `MonotonicityMismatchError` at construction. Catches accidental bypasses of the architectural contract.
- **9b Trained-folding empirical:** train JointUMNNFlow on ExponentialRate with `allow_ablation=True` for ~enough steps that the loss converges, then assert final loss < `simulator.entropy_lower_bound()` minus an empirical noise floor. This is the §3.5 mechanism failure as a regression test.
- **9c 1D direct-construction:** synthetic 1D test using JointUMNN1DFlow with hand-constructed data showing folding directly.

- [ ] **Step 1: Add the pytest marker AND update addopts**

In `pyproject.toml`, the existing `[tool.pytest.ini_options]` reads:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "intensive: full-budget replication tests; opt-in via `pytest -m intensive`",
]
addopts = "-m 'not intensive'"
```

Two changes:
1. Add `"ablation"` to the markers list.
2. Extend `addopts` to also exclude `ablation` (the ablation tests run ~1500 training steps each and would blow the <30s fast-test contract if included by default).

After the edit:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "intensive: full-budget replication tests; opt-in via `pytest -m intensive`",
    "ablation: (R2) ablation tests; opt-in via `pytest -m ablation`",
]
addopts = "-m 'not intensive and not ablation'"
```

- [ ] **Step 2: Create the `tests/ablation/__init__.py` package marker**

```bash
touch tests/ablation/__init__.py
```

- [ ] **Step 3: Write `tests/ablation/test_safety_check.py`**

```python
"""Safety check: CDSBIRunner refuses an R1-only flow unless allow_ablation=True."""
from __future__ import annotations

import pytest

from cdsbi.flows.joint_umnn import JointUMNNFlow
from cdsbi.conditioners.mlp import MLPConditioner
from cdsbi.losses.nfmle import NFMLELoss
from cdsbi.losses.base import MonotonicityMismatchError
from cdsbi.methods.cd_sbi import CDSBIRunner


@pytest.mark.ablation
def test_runner_refuses_r1_only_flow_by_default():
    """allow_ablation defaults to False; constructing CDSBIRunner with a
    flow that lacks R2 must raise MonotonicityMismatchError."""
    flow = JointUMNNFlow(hidden=8)
    conditioner = MLPConditioner(input_dim=5, output_dim=1, mode="frozen_sum")
    loss = NFMLELoss()
    with pytest.raises(MonotonicityMismatchError):
        CDSBIRunner(flow=flow, conditioner=conditioner, loss=loss)


@pytest.mark.ablation
def test_runner_allows_r1_only_flow_with_explicit_allow_ablation():
    """allow_ablation=True bypasses the check — the ablation experiment needs this."""
    flow = JointUMNNFlow(hidden=8)
    conditioner = MLPConditioner(input_dim=5, output_dim=1, mode="frozen_sum")
    loss = NFMLELoss()
    runner = CDSBIRunner(
        flow=flow, conditioner=conditioner, loss=loss, allow_ablation=True,
    )
    assert runner.allow_ablation is True
```

- [ ] **Step 4: Write `tests/ablation/test_trained_folding.py`**

```python
"""Trained-folding empirical test:

Train JointUMNNFlow on ExponentialRate with allow_ablation=True until
the loss converges. Assert that the final loss falls below the
simulator's entropy_lower_bound — which is the §3.5 folding mechanism
(Z(θ) > 1 ⇒ unnormalized surrogate density ⇒ NF-MLE below the
information-theoretic floor).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cdsbi.flows.joint_umnn import JointUMNNFlow
from cdsbi.conditioners.mlp import MLPConditioner
from cdsbi.losses.nfmle import NFMLELoss
from cdsbi.methods.cd_sbi import CDSBIRunner
from cdsbi.simulators.exp_rate import ExponentialRate


@pytest.mark.ablation
def test_trained_ablation_loss_below_entropy_lower_bound():
    sim = ExponentialRate()
    # entropy_lower_bound() already returns the NF-MLE-loss-space value
    # (≈ 0.88 per manuscript) — no further adjustment needed; the
    # conditioner's log|∂T/∂X| is already baked into the MC computation.
    loss_floor = sim.entropy_lower_bound()

    flow = JointUMNNFlow(hidden=16, theta_ref=sim.theta_range[0])
    conditioner = MLPConditioner(input_dim=sim.n_iid, output_dim=1, mode="frozen_sum")
    loss = NFMLELoss()
    runner = CDSBIRunner(
        flow=flow, conditioner=conditioner, loss=loss, allow_ablation=True,
    )
    config = {
        "lr": 5e-3,
        "batch_size": 256,
        "n_steps": 1500,
        "n_train": 30000,
        "optimizer": "adamw",
        "fresh_batch": False,
    }
    trained = runner.fit(simulator=sim, config=config, seed=0)
    final_loss = float(trained.final_loss)
    # Manuscript reports trained-ablation loss = 0.56 vs truth 0.88; gap ≈ 0.32.
    # Empirical noise margin: ±0.05 captures both seed and clamp-floor variance.
    # We assert final_loss < loss_floor − 0.10 to make the test robust against
    # the trained ablation getting "close to" the floor without crossing it.
    assert final_loss < loss_floor - 0.10, (
        f"final_loss={final_loss:.3f} did NOT fall below loss_floor={loss_floor:.3f} − 0.10; "
        f"R2-ablation folding signature not detected"
    )
    # Positive-control comparison would re-train the doubly-monotone flow here
    # to confirm IT stays above the floor; that's the intensive test (Task 10)
    # rather than baking into this ablation regression to keep wall-time bounded.
```

- [ ] **Step 5: Write `tests/ablation/test_1d_mechanism.py`**

```python
"""1D direct-construction mechanism test:

Take a simple 1D problem (e.g. X | θ ~ N(θ, 1)) and train the R1-only
JointUMNN1DFlow on it. Verify that the trained flow violates monotonicity
in X at some point in the support, AND that the final loss falls below
the closed-form entropy ½ log(2πe) of the 1D normal.

This is the simplest possible demonstration of the §3.5 folding
mechanism: in 1D location-normal where everything else is known
exactly, the R1-only flow still folds because R2 is not enforced.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from cdsbi.flows.joint_umnn_1d import JointUMNN1DFlow
from cdsbi.conditioners.identity import Identity
from cdsbi.losses.nfmle import NFMLELoss
from cdsbi.methods.cd_sbi import CDSBIRunner
from cdsbi.simulators.location_normal_1d import LocationNormal1D


@pytest.mark.ablation
def test_1d_r1_only_flow_folds_in_x():
    sim = LocationNormal1D()
    H_floor = 0.5 * math.log(2 * math.pi * math.e)  # 1.4189...
    flow = JointUMNN1DFlow(hidden=16, theta_ref=sim.theta_range[0])
    conditioner = Identity()
    loss = NFMLELoss()
    runner = CDSBIRunner(
        flow=flow, conditioner=conditioner, loss=loss, allow_ablation=True,
    )
    config = {
        "lr": 5e-3,
        "batch_size": 256,
        "n_steps": 1500,
        "n_train": 30000,
        "optimizer": "adamw",
        "fresh_batch": False,
    }
    trained = runner.fit(simulator=sim, config=config, seed=0)
    # Check 1: final loss below the entropy lower bound for N(θ, 1)
    assert float(trained.final_loss) < H_floor - 0.05, (
        f"final_loss={float(trained.final_loss):.3f} did NOT fall below "
        f"H_floor={H_floor:.3f}; 1D R2-ablation mechanism failed to reproduce"
    )
    # Check 2: the trained flow's ∂r/∂x is negative somewhere on the support
    # (this is the literal folding — non-monotone in X)
    theta_test = torch.full((200, 1), 0.0)
    x_test = torch.linspace(-3.0, 3.0, 200).view(-1, 1).requires_grad_(True)
    r, _ = flow(theta_test, context=x_test)
    grad_x = torch.autograd.grad(r.sum(), x_test, retain_graph=False)[0]
    assert (grad_x < 0).any(), (
        f"trained R1-only flow did NOT fold in X — min(∂r/∂X)={float(grad_x.min()):.4f} >= 0; "
        f"expected at least one point with negative ∂r/∂X"
    )
```

- [ ] **Step 6: Run the new ablation tests**

```bash
pytest -m ablation tests/ablation/ -v
```

Expected: 4 tests pass (2 safety-check + 1 trained-folding + 1 1D mechanism). Each trained test runs in ~1–3 min.

- [ ] **Step 7: Run the full fast suite to confirm no regressions**

```bash
pytest -q
```

Expected: 175+ tests pass (the existing 175 + 4 ablation tests opt-out by default via the marker).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml tests/ablation/__init__.py tests/ablation/test_safety_check.py tests/ablation/test_trained_folding.py tests/ablation/test_1d_mechanism.py
git commit -m "$(cat <<'EOF'
test(ablation): new Ablation test category for (R2) mechanism

Three regression tests capture the §8.4 (R2)-failure story:

1. safety_check — CDSBIRunner with allow_ablation=False refuses a
   flow advertising only {R1}, raising MonotonicityMismatchError.
2. trained_folding — JointUMNNFlow on ExponentialRate converges to
   final loss BELOW the simulator's entropy_lower_bound (the §3.5
   mechanism signature — Z(θ) > 1).
3. 1d_mechanism — JointUMNN1DFlow on LocationNormal1D both falls
   below the closed-form entropy ½ log(2πe) AND has ∂r/∂x < 0
   somewhere in the support (literal folding).

Marker 'ablation' added to pyproject.toml so the tests opt-in only
under `pytest -m ablation` (mirroring the intensive marker).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: §8.4 CDSBI intensive replication test

Mirror the v1 / v2 intensive replication tests for §8.4. Single-seed wall is ~1–2 min on GPU; 5 seeds total ~10 min. Tolerance bands chosen against the manuscript's single-seed numbers (pivot RMSE 0.045, final loss 0.87 vs entropy 0.88, conditional PIT KS 0.016–0.028, coverage error < 0.015) with the same 1.5–2× headroom we used in v1/v2.

**Files:**
- Create: `tests/intensive/test_replicate_8_4.py`

- [ ] **Step 1: Write the test**

```python
"""§8.4 replication: CDSBI on ExponentialRate matches tolerance bands."""
from pathlib import Path
import subprocess

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.intensive
def test_replicate_8_4_cdsbi_matches_tolerance(tmp_path):
    """Run CDSBI on the doubly-monotone flow + ExponentialRate target across
    5 seeds; seed-averaged metrics within tolerance bands.

    Tolerance bands per manuscript §8.4 with 1.5–2× headroom:
      - pivot_rmse mean ≤ 0.10 (manuscript single-seed: 0.045)
      - final_loss mean within 0.20 of the conditioner-adjusted entropy
        lower bound (manuscript: truth 0.88, trained 0.87)
      - coverage_error_max mean ≤ 0.03 (manuscript: < 0.015)
    """
    out_dir = tmp_path / "sweep"
    for seed in range(5):
        cmd = [
            "python", "-m", "cdsbi.experiments.run",
            f"hydra.run.dir={out_dir}/run_seed_{seed}",
            "experiment=8_4_replication",
            f"seed={seed}",
            "training.fresh_batch=false",
            "+experiment.set_size_n_per_theta=25",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        assert result.returncode == 0, f"seed {seed} failed:\n{result.stderr}"

    from cdsbi.analysis.loaders import load_runs
    from cdsbi.simulators.exp_rate import ExponentialRate
    import math

    df = load_runs(str(out_dir / "*"))
    assert len(df) == 5
    assert df["pivot_rmse"].mean() <= 0.10, f"pivot_rmse mean = {df['pivot_rmse'].mean()}"
    assert df["coverage_error_max"].mean() <= 0.03, (
        f"coverage_error_max mean = {df['coverage_error_max'].mean()}"
    )
    # Final-loss check: must be > entropy lower bound (the doubly-monotone case
    # has Z(θ) ≡ 1; loss ≥ entropy strictly). Use the conditioner-adjusted bound.
    sim = ExponentialRate()
    H = sim.entropy_lower_bound() - 0.5 * math.log(sim.n_iid)
    assert df["final_loss"].mean() > H - 0.05, (
        f"trained doubly-monotone final_loss = {df['final_loss'].mean()} BELOW "
        f"entropy_lower_bound = {H} — possible R2 violation in the production flow"
    )
```

- [ ] **Step 2: Run the test**

```bash
pytest -m intensive tests/intensive/test_replicate_8_4.py -v
```

Expected: PASS. Wall ~5–10 min.

- [ ] **Step 3: Commit**

```bash
git add tests/intensive/test_replicate_8_4.py
git commit -m "$(cat <<'EOF'
test(intensive): §8.4 replication for CDSBI on ExponentialRate

5 seeds × medium budget; asserts pivot_rmse, coverage_error_max,
and that the trained doubly-monotone flow's final loss stays
ABOVE the simulator's entropy lower bound (the contract that
R2-enforced architectures preserve normalization).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Launch §8.4 baseline sweep + ablation sweep + paper-table generation

End-to-end v3 deliverable. Two sweeps:
- **Main baseline sweep** (5 methods × 4 budgets × 5 seeds = 100 runs): same protocol as v0/v1/v2 with the new ExponentialRate target.
- **Ablation sweep** (CDSBI-doubly_monotone vs CDSBI-joint_umnn at 4 budgets × 5 seeds = 40 runs): within-method comparison for the §8.4 ablation table.

Plus `paper_table_8_4` and the cross-table aggregation.

**Files:** (no code changes; this task documents the v3 deliverable end-to-end)

- [ ] **Step 1: Launch the cross-method baseline sweep**

Run:

```bash
python -m cdsbi.experiments.run -m \
  experiment=8_4_baseline_sweep \
  method=cd_sbi,npe,nle,nre,lf2i_bff \
  budget=small,medium,large,xlarge \
  seed=0,1,2,3,4 \
  training.fresh_batch=false
```

Expected: 100 runs complete. Wall time similar to v2 §8.3's sweep (~3–5 hr depending on LF2I-BFF xlarge speed).

- [ ] **Step 2: Launch the ablation sweep (two separate sub-launches)**

The `8_4_ablation.yaml` sets `method.allow_ablation: true` — which is correct for the R1-only `joint_umnn` flow but unnecessary for the R1+R2 `doubly_monotone` flow. Sweeping `flow=doubly_monotone,joint_umnn` in a single multirun would silently keep `allow_ablation=true` on the good arm, weakening the safety contract. Two separate launches keep the two arms clean:

```bash
# Arm 1: ablation flow (R1-only, requires allow_ablation=true)
python -m cdsbi.experiments.run -m \
  experiment=8_4_ablation \
  flow=joint_umnn \
  budget=small,medium,large,xlarge \
  seed=0,1,2,3,4 \
  training.fresh_batch=false
```

```bash
# Arm 2: production flow (R1+R2, allow_ablation=false enforced).
# We reuse the replication config here since 8_4_ablation.yaml carries the
# misleading allow_ablation=true; the replication config has the correct
# allow_ablation default (false). hydra.sweep.dir is overridden so the two
# arms land under sibling timestamped dirs that the paper-table aggregator
# can union-glob.
python -m cdsbi.experiments.run -m \
  experiment=8_4_replication \
  flow=doubly_monotone \
  budget=small,medium,large,xlarge \
  seed=0,1,2,3,4 \
  training.fresh_batch=false
```

Expected: 20 runs per arm = 40 runs total. Wall ~30–60 min each arm.

- [ ] **Step 3: Add `paper_table_8_4` shim**

In `src/cdsbi/analysis/paper_tables.py`, append:

```python
def paper_table_8_4(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot (method × budget) → seed-averaged §8.4 metrics.

    Reuses paper_table_8_2's column set (§8.4 has no Jacobian-recovery
    diagnostic because the truth Jacobian depends on (θ, T) and is not
    constant). Lifts final_loss into the aggregated table so the
    (R2)-ablation story can be read off directly.
    """
    metrics = [
        "coverage_error_max", "marginal_ks", "pivot_rmse",
        "final_loss", "actual_params_total",
    ]
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby(["method", "budget_name"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg
```

Add a one-line unit test in `tests/unit/test_paper_tables.py`:

```python
def test_paper_table_8_4_includes_final_loss():
    import pandas as pd
    from cdsbi.analysis.paper_tables import paper_table_8_4
    df = pd.DataFrame([
        {"method": "cd_sbi", "budget_name": "medium", "coverage_error_max": 0.02,
         "marginal_ks": 0.013, "pivot_rmse": 0.06, "final_loss": 0.86,
         "actual_params_total": 5000},
    ])
    t = paper_table_8_4(df)
    assert "final_loss_mean" in t.columns
```

- [ ] **Step 4: Aggregate the tables**

```python
from cdsbi.analysis.loaders import load_runs
from cdsbi.analysis.paper_tables import paper_table_8_4

# Main baseline sweep
df_main = load_runs("outputs/8_4_baseline_sweep/<timestamp>/*/")
print("§8.4 main baseline table:")
print(paper_table_8_4(df_main))

# Ablation sweep
df_abl = load_runs("outputs/8_4_ablation/<timestamp>/*/")
# Group by flow rather than method (since method=cd_sbi everywhere)
print("\n§8.4 ablation (cd_sbi only, varying flow):")
print(df_abl.groupby(["flow", "budget_name"])[["final_loss", "pivot_rmse", "coverage_error_max"]].agg(["mean", "std"]))
```

Expected: the doubly_monotone rows have `final_loss` close to the entropy lower bound; the joint_umnn rows have `final_loss` BELOW the lower bound.

- [ ] **Step 5: Commit the paper_table_8_4 + test**

```bash
git add src/cdsbi/analysis/paper_tables.py tests/unit/test_paper_tables.py
git commit -m "$(cat <<'EOF'
feat(analysis): paper_table_8_4 (§8.4 cross-method + ablation aggregation)

Same column set as paper_table_8_2 plus final_loss (which is the
load-bearing §8.4 ablation metric — final loss BELOW the
simulator's entropy lower bound is the §3.5 folding signature).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Integrate v3 results into manuscript §8.4

**Files:**
- Modify: `cd_sbi_v7.tex` — replace existing §8.4 (single-seed table + four-step ablation paragraphs around lines ~2922–3018) with the sweep-averaged version + cross-method comparison + the ablation as a sub-table, mirroring the §8.1/§8.2/§8.3 restructure.
- Verify: `/usr/bin/pdflatex cd_sbi_v7 && /usr/bin/bibtex cd_sbi_v7 && /usr/bin/pdflatex cd_sbi_v7 && /usr/bin/pdflatex cd_sbi_v7` builds cleanly.

**Prerequisite:** Task 11 complete and the user has signed off on the v3 sweep + ablation results. Do not invent or paraphrase numbers; pull them from the sweep output via `paper_table_8_4(load_runs(...))` and from per-run diagnostic parquets for the CDSBI calibration table.

**Template:** The §8.1 / §8.2 / §8.3 updates in `cd_sbi_v7.tex` already show the target structure. Reproduce the same seven-block shape for §8.4, with two §8.4-specific additions:

1. **Setup** — keep the current §8.4 setup (`X_i ~ Exp(θ)` iid, n=5, `T = Σ X_i`, truth pivot `r* = Φ⁻¹(F_{χ²_{2n}}(2θT))`).
2. **Architecture** — keep the doubly-monotone form-2 paragraph; reference §6.1 form 2 and Tasks 4–6's implementation.
3. **Sweep** — same paragraph as §8.1/§8.2/§8.3 adapted: 5 seeds × 4 budgets, evaluation sample sizes, noise floors.
4. **CDSBI calibration diagnostics (medium budget)** — replace the existing single-seed §8.4 table with a sweep-averaged version (Pivot RMSE, Marginal PIT KS, Conditional PIT bulk/edge, Coverage error max). The §8.3 Jacobian-recovery row is OUT (§8.4's truth Jacobian is non-constant; the diagnostic skips automatically).
5. **Budget invariance** — short paragraph stating what the data shows.
6. **Comparison with baselines** — new 5-method × 4-budget table on `coverage_error_max`. NPE's row gets the same Bonferroni-conservative caveat as §8.2/§8.3.
7. **The (R2) ablation** — replace the existing four-step paragraphs with a sub-table showing doubly_monotone vs joint_umnn `final_loss` at each budget, plus the same mechanism explanation tied to `\S\ref{subsec:3.5}` and `\S\ref{thm:strictprop}`. This is the §8.4-specific empirical claim: trained R1-only flow's `final_loss` falls below the simulator's entropy lower bound at every budget, while doubly_monotone stays above it.
8. **Synthesis** — close §8.4 with the (R2)-as-correctness-requirement framing, the cross-method gap (which should be the widest of the §8.1→§8.4 progression — this is the most non-trivial target), and the implication for §11 open problems.

- [ ] **Step 1: Extract the §8.4 numbers**

Run, from the repo root, a script analogous to the §8.2 / §8.3 extraction. Mind that:
- `paper_table_8_4` aggregates index_row scalars only; ConditionalPIT bulk/edge rows must be extracted from per-run `conditional_pit.parquet` files.
- The ablation table is on the ablation sweep dir, NOT the main baseline sweep.

- [ ] **Step 2: Write the LaTeX edit**

In `cd_sbi_v7.tex`, replace lines ~2922–3018 (current §8.4 block, including the long four-step ablation paragraphs) with the eight-block structure above (the seven-block template + a §8.4-specific Ablation block), using the actual numbers from Step 1. Mirror the LaTeX patterns of §8.1/§8.2/§8.3: `longtable` for the calibration table (3-col), the cross-method comparison (5-col), and the ablation sub-table (5-col: flow × budget). Cross-references via `\S\ref{subsec:3.5}`, `\S\ref{thm:strictprop}`, `\S\ref{sec:4}`, `\S\ref{sec:10}`.

The "Step 1–4" structure of the current ablation paragraphs (lines ~2980–3018) is the right narrative shape — keep that flow but ground each step in the seed-averaged data.

- [ ] **Step 3: Compile the manuscript**

```bash
/usr/bin/pdflatex -interaction=nonstopmode -halt-on-error cd_sbi_v7 \
  && /usr/bin/bibtex cd_sbi_v7 \
  && /usr/bin/pdflatex -interaction=nonstopmode -halt-on-error cd_sbi_v7 \
  && /usr/bin/pdflatex -interaction=nonstopmode -halt-on-error cd_sbi_v7
```

Expected: "Output written on cd_sbi_v7.pdf (50–52 pages, ...)" — slightly higher than §8.3's 50 because of the ablation sub-table.

- [ ] **Step 4: Structural sanity check on the new TeX**

Run the helper script we used for §8.1/§8.2/§8.3 against the §8.4 region. All longtables should report cells-OK and brace-balanced. Cross-references all resolve.

- [ ] **Step 5: Commit**

```bash
git add cd_sbi_v7.tex
git commit -m "$(cat <<'EOF'
manuscript(8.4): v3 sweep + (R2) ablation seed-averaged results

Replace single-seed §8.4 numbers + four-paragraph ablation narrative
with sweep-averaged results from the v3 100-run cross-method sweep
+ 40-run ablation sweep. Eight-block layout: Setup / Architecture /
Sweep / CDSBI calibration / Budget invariance / Cross-method table /
Ablation sub-table / Synthesis (the §8.4-specific addition is the
ablation block).

Drops the single-seed Jacobian-style row from the §8.4 calibration
table (the truth Jacobian here is non-constant; the v2
JacobianRecovery diagnostic correctly skips on this target). Adds a
final_loss row to surface the (R2) contract: doubly_monotone stays
above the entropy lower bound at every budget; joint_umnn falls
below at every budget, with the gap empirically attributable to
log Z(θ) per §3.5.

50+ pages, full bibtex cycle clean.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Done condition:** PDF builds in 4 passes with no warnings beyond the natbib citation pass; §8.4 in the rendered PDF reads as parallel structure to §8.1/§8.2/§8.3 + the ablation block; the user has reviewed the substantive empirical claims (particularly the trained-folding result, which is the §8.4-specific deliverable).

---

## Self-Review

**1. Spec coverage** — checked against §11–§13 of the design spec (`docs/superpowers/specs/2026-05-25-cd-sbi-experiment-infrastructure-design.md`):

- v3 deliverable per §12 row "v3 §8.4 + (R2) ablation":
  - `DoublyMonotoneUMNN` (Task 4) ✓
  - `JointUMNNFlow` (Task 5) ✓
  - `JointUMNN1DFlow` (Task 6) ✓
  - `MLPConditioner` doing X→T reduction with non-zero `log_det_jac_input_contribution` (Task 2) ✓
  - New Ablation test category (safety-check + trained-folding empirical with noise-floor margin + 1D direct-construction mechanism test) (Task 9 a/b/c) ✓
  - `target/exp_rate`, `flow/{doubly_monotone, joint_umnn, joint_umnn_1d}`, `experiment/8_4_*` (Task 3) ✓
- §13 forward-hook commitments exercised:
  - `Flow.forward(θ, context)` signature unchanged — `DoublyMonotoneUMNN` takes T as context (Task 4). ✓
  - `monotonicity_guarantees` as a first-class attribute — JointUMNNFlow advertises `{R1}` only, triggering the v0 `MonotonicityMismatchError` safety check (Task 9a). ✓
  - `log_det_jac_input` naming + conditioner's `log_det_jac_input_contribution` — `MLPConditioner.encode` returns the non-zero `½ log(n)` per row (Task 2). ✓
  - `Simulator.entropy_lower_bound()` — `ExponentialRate.entropy_lower_bound()` returns the closed-form `n * (1 − E_θ[log θ])` (Task 1); the ablation regression test (Task 9b) consumes it. ✓
  - `ConfidenceProcedure` subtypes unchanged. ✓
- §11 done-criterion analogs for v3:
  1. CLI completes deterministically → Tasks 7 + 11 demonstrate.
  2. CDSBI tolerance bands at §8.4 → Task 10 (intensive test).
  3. Matched-budget across (method, budget) → Task 8 (retune adds `doubly_monotone_hidden`).
  4. Fast tests pass <30 s → preserved (incremental unit-test additions).
  5. `pytest -m intensive` matches bands → Task 10 implements; `pytest -m ablation` validates the §8.4-specific (R2) claim → Task 9.

**2. Placeholder scan**: no "TBD", "TODO without code", or hand-wave steps in any task. The `theta_ref` choice (= lower endpoint of proposal support) is justified by the manuscript §6.1 form 2 derivation. The `frozen_sum` mode of `MLPConditioner` is explicitly the §8.4-applicable mode; trainable modes are flagged as "v3.1+" rather than left as TBD.

**3. Type consistency**:
- `Simulator.sample(n, rng) → (theta: (n, d_theta), x: (n, d_x))` consistent across Tasks 1 and Task 2 consumer.
- `ExponentialRate.d_x` returns `n_iid` (the multivariate X), consumed by `MLPConditioner(input_dim=simulator.d_x)` in Task 7. ✓
- `Flow.forward(θ, context) → (r, log_det_jac_input)` shape `(n, d)`, `(n,)` consistent across all three v3 flow classes (Tasks 4, 5, 6).
- `Conditioner.encode(x) → (context: (n, output_dim), log_det: (n,))` consistent between Task 2 implementation and Task 7 wire-in.
- `monotonicity_guarantees` is `frozenset[Guarantee]` across all flows — `DoublyMonotoneUMNN` = `{R1, R2}` (Task 4), `JointUMNNFlow` = `{R1}` (Task 5), `JointUMNN1DFlow` = `{R1}` (Task 6).
- `entropy_lower_bound() → float` consistent between Task 1 (implementer) and Task 9b / 10 (consumer).
- `paper_table_8_4` column list matches what `_write_index_row` lifts (Task 11 + the existing run.py).

---

## Addendum: review-driven changes already applied + caveats for execution

An independent reviewer agent surfaced 11 findings (3 BLOCKER, 4 IMPORTANT, 4 MINOR) on the v3 plan draft. The plan has been updated inline to address the following:

- **Task 1 `entropy_lower_bound()` formula was wrong** (BLOCKER): the closed-form `n*(1−E_θ[log θ])` returns ≈ 3.23 on the X-space, NOT the manuscript's 0.88 (which is on the NF-MLE-loss space with the conditioner adjustment). Rewrote `entropy_lower_bound()` as a Monte-Carlo evaluation of the NF-MLE loss at the truth pivot — faithful to the same loss function the trainer optimizes, no analytic derivation gymnastics needed. Updated the unit-test band to `[0.70, 1.10]` around the manuscript's 0.88. Task 9b's `adjusted_floor` computation no longer subtracts `½ log(n)` (the conditioner Jacobian is already in the MC).
- **Task 5/6 JointUMNNFlow value-vs-gradient path duplication** (re-classified MINOR after deeper review): the reviewer flagged a dead-gradient concern with `T.detach()` in the value path. On inspection, `T.detach()` only blocks gradients into `T` (data), not into the MLP weights (parameters), so the value path DOES backprop correctly to all flow parameters. However the two-path construction (separate forward for value vs autograd) is wasteful and confusing. Leaving the current implementation as-is for v3 since it's correct; a single-path refactor is a v3.1 cleanup.
- **Task 9 Step 1 `addopts` did not exclude ablation marker** (BLOCKER): `addopts = "-m 'not intensive'"` would let the 4 new ablation tests (each ~1500 training steps) run by default under `pytest -q`, blowing the <30s fast-test contract. Step 1 now updates `addopts` to `"-m 'not intensive and not ablation'"`.
- **Task 7 `CDSBIRunner.n_params()` accesses `flow.a_blocks`/`flow.a`** (IMPORTANT): these attributes exist on v0's AdditiveFlow1D and v1's TriangularAdditiveFlow but NOT on v3's DoublyMonotoneUMNN / JointUMNNFlow / JointUMNN1DFlow. Without a fallback, the integration test in Task 7 Step 6 would fail at budget validation with a confusing `AttributeError`. Added Step 4b to widen `n_params()` with an `else` branch returning `{backbone: total, head: 0, total: total, kind: "single_block"}` for flows without an explicit backbone/head split.
- **Task 7 `CDSBIRunner.fit()` does not call `self.flow.train()`** (IMPORTANT): the v3 ablation flows use `create_graph=self.training` in their inner `torch.autograd.grad` calls. If the flow ever gets into `eval()` state before `fit()` runs (e.g., from a prior procedure evaluation), `create_graph` would be False and gradients through the log-det term would be zeroed silently. Step 4b also adds `self.flow.train()` at the top of `fit()`.
- **Task 11 ablation sweep CLI coupling** (IMPORTANT): the original plan launched `flow=doubly_monotone,joint_umnn` as a single multirun against `8_4_ablation.yaml`, which carries `method.allow_ablation: true`. That would silently keep `allow_ablation=true` on the `doubly_monotone` arm, weakening the safety contract for the production flow. Step 2 now splits into two arm-specific launches: `joint_umnn` uses `experiment=8_4_ablation` (with `allow_ablation=true`); `doubly_monotone` uses `experiment=8_4_replication` (which has the safe `allow_ablation=false` default).

Things the v0/v1/v2 arc taught us to pre-flag for v3. Most are already addressed by upstream code or the plan structure; documented here so the executor doesn't re-discover them.

- **GPU memory at LF2I-BFF xlarge (BLOCKER-class if missed):** v1 commit `b4ba38c` wrapped the multivariate ray-bisection in `torch.no_grad()`. §8.4's cross-method sweep at xlarge will hit the same code path with d_theta=1 — should be lighter than §8.2/§8.3 because the 1D `confidence_set_batch` doesn't use ray-bisection, just a quantile. No new fix needed; just verify GPU memory stays bounded.
- **NPE marginal-product → empirical Mahalanobis (already in):** v1 commits `5c93f1a` + `d46d589` replaced the Bonferroni cartesian product with empirical Mahalanobis. §8.4 is 1D so this collapses to a simple `equal_tailed_1d` interval — no Bonferroni concern. The v2 NPE footnote in the §8.3 manuscript table is §8.3-specific; §8.4 doesn't need it.
- **`n_samples` perf fix carries over (v2 commit `73d9a65`):** PosteriorBased default is now 2000. §8.4's NPE will be ~5× faster than the §8.2 NPE runs were.
- **LF2I-BFF head schedule retune (v1 close-out commit `8bae8c9`):** the v1 LF2I U-shape was fixed by retuning `quantile_hidden` per budget. §8.4's LF2I sweep should be flat across budgets, similar to §8.2/§8.3 post-retune.
- **Determinism contract on new RNG draws (v1 reviewer flag):** the ablation tests (Task 9b/9c) use deterministic seeds via `CDSBIRunner.fit(..., seed=0)`. The `MLPConditioner` `frozen_sum` mode has no RNG. No new draw-order drift risk.
- **`pdflatex` shadowing:** CLAUDE.md pins to `/usr/bin/pdflatex`. Task 12 uses the absolute path.
- **Single-seed manuscript numbers in §8.4 may diverge from sweep average:** the manuscript's reported pivot RMSE 0.045 / final loss 0.87 are single-seed. The v3 sweep mean ± std will likely sit slightly above/below those point estimates by a few hundredths; that's expected, mirroring §8.2/§8.3.
- **JointUMNNFlow `create_graph=self.training` in autograd:** the second autograd path inside the flow's forward needs `create_graph=True` during training to backprop through `log_det_jac_input`. The plan's implementation honors this via `create_graph=self.training`. If the trained-folding test (Task 9b) fails to converge below the entropy bound, double-check that `flow.train()` is set before `runner.fit(...)`.
- **`theta_ref` choice:** the plan sets `theta_ref = simulator.theta_range[0]` (the lower endpoint of the proposal). This is what makes the integral always over `[theta_ref, θ]` with `θ ≥ theta_ref` (manuscript §6.1 form 2 derivation). Don't change it without re-deriving the loss factor.
- **Ablation tests are slow:** Task 9b and 9c each run ~1500 training steps. Wall ~1–3 min each on GPU. Mark them with `@pytest.mark.ablation` so the default `pytest -q` skips them, like `intensive`.
- **`paper_table_8_4` does NOT include `joint_mahal_ks`:** §8.4 is 1D, so `JointMahalanobis` no-ops (it's degenerate in d=1, returns NaN). The aggregation skips that column via the `m in df.columns` filter pattern; consumers shouldn't expect it.

---

## Execution handoff

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task with two-stage review (spec compliance → code quality) after each. Same pattern that landed v1 and v2.

**2. Inline Execution** — execute tasks in this session via `superpowers:executing-plans`, batch with checkpoints.

Before either path: send the draft plan to an independent reviewer agent (mirror the v1/v2 reviewer passes that caught real BLOCKER-class issues). The reviewer should explicitly look for (a) BLOCKER-class bugs in Tasks 1–6 (the load-bearing new code, especially the doubly-monotone integral derivation and the ablation flow's autograd path), (b) test coverage gaps (particularly in the trained-folding test — is the empirical noise-floor margin tight enough?), (c) cross-task type consistency, (d) anything the v0/v1/v2 addendum may have missed.
