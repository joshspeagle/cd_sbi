# CD-SBI v0 experiment infrastructure — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v0 milestone: a single CLI invocation
(`python -m cdsbi.experiments.run -m experiment=8_1_baseline_sweep`)
runs 5 methods × 4 budgets × 5 seeds = 100 deterministic runs on
`LocationNormal1D`, produces structured parquet outputs, and yields a
publishable cross-method comparison DataFrame via
`analysis.paper_table_8_1`.

**Architecture:** A layered Python package (`cdsbi`) with narrow
per-layer interfaces (`Simulator`, `Flow`, `Conditioner`, `Loss`,
`Method.Runner`, `ConfidenceProcedure`, `Diagnostic`). Cross-method
comparison happens through `ConfidenceProcedure` — pivots, posteriors,
likelihoods, ratios, and test-stat+critical-value pairs each produce
confidence sets via 1D root-finding. NPE/NLE/NRE wrap the `sbi`
package with custom density estimators / classifiers so backbone
parameter count is under our control; LF2I is built directly on the
same MAF backbone as NLE. Hydra drives config composition; each run
writes a self-contained directory under `outputs/` with a `STATUS`
file, parquet metrics/diagnostics, and a one-row `index_row.parquet`
that `analysis.load_runs()` globs and concatenates.

**Tech Stack:** Python 3.10+, PyTorch, `sbi`, `nflows`, `hydra-core`,
`pandas`, `pyarrow`, `pytest`. GPU when available, CPU fallback.

**Source spec:** `docs/superpowers/specs/2026-05-25-cd-sbi-experiment-infrastructure-design.md`

---

## File structure (created by this plan)

```
cd_sbi/                                  # repo root (exists)
├── .gitignore                           # MODIFY: add outputs/, __pycache__, *.egg-info, .pytest_cache
├── pyproject.toml                       # CREATE
├── src/cdsbi/
│   ├── __init__.py                      # CREATE (empty)
│   ├── device.py                        # CREATE
│   ├── reproducibility/
│   │   ├── __init__.py                  # CREATE
│   │   ├── seeding.py                   # CREATE
│   │   ├── env.py                       # CREATE
│   │   └── run_dir.py                   # CREATE
│   ├── simulators/
│   │   ├── __init__.py                  # CREATE
│   │   ├── base.py                      # CREATE — Simulator protocol
│   │   └── location_normal_1d.py        # CREATE
│   ├── flows/
│   │   ├── __init__.py                  # CREATE
│   │   ├── base.py                      # CREATE — Flow protocol + Guarantee enum
│   │   ├── umnn.py                      # CREATE — UMNNBlock
│   │   ├── additive.py                  # CREATE — AdditiveFlow1D
│   │   └── maf_adapter.py               # CREATE — MAFAdapter (wraps nflows)
│   ├── conditioners/
│   │   ├── __init__.py                  # CREATE
│   │   ├── base.py                      # CREATE — Conditioner protocol
│   │   └── identity.py                  # CREATE — Identity conditioner
│   ├── losses/
│   │   ├── __init__.py                  # CREATE
│   │   ├── base.py                      # CREATE — Loss protocol + MonotonicityMismatchError
│   │   └── nfmle.py                     # CREATE — NFMLELoss
│   ├── confidence_set/
│   │   ├── __init__.py                  # CREATE
│   │   ├── datatypes.py                 # CREATE — ConfidenceSet dataclass
│   │   ├── procedures.py                # CREATE — ConfidenceProcedure subtypes
│   │   ├── root_find.py                 # CREATE — 1D bisection
│   │   └── hpd.py                       # CREATE — 1D HPD extractor
│   ├── methods/
│   │   ├── __init__.py                  # CREATE
│   │   ├── base.py                      # CREATE — Runner protocol + TrainedModel + BudgetUnreachableError
│   │   ├── budget.py                    # CREATE — build_from_budget enumeration
│   │   ├── cd_sbi.py                    # CREATE — CDSBIRunner
│   │   ├── npe.py                       # CREATE — NPERunner (wraps sbi)
│   │   ├── nle.py                       # CREATE — NLERunner (wraps sbi)
│   │   ├── nre.py                       # CREATE — NRERunner (wraps sbi)
│   │   └── lf2i.py                      # CREATE — LF2IRunner (two-stage)
│   ├── diagnostics/
│   │   ├── __init__.py                  # CREATE
│   │   ├── base.py                      # CREATE — Diagnostic protocol + DiagnosticResult
│   │   ├── ks_floor.py                  # CREATE — ks_noise_floor helper
│   │   ├── pivot_rmse.py                # CREATE
│   │   ├── marginal_pit.py              # CREATE
│   │   ├── conditional_pit.py           # CREATE
│   │   └── coverage.py                  # CREATE
│   ├── experiments/
│   │   ├── __init__.py                  # CREATE
│   │   └── run.py                       # CREATE — Hydra CLI entrypoint
│   └── analysis/
│       ├── __init__.py                  # CREATE
│       ├── loaders.py                   # CREATE — load_run, load_runs
│       └── paper_tables.py              # CREATE — paper_table_8_1
├── configs/                             # CREATE
│   ├── config.yaml
│   ├── target/loc_normal_1d.yaml
│   ├── flow/{additive_umnn,maf}.yaml
│   ├── conditioner/identity.yaml
│   ├── method/{cd_sbi,npe,nle,nre,lf2i}.yaml
│   ├── training/adam_3e-3_4k_steps.yaml
│   ├── budget/{small,medium,large,xlarge}.yaml
│   └── experiment/{8_1_replication,8_1_baseline_sweep}.yaml
└── tests/
    ├── __init__.py                      # CREATE
    ├── conftest.py                      # CREATE
    ├── unit/                            # CREATE
    │   └── test_*.py                    # one per module
    ├── integration/                     # CREATE
    │   └── test_*.py
    ├── diagnostics/                     # CREATE
    │   └── test_*.py
    └── intensive/                       # CREATE
        ├── README.md
        └── test_replicate_8_1.py
```

---

## Task 1: Package skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/cdsbi/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cdsbi"
version = "0.0.1"
description = "Confidence-distribution SBI experiment infrastructure"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "sbi>=0.22",
    "nflows>=0.14",
    "hydra-core>=1.3",
    "numpy>=1.24",
    "scipy>=1.11",
    "pandas>=2.0",
    "pyarrow>=14.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-xdist>=3.3"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "intensive: full-budget replication tests; opt-in via `pytest -m intensive`",
]
addopts = "-m 'not intensive'"
```

- [ ] **Step 2: Create empty package init**

```python
# src/cdsbi/__init__.py
"""CD-SBI experiment infrastructure."""
```

- [ ] **Step 3: Update `.gitignore`**

Append to existing `.gitignore`:

```
# CD-SBI experiment outputs
outputs/

# Python build artifacts
*.egg-info/
__pycache__/
.pytest_cache/
.coverage
```

- [ ] **Step 4: Create tests skeleton**

```python
# tests/__init__.py
```

```python
# tests/conftest.py
"""Shared pytest fixtures."""
import os
import pytest
import torch

# Force deterministic CUDA when present; harmless on CPU
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture
def seed():
    """Default seed for tests that need one."""
    return 12345


@pytest.fixture
def tmp_run_dir(tmp_path):
    """Throwaway run directory."""
    d = tmp_path / "run"
    d.mkdir()
    return d
```

- [ ] **Step 5: Install in editable mode**

Run: `pip install -e ".[dev]"`
Expected: installs without error; `python -c "import cdsbi"` succeeds.

- [ ] **Step 6: Sanity-check the skeleton runs pytest**

Run: `pytest tests/`
Expected: `collected 0 items` (no tests yet) — exits 0 (no error).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/cdsbi/__init__.py tests/__init__.py tests/conftest.py .gitignore
git commit -m "feat: package skeleton + pytest config"
```

---

## Task 2: Reproducibility — seeding

**Files:**
- Create: `src/cdsbi/reproducibility/__init__.py`
- Create: `src/cdsbi/reproducibility/seeding.py`
- Test: `tests/unit/test_seeding.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_seeding.py
import numpy as np
import torch
from cdsbi.reproducibility.seeding import seed_everything


def test_seed_everything_returns_named_streams(seed):
    rngs = seed_everything(seed)
    assert hasattr(rngs, "train")
    assert hasattr(rngs, "eval")
    assert hasattr(rngs, "init")
    # Streams are distinct
    assert rngs.train.bit_generator.state != rngs.eval.bit_generator.state


def test_seed_everything_is_deterministic(seed):
    seed_everything(seed)
    a = torch.randn(5)
    b = np.random.randn(5)
    seed_everything(seed)
    c = torch.randn(5)
    d = np.random.randn(5)
    assert torch.equal(a, c)
    np.testing.assert_array_equal(b, d)


def test_eval_stream_differs_from_train_stream(seed):
    rngs = seed_everything(seed)
    train_draw = rngs.train.standard_normal(3)
    rngs2 = seed_everything(seed)
    eval_draw = rngs2.eval.standard_normal(3)
    # Different streams ⇒ different draws even from the same master seed
    assert not np.allclose(train_draw, eval_draw)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_seeding.py -v`
Expected: ImportError (module doesn't exist yet).

- [ ] **Step 3: Implement seeding**

```python
# src/cdsbi/reproducibility/__init__.py
```

```python
# src/cdsbi/reproducibility/seeding.py
"""Deterministic seeding for torch / numpy / python; named per-purpose RNG streams."""
import hashlib
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SeededRNGs:
    train: np.random.Generator
    eval: np.random.Generator
    init: np.random.Generator


def _derive(master: int, label: str) -> int:
    """Deterministic 64-bit substream seed from (master, label)."""
    h = hashlib.sha256(f"{master}:{label}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**63 - 1)


def seed_everything(seed: int) -> SeededRNGs:
    """Seed all global RNGs and return named numpy.Generator streams.

    `sbi`-based runners pass explicit per-call seeds derived from the
    same master; they do not rely on global `sbi` state.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return SeededRNGs(
        train=np.random.default_rng(_derive(seed, "train")),
        eval=np.random.default_rng(_derive(seed, "eval")),
        init=np.random.default_rng(_derive(seed, "init")),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_seeding.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/reproducibility/ tests/unit/test_seeding.py
git commit -m "feat(reproducibility): seed_everything with named RNG streams"
```

---

## Task 3: Reproducibility — env capture + run dir + STATUS

**Files:**
- Create: `src/cdsbi/reproducibility/env.py`
- Create: `src/cdsbi/reproducibility/run_dir.py`
- Test: `tests/unit/test_env.py`, `tests/unit/test_run_dir.py`

- [ ] **Step 1: Write the failing env test**

```python
# tests/unit/test_env.py
from cdsbi.reproducibility.env import capture_env


def test_capture_env_has_required_keys():
    env = capture_env()
    assert "git_sha" in env
    assert "dirty_tree" in env
    assert "python_version" in env
    assert "torch_version" in env
    assert "device" in env
    assert "platform" in env
    assert isinstance(env["dirty_tree"], bool)
```

- [ ] **Step 2: Write the failing run-dir test**

```python
# tests/unit/test_run_dir.py
import json
import pytest
from cdsbi.reproducibility.run_dir import RunDir, RunStatus


def test_set_status_writes_atomic(tmp_run_dir):
    rd = RunDir(tmp_run_dir)
    rd.set_status(RunStatus.RUNNING)
    assert (tmp_run_dir / "STATUS").read_text().strip() == "RUNNING"
    rd.set_status(RunStatus.OK)
    assert (tmp_run_dir / "STATUS").read_text().strip() == "OK"


def test_write_atomic_no_partial_on_crash(tmp_run_dir):
    rd = RunDir(tmp_run_dir)
    rd.write_atomic(tmp_run_dir / "config.yaml", "key: value\n")
    assert (tmp_run_dir / "config.yaml").read_text() == "key: value\n"
    # No leftover .tmp file
    assert not (tmp_run_dir / "config.yaml.tmp").exists()


def test_write_atomic_overwrites_existing(tmp_run_dir):
    rd = RunDir(tmp_run_dir)
    p = tmp_run_dir / "x.json"
    rd.write_atomic(p, json.dumps({"a": 1}))
    rd.write_atomic(p, json.dumps({"a": 2}))
    assert json.loads(p.read_text()) == {"a": 2}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_env.py tests/unit/test_run_dir.py -v`
Expected: ImportError (modules don't exist).

- [ ] **Step 4: Implement env capture**

```python
# src/cdsbi/reproducibility/env.py
"""Capture environment metadata for a run."""
import platform as _platform
import subprocess
import sys

import torch


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def capture_env() -> dict:
    return {
        "git_sha": _git_sha(),
        "dirty_tree": _git_dirty(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "sbi_version": _package_version("sbi"),
        "nflows_version": _package_version("nflows"),
        "numpy_version": _package_version("numpy"),
        "scipy_version": _package_version("scipy"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": _platform.platform(),
        "num_threads": torch.get_num_threads(),
    }
```

- [ ] **Step 5: Implement run dir**

```python
# src/cdsbi/reproducibility/run_dir.py
"""Atomic writes and STATUS file management for run directories."""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Union


class RunStatus(str, Enum):
    RUNNING = "RUNNING"
    OK = "OK"
    FAILED = "FAILED"


class RunDir:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def set_status(self, status: RunStatus) -> None:
        self.write_atomic(self.path / "STATUS", status.value + "\n")

    def write_atomic(self, target: Path, content: Union[str, bytes]) -> None:
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(tmp, mode) as f:
            f.write(content)
        os.replace(tmp, target)  # atomic on POSIX and modern Windows
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_env.py tests/unit/test_run_dir.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add src/cdsbi/reproducibility/env.py src/cdsbi/reproducibility/run_dir.py tests/unit/test_env.py tests/unit/test_run_dir.py
git commit -m "feat(reproducibility): env capture and atomic-write RunDir with STATUS"
```

---

## Task 4: Device resolution

**Files:**
- Create: `src/cdsbi/device.py`
- Test: `tests/unit/test_device.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_device.py
import torch
from cdsbi.device import get_device


def test_get_device_auto():
    d = get_device("auto")
    assert d.type in {"cpu", "cuda"}
    if torch.cuda.is_available():
        assert d.type == "cuda"
    else:
        assert d.type == "cpu"


def test_get_device_cpu_force():
    assert get_device("cpu").type == "cpu"


def test_get_device_invalid():
    import pytest
    with pytest.raises(ValueError):
        get_device("tpu")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_device.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement device**

```python
# src/cdsbi/device.py
"""Device resolution: GPU by default with CPU fallback."""
import torch


def get_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec in {"cpu", "cuda"}:
        if spec == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA not available")
        return torch.device(spec)
    raise ValueError(f"Unknown device spec: {spec!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_device.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/device.py tests/unit/test_device.py
git commit -m "feat(device): GPU-with-CPU-fallback resolution"
```

---

## Task 5: Simulator base + LocationNormal1D

**Files:**
- Create: `src/cdsbi/simulators/__init__.py`, `base.py`, `location_normal_1d.py`
- Test: `tests/unit/test_location_normal_1d.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_location_normal_1d.py
import math
import numpy as np
import torch
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def test_sample_shapes():
    sim = LocationNormal1D()
    rng = np.random.default_rng(0)
    theta, x = sim.sample(100, rng)
    assert theta.shape == (100, 1)
    assert x.shape == (100, 1)


def test_sample_theta_in_range():
    sim = LocationNormal1D(theta_range=(-7.0, 7.0))
    rng = np.random.default_rng(0)
    theta, _ = sim.sample(1000, rng)
    assert (theta >= -7.0).all() and (theta <= 7.0).all()


def test_sample_x_conditional_mean():
    sim = LocationNormal1D()
    rng = np.random.default_rng(0)
    theta, x = sim.sample(50_000, rng)
    # E[X | θ] = θ ⇒ (X − θ) has mean 0
    diff = (x - theta).numpy().flatten()
    assert abs(diff.mean()) < 0.02
    assert abs(diff.std() - 1.0) < 0.02


def test_r_star_is_theta_minus_x():
    sim = LocationNormal1D()
    theta = torch.tensor([[1.0], [2.0]])
    x = torch.tensor([[0.5], [3.0]])
    r = sim.r_star(theta, x)
    assert torch.allclose(r, torch.tensor([[0.5], [-1.0]]))


def test_entropy_lower_bound():
    sim = LocationNormal1D()
    assert abs(sim.entropy_lower_bound() - 0.5 * math.log(2 * math.pi * math.e)) < 1e-12


def test_log_prob_matches_normal():
    sim = LocationNormal1D()
    theta = torch.tensor([[0.0]])
    x = torch.tensor([[1.0]])
    expected = -0.5 * math.log(2 * math.pi) - 0.5
    assert abs(sim.log_prob(x, theta).item() - expected) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_location_normal_1d.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement simulator base**

```python
# src/cdsbi/simulators/__init__.py
```

```python
# src/cdsbi/simulators/base.py
"""Simulator protocol."""
from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class Simulator(Protocol):
    """Generates (θ, X | θ) pairs. Optionally exposes analytical r*, log p, entropy floor."""

    d_theta: int
    d_x: int

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (theta, x) with shapes (n, d_theta) and (n, d_x)."""
        ...

    def r_star(self, theta: torch.Tensor, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Analytical canonical pivot if known; None otherwise."""
        ...

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> Optional[torch.Tensor]:
        """Analytical conditional log p(x | theta) if known; None otherwise."""
        ...

    def entropy_lower_bound(self) -> Optional[float]:
        """E_ρ[H(X | θ)] when known in closed form; None otherwise."""
        ...
```

- [ ] **Step 4: Implement LocationNormal1D**

```python
# src/cdsbi/simulators/location_normal_1d.py
"""LocationNormal1D simulator: X ~ N(θ, 1) with θ ~ U[a, b]."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class LocationNormal1D:
    theta_range: Tuple[float, float] = (-7.0, 7.0)
    d_theta: int = 1
    d_x: int = 1

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = self.theta_range
        theta_np = rng.uniform(a, b, size=(n, 1))
        eps_np = rng.standard_normal(size=(n, 1))
        x_np = theta_np + eps_np
        return (
            torch.from_numpy(theta_np).float(),
            torch.from_numpy(x_np).float(),
        )

    def r_star(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return theta - x

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return -0.5 * math.log(2 * math.pi) - 0.5 * (x - theta).pow(2).sum(dim=-1)

    def entropy_lower_bound(self) -> float:
        return 0.5 * math.log(2 * math.pi * math.e)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_location_normal_1d.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/simulators/ tests/unit/test_location_normal_1d.py
git commit -m "feat(simulators): Simulator protocol + LocationNormal1D"
```

---

## Task 6: Flow base + UMNNBlock

**Files:**
- Create: `src/cdsbi/flows/__init__.py`, `base.py`, `umnn.py`
- Test: `tests/unit/test_umnn.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_umnn.py
import torch
from cdsbi.flows.umnn import UMNNBlock


def test_umnn_strictly_monotone_on_grid(seed):
    torch.manual_seed(seed)
    block = UMNNBlock(context_dim=0, hidden=32)
    z = torch.linspace(-3, 3, 200).unsqueeze(-1)
    g = block(z, context=None)
    # strictly increasing
    diffs = g[1:] - g[:-1]
    assert (diffs > 0).all()


def test_umnn_jacobian_factor_positive(seed):
    torch.manual_seed(seed)
    block = UMNNBlock(context_dim=0, hidden=32)
    z = torch.linspace(-3, 3, 200).unsqueeze(-1)
    j = block.jacobian_factor(z, context=None)
    assert (j > 0).all()


def test_umnn_with_context_shape(seed):
    torch.manual_seed(seed)
    block = UMNNBlock(context_dim=3, hidden=16)
    z = torch.randn(10, 1)
    ctx = torch.randn(10, 3)
    g = block(z, context=ctx)
    assert g.shape == (10, 1)


def test_umnn_n_params_matches_torch(seed):
    torch.manual_seed(seed)
    block = UMNNBlock(context_dim=0, hidden=32)
    counted = block.n_params()
    actual = sum(p.numel() for p in block.parameters())
    assert counted == actual
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_umnn.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement Flow base**

```python
# src/cdsbi/flows/__init__.py
```

```python
# src/cdsbi/flows/base.py
"""Flow protocol + monotonicity-guarantee enum."""
from __future__ import annotations

from enum import Enum
from typing import Optional, Protocol, Tuple, runtime_checkable

import torch


class Guarantee(str, Enum):
    R1 = "R1"  # monotone in θ
    R2 = "R2"  # monotone in X (more precisely: invertible in X)


@runtime_checkable
class Flow(Protocol):
    monotonicity_guarantees: frozenset

    def forward(
        self, theta: torch.Tensor, context: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (r, log_det_jac_input). r has shape (n, d); log_det_jac_input has shape (n,)."""
        ...

    def n_params(self) -> int: ...
```

- [ ] **Step 4: Implement UMNNBlock**

```python
# src/cdsbi/flows/umnn.py
"""Unconstrained Monotonic Neural Network with softplus + 12-pt Gauss–Legendre."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 12-point Gauss–Legendre nodes/weights on [-1, 1]
_NODES_NP, _WEIGHTS_NP = np.polynomial.legendre.leggauss(12)


class UMNNBlock(nn.Module):
    """g(z; c) = bias(c) + ∫_0^z softplus(MLP(t, c)) dt, monotone in z by construction.

    Implementation uses 12-point Gauss–Legendre quadrature mapped to [0, z]. The
    Jacobian factor ∂g/∂z is the integrand at z, returned in closed form (no
    recursive autograd).
    """

    def __init__(self, context_dim: int, hidden: int = 32):
        super().__init__()
        self.context_dim = context_dim
        in_dim = 1 + context_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init final layer so g starts near identity-ish at init
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        # Bias network on context (or scalar bias if context_dim == 0)
        if context_dim == 0:
            self.bias_param = nn.Parameter(torch.zeros(1))
            self.bias_net = None
        else:
            self.bias_param = None
            self.bias_net = nn.Linear(context_dim, 1)

        self.register_buffer("_nodes", torch.tensor(_NODES_NP, dtype=torch.float32))
        self.register_buffer("_weights", torch.tensor(_WEIGHTS_NP, dtype=torch.float32))

    def _bias(self, context: Optional[torch.Tensor]) -> torch.Tensor:
        if self.bias_net is not None:
            return self.bias_net(context)
        return self.bias_param

    def _integrand(self, t: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        """softplus(MLP([t, c])) — strictly positive."""
        if context is None or self.context_dim == 0:
            inputs = t
        else:
            # t: (n, K, 1); context: (n, C) → broadcast to (n, K, C)
            ctx = context.unsqueeze(1).expand(-1, t.size(1), -1)
            inputs = torch.cat([t, ctx], dim=-1)
        return F.softplus(self.mlp(inputs))

    def forward(self, z: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """g(z; c). z has shape (n, 1); returns (n, 1)."""
        n = z.size(0)
        # Map nodes from [-1, 1] to [0, z]: t = z/2 (u + 1)
        u = self._nodes.view(1, -1, 1).expand(n, -1, 1)  # (n, K, 1)
        z_exp = z.view(n, 1, 1).expand(-1, u.size(1), -1)  # (n, K, 1)
        t = 0.5 * z_exp * (u + 1.0)
        integrand = self._integrand(t, context)  # (n, K, 1)
        weights = self._weights.view(1, -1, 1)  # (1, K, 1)
        # ∫_0^z f(t) dt = (z/2) ∑ w_i f(t_i)
        integral = 0.5 * z * (weights * integrand).sum(dim=1)
        return self._bias(context) + integral

    def jacobian_factor(
        self, z: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """∂g/∂z evaluated at z — softplus(MLP([z, c]))."""
        if context is None or self.context_dim == 0:
            inputs = z
        else:
            inputs = torch.cat([z, context], dim=-1)
        return F.softplus(self.mlp(inputs))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_umnn.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/flows/__init__.py src/cdsbi/flows/base.py src/cdsbi/flows/umnn.py tests/unit/test_umnn.py
git commit -m "feat(flows): Flow protocol + UMNNBlock with Gauss-Legendre quadrature"
```

---

## Task 7: AdditiveFlow1D

**Files:**
- Create: `src/cdsbi/flows/additive.py`
- Test: `tests/unit/test_additive_flow.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_additive_flow.py
import torch
from cdsbi.flows.additive import AdditiveFlow1D
from cdsbi.flows.base import Guarantee


def test_additive_flow_guarantees_r1_r2():
    flow = AdditiveFlow1D(hidden=16)
    assert flow.monotonicity_guarantees == frozenset({Guarantee.R1, Guarantee.R2})


def test_additive_flow_forward_shapes(seed):
    torch.manual_seed(seed)
    flow = AdditiveFlow1D(hidden=16)
    theta = torch.linspace(-3, 3, 10).unsqueeze(-1)
    context = torch.randn(10, 1)  # this conditioner-encoded X
    r, log_det = flow.forward(theta, context)
    assert r.shape == (10, 1)
    assert log_det.shape == (10,)


def test_additive_flow_monotone_in_theta(seed):
    torch.manual_seed(seed)
    flow = AdditiveFlow1D(hidden=16)
    theta = torch.linspace(-3, 3, 100).unsqueeze(-1)
    context = torch.zeros(100, 1)
    r, _ = flow.forward(theta, context)
    diffs = r[1:] - r[:-1]
    assert (diffs > 0).all()


def test_additive_flow_monotone_decreasing_in_x(seed):
    torch.manual_seed(seed)
    flow = AdditiveFlow1D(hidden=16)
    theta = torch.zeros(100, 1)
    context = torch.linspace(-3, 3, 100).unsqueeze(-1)
    r, _ = flow.forward(theta, context)
    diffs = r[1:] - r[:-1]
    assert (diffs < 0).all()  # subtraction of monotone-in-X term
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_additive_flow.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement AdditiveFlow1D**

```python
# src/cdsbi/flows/additive.py
"""AdditiveFlow1D: r(θ, X) = α_a · UMNN(θ) − α_b · UMNN(X). Monotone in θ and X by construction."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from cdsbi.flows.base import Flow, Guarantee
from cdsbi.flows.umnn import UMNNBlock


class AdditiveFlow1D(nn.Module, Flow):
    monotonicity_guarantees = frozenset({Guarantee.R1, Guarantee.R2})

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.a = UMNNBlock(context_dim=0, hidden=hidden)
        self.b = UMNNBlock(context_dim=0, hidden=hidden)
        # init at 0.5 / ln(2) ≈ 0.72 ⇒ deliberately-wrong slope 0.5 so optimization moves
        init_alpha = 0.5 / math.log(2.0)
        self.alpha_a = nn.Parameter(torch.tensor(init_alpha))
        self.alpha_b = nn.Parameter(torch.tensor(init_alpha))

    def forward(
        self, theta: torch.Tensor, context: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # context is the conditioner-encoded X (Identity ⇒ context == X)
        a = self.a(theta, context=None)
        b = self.b(context, context=None)
        r = self.alpha_a * a - self.alpha_b * b
        # ∂r/∂input where "input" = X = context: derivative is −α_b · b'(X)
        # log |∂r/∂X| = log(α_b) + log(b'(X))
        b_prime = self.b.jacobian_factor(context, context=None)
        log_det = torch.log(self.alpha_b * b_prime).sum(dim=-1)
        return r, log_det

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_additive_flow.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/flows/additive.py tests/unit/test_additive_flow.py
git commit -m "feat(flows): AdditiveFlow1D advertising R1+R2 guarantees"
```

---

## Task 8: Conditioner base + Identity

**Files:**
- Create: `src/cdsbi/conditioners/__init__.py`, `base.py`, `identity.py`
- Test: `tests/unit/test_identity_conditioner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_identity_conditioner.py
import torch
from cdsbi.conditioners.identity import Identity


def test_identity_returns_x_unchanged():
    cond = Identity()
    x = torch.randn(5, 1)
    ctx, log_det_contrib = cond.encode(x)
    assert torch.equal(ctx, x)
    assert torch.equal(log_det_contrib, torch.zeros(5))


def test_identity_n_params():
    cond = Identity()
    assert cond.n_params() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_identity_conditioner.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement Conditioner base + Identity**

```python
# src/cdsbi/conditioners/__init__.py
```

```python
# src/cdsbi/conditioners/base.py
"""Conditioner protocol: encode X into a context vector + a log-det contribution."""
from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import torch


@runtime_checkable
class Conditioner(Protocol):
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (context_vec, log_det_jac_input_contribution).

        log_det_jac_input_contribution has shape (n,); for X → X (identity) it is zero.
        For an X → T sufficient-statistic reduction (v3+) it is log |∂T/∂X|.
        """
        ...

    def n_params(self) -> int: ...
```

```python
# src/cdsbi/conditioners/identity.py
"""Identity conditioner: returns X as the context, zero log-det contribution."""
from __future__ import annotations

from typing import Tuple

import torch


class Identity:
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, torch.zeros(x.shape[0], device=x.device)

    def n_params(self) -> int:
        return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_identity_conditioner.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/conditioners/ tests/unit/test_identity_conditioner.py
git commit -m "feat(conditioners): Conditioner protocol + Identity"
```

---

## Task 9: Loss base + NFMLELoss + monotonicity check

**Files:**
- Create: `src/cdsbi/losses/__init__.py`, `base.py`, `nfmle.py`
- Test: `tests/unit/test_nfmle_loss.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_nfmle_loss.py
import math
import pytest
import torch
from cdsbi.flows.additive import AdditiveFlow1D
from cdsbi.flows.base import Guarantee
from cdsbi.losses.base import MonotonicityMismatchError
from cdsbi.losses.nfmle import NFMLELoss


def test_required_guarantees():
    loss = NFMLELoss()
    assert loss.required_guarantees == frozenset({Guarantee.R1, Guarantee.R2})


def test_population_lower_bound_when_simulator_has_entropy():
    from cdsbi.simulators.location_normal_1d import LocationNormal1D
    loss = NFMLELoss()
    sim = LocationNormal1D()
    expected = 0.5 * math.log(2 * math.pi * math.e)
    assert abs(loss.population_lower_bound(sim) - expected) < 1e-12


def test_population_lower_bound_none_when_simulator_lacks_entropy():
    class DummySim:
        def entropy_lower_bound(self): return None
    loss = NFMLELoss()
    assert loss.population_lower_bound(DummySim()) is None


def test_check_guarantees_passes_on_r1_r2_flow():
    loss = NFMLELoss()
    flow = AdditiveFlow1D(hidden=8)
    loss.check_guarantees(flow)  # should not raise


def test_check_guarantees_raises_on_missing_r2():
    class FlowMissingR2:
        monotonicity_guarantees = frozenset({Guarantee.R1})
    loss = NFMLELoss()
    with pytest.raises(MonotonicityMismatchError):
        loss.check_guarantees(FlowMissingR2())


def test_nfmle_value_on_oracle_pivot(seed):
    """NF-MLE loss for r* = θ − X on N(θ, 1) data ≈ ½ log(2π) + ½ = entropy of N(0,1)."""
    torch.manual_seed(seed)
    n = 5000
    theta = torch.empty(n, 1).uniform_(-7, 7)
    x = theta + torch.randn(n, 1)
    r = theta - x
    log_det = torch.zeros(n)  # |∂r/∂X| = 1 ⇒ log = 0
    loss_val = NFMLELoss()(r=r, log_det_jac_input=log_det)
    # E[½ r²] − E[log |∂r/∂X|] = ½ · 1 + ½ log(2π) (the ½ log 2π comes from the standard-normal
    # density evaluated at r, not from log_det)
    expected = 0.5 + 0.5 * math.log(2 * math.pi)
    assert abs(loss_val.item() - expected) < 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_nfmle_loss.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement Loss base + error**

```python
# src/cdsbi/losses/__init__.py
```

```python
# src/cdsbi/losses/base.py
"""Loss protocol + MonotonicityMismatchError."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


class MonotonicityMismatchError(RuntimeError):
    """Raised when a Loss requires guarantees the Flow doesn't advertise."""


@runtime_checkable
class Loss(Protocol):
    required_guarantees: frozenset

    def check_guarantees(self, flow) -> None:
        """Raise MonotonicityMismatchError if flow.monotonicity_guarantees < self.required_guarantees."""
        ...

    def population_lower_bound(self, simulator) -> Optional[float]: ...
```

- [ ] **Step 4: Implement NFMLELoss**

```python
# src/cdsbi/losses/nfmle.py
"""NF-MLE loss: ½ ‖r‖² − log |∂r/∂X|, with monotonicity-guarantee check."""
from __future__ import annotations

import math
from typing import Optional

import torch

from cdsbi.flows.base import Guarantee
from cdsbi.losses.base import Loss, MonotonicityMismatchError


class NFMLELoss(Loss):
    required_guarantees = frozenset({Guarantee.R1, Guarantee.R2})

    def __call__(self, r: torch.Tensor, log_det_jac_input: torch.Tensor) -> torch.Tensor:
        """Per-sample loss: ½ ‖r‖² + ½ d log(2π) − log |∂r/∂X|, then averaged.

        The ½ d log(2π) term is the standard-normal normalization for r;
        we include it so loss values are directly comparable to the
        conditional-entropy lower bound (Theorem 3.2).
        """
        d = r.shape[-1]
        per_sample = 0.5 * r.pow(2).sum(dim=-1) + 0.5 * d * math.log(2 * math.pi) - log_det_jac_input
        return per_sample.mean()

    def check_guarantees(self, flow) -> None:
        guarantees = getattr(flow, "monotonicity_guarantees", frozenset())
        if not self.required_guarantees.issubset(guarantees):
            missing = self.required_guarantees - guarantees
            raise MonotonicityMismatchError(
                f"NFMLELoss requires {sorted(g.value for g in self.required_guarantees)}; "
                f"flow {type(flow).__name__} provides {sorted(g.value for g in guarantees)}; "
                f"missing {sorted(g.value for g in missing)}."
            )

    def population_lower_bound(self, simulator) -> Optional[float]:
        return simulator.entropy_lower_bound()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_nfmle_loss.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/losses/ tests/unit/test_nfmle_loss.py
git commit -m "feat(losses): NFMLELoss + MonotonicityMismatchError safety check"
```

---

## Task 10: MAFAdapter (wraps nflows.MAF)

**Files:**
- Create: `src/cdsbi/flows/maf_adapter.py`
- Test: `tests/unit/test_maf_adapter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_maf_adapter.py
import torch
from cdsbi.flows.maf_adapter import MAFAdapter


def test_maf_adapter_no_guarantees():
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    assert flow.monotonicity_guarantees == frozenset()


def test_maf_adapter_log_prob_shape(seed):
    torch.manual_seed(seed)
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    x = torch.randn(10, 1)
    context = torch.randn(10, 1)
    log_p = flow.log_prob(x, context)
    assert log_p.shape == (10,)


def test_maf_adapter_n_params_matches_torch(seed):
    torch.manual_seed(seed)
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    assert flow.n_params() == sum(p.numel() for p in flow.parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_maf_adapter.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement MAFAdapter**

```python
# src/cdsbi/flows/maf_adapter.py
"""MAFAdapter: wraps nflows MaskedAutoregressiveFlow with the Flow protocol.

MAF does not advertise monotonicity in either θ or X — it's a generic
density estimator. Used as the natural backbone for NPE / NLE / LF2I
baselines so their flow capacity is matched to CDSBI's at the same
budget.

Exposes log_prob(x, context) for NPE/NLE wrappers; the Flow.forward
contract is not used for these baselines (they don't produce a pivot).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow as NFlow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation

from cdsbi.flows.base import Guarantee


class MAFAdapter(nn.Module):
    monotonicity_guarantees = frozenset()  # no monotonicity claims

    def __init__(
        self,
        features: int,
        context_features: int,
        hidden: int = 32,
        num_layers: int = 4,
    ):
        super().__init__()
        transforms = []
        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=features))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=hidden,
                    context_features=context_features if context_features > 0 else None,
                    num_blocks=1,
                )
            )
        transform = CompositeTransform(transforms)
        base = StandardNormal(shape=[features])
        self.flow = NFlow(transform=transform, distribution=base)

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.flow.log_prob(inputs=x, context=context)

    def sample(self, n: int, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.flow.sample(num_samples=n, context=context)

    def forward(
        self, theta: torch.Tensor, context: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flow-protocol shim — MAF is not used as a pivot, so this is unused in v0.

        Returns (r, log_det) so the protocol is satisfied; callers
        responsible for using log_prob() instead.
        """
        raise NotImplementedError("MAFAdapter is used via log_prob/sample, not forward().")

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_maf_adapter.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/flows/maf_adapter.py tests/unit/test_maf_adapter.py
git commit -m "feat(flows): MAFAdapter wrapping nflows.MAF for baseline backbones"
```

---

## Task 11: ConfidenceSet datatype + ConfidenceProcedure subtypes

**Files:**
- Create: `src/cdsbi/confidence_set/__init__.py`, `datatypes.py`, `procedures.py`
- Test: `tests/unit/test_confidence_procedures.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_confidence_procedures.py
import math
import torch
from cdsbi.confidence_set.datatypes import ConfidenceSet
from cdsbi.confidence_set.procedures import PivotBasedProcedure


def test_confidence_set_contains_query():
    cs = ConfidenceSet(
        contains=lambda th: bool((-1.0 <= th <= 1.0)),
        boundary_repr=torch.tensor([-1.0, 1.0]),
        alpha=0.95,
    )
    assert cs.contains(0.0)
    assert not cs.contains(2.0)


def test_pivot_based_procedure_1d_chi_sq_inversion(seed):
    """For r(θ, X) = θ − X with X_obs = 0, the 1D α-confidence set is
       {θ : |θ|² ≤ χ²_{1, α}} = [−z_α, z_α] where z_α = sqrt(χ²_{1, α})."""
    torch.manual_seed(seed)
    proc = PivotBasedProcedure(pivot_fn=lambda theta, x: theta - x, d_theta=1)
    x_obs = torch.tensor([[0.0]])
    cs = proc.confidence_set(x_obs, alpha=0.95)
    # χ²_{1, 0.95} = 3.841 ⇒ z = 1.960
    z = math.sqrt(3.8414588)
    assert abs(cs.boundary_repr[0].item() + z) < 0.01
    assert abs(cs.boundary_repr[1].item() - z) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_confidence_procedures.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement ConfidenceSet datatype**

```python
# src/cdsbi/confidence_set/__init__.py
```

```python
# src/cdsbi/confidence_set/datatypes.py
"""ConfidenceSet dataclass."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class ConfidenceSet:
    """Confidence set at a single X_obs and confidence level α.

    `contains(theta_value)` → bool; `boundary_repr` is shape (2,) in 1D
    (lower, upper) and a set of boundary samples in higher d (added v1+).
    """
    contains: Callable[[float], bool]
    boundary_repr: torch.Tensor
    alpha: float
```

- [ ] **Step 4: Implement ConfidenceProcedure subtypes**

```python
# src/cdsbi/confidence_set/procedures.py
"""ConfidenceProcedure subtypes — what every method's TrainedModel.procedure is.

Each subtype produces a ConfidenceSet via its own mechanism:
- PivotBased: chi-square inversion of ‖r‖² ≤ χ²_{d, α}
- CriticalValue: {θ : T(θ, X) ≤ c_α(θ)} (LF2I)
- PosteriorBased: highest-posterior-density region on sampled posterior (NPE)
- LikelihoodBased: Wilks-style likelihood-ratio inversion (NLE)
- RatioBased: ratio thresholding (NRE)
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable

import torch
from scipy.stats import chi2

from cdsbi.confidence_set.datatypes import ConfidenceSet
from cdsbi.confidence_set.root_find import bisect_1d


@runtime_checkable
class ConfidenceProcedure(Protocol):
    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet: ...


class PivotBasedProcedure:
    """CDSBI: 1D pivot inverted via chi-square."""

    def __init__(self, pivot_fn: Callable, d_theta: int, theta_range: tuple = (-20.0, 20.0)):
        self.pivot_fn = pivot_fn
        self.d_theta = d_theta
        self.theta_range = theta_range

    def pivot(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.pivot_fn(theta, x)

    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        assert self.d_theta == 1, "Multivariate confidence_set lands in v1+."
        thresh = float(chi2.ppf(alpha, df=1))

        def f(theta_val: float) -> float:
            theta = torch.tensor([[theta_val]], dtype=x_obs.dtype, device=x_obs.device)
            r = self.pivot_fn(theta, x_obs)
            return (r.pow(2).sum().item() - thresh)

        # Pivot is monotone in θ at fixed X_obs; r² is U-shaped with min where r=0.
        # Locate the minimum (where r ≈ 0) by bisecting r itself.
        def r_only(theta_val: float) -> float:
            theta = torch.tensor([[theta_val]], dtype=x_obs.dtype, device=x_obs.device)
            return self.pivot_fn(theta, x_obs).item()

        lo, hi = self.theta_range
        center = bisect_1d(r_only, lo, hi, tol=1e-4)
        # Now find left and right zeros of f(θ) = r(θ)² − thresh
        # On each side of the center, r is monotone, so f has a single zero
        left = bisect_1d(f, lo, center, tol=1e-4)
        right = bisect_1d(f, center, hi, tol=1e-4)

        def contains(theta_val: float) -> bool:
            return left <= theta_val <= right

        return ConfidenceSet(
            contains=contains,
            boundary_repr=torch.tensor([left, right]),
            alpha=alpha,
        )


class CriticalValueProcedure:
    """LF2I: {θ : T(θ, X) ≤ c_α(θ)}."""

    def __init__(
        self,
        test_stat_fn: Callable,
        critical_value_fn: Callable,
        d_theta: int,
        theta_range: tuple = (-20.0, 20.0),
    ):
        self.test_stat_fn = test_stat_fn
        self.critical_value_fn = critical_value_fn
        self.d_theta = d_theta
        self.theta_range = theta_range

    def test_statistic(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.test_stat_fn(theta, x)

    def critical_value(self, theta: torch.Tensor, alpha: float) -> torch.Tensor:
        return self.critical_value_fn(theta, alpha)

    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        assert self.d_theta == 1

        def f(theta_val: float) -> float:
            theta = torch.tensor([[theta_val]], dtype=x_obs.dtype, device=x_obs.device)
            t = self.test_stat_fn(theta, x_obs).item()
            c = self.critical_value_fn(theta, alpha).item()
            return t - c  # negative ⇒ in set; positive ⇒ outside

        lo, hi = self.theta_range
        # Find the "accept region" by locating sign changes of f.
        # In well-behaved 1D LF2I, the set is an interval — find its endpoints.
        # Strategy: grid + bisection refinement.
        grid = torch.linspace(lo, hi, 200).tolist()
        signs = [f(t) <= 0 for t in grid]
        # Find first True and last True
        try:
            i_first = signs.index(True)
            i_last = len(signs) - 1 - signs[::-1].index(True)
        except ValueError:
            return ConfidenceSet(
                contains=lambda th: False, boundary_repr=torch.tensor([0.0, 0.0]), alpha=alpha
            )
        left_lo, left_hi = grid[max(i_first - 1, 0)], grid[i_first]
        right_lo, right_hi = grid[i_last], grid[min(i_last + 1, len(grid) - 1)]
        left = bisect_1d(f, left_lo, left_hi, tol=1e-4) if i_first > 0 else grid[0]
        right = bisect_1d(f, right_lo, right_hi, tol=1e-4) if i_last < len(grid) - 1 else grid[-1]

        def contains(theta_val: float) -> bool:
            return left <= theta_val <= right

        return ConfidenceSet(
            contains=contains, boundary_repr=torch.tensor([left, right]), alpha=alpha
        )


class PosteriorBasedProcedure:
    """NPE: highest-posterior-density region via posterior samples."""

    def __init__(self, sample_fn: Callable, d_theta: int):
        self.sample_fn = sample_fn
        self.d_theta = d_theta

    def posterior_samples(self, x_obs: torch.Tensor, n: int = 10_000) -> torch.Tensor:
        return self.sample_fn(x_obs, n)

    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        from cdsbi.confidence_set.hpd import hpd_1d
        samples = self.sample_fn(x_obs, 10_000).flatten()
        return hpd_1d(samples, alpha=alpha)


class LikelihoodBasedProcedure:
    """NLE: Wilks-style likelihood-ratio confidence set."""

    def __init__(self, log_likelihood_fn: Callable, d_theta: int, theta_range: tuple = (-20.0, 20.0)):
        self.log_likelihood_fn = log_likelihood_fn
        self.d_theta = d_theta
        self.theta_range = theta_range

    def log_likelihood(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.log_likelihood_fn(theta, x)

    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        assert self.d_theta == 1
        thresh = float(chi2.ppf(alpha, df=1))  # 2 (ll_max − ll) ≤ χ²_{1, α}

        # find MLE on grid
        lo, hi = self.theta_range
        grid = torch.linspace(lo, hi, 200).view(-1, 1).to(x_obs.device)
        lls = torch.stack([self.log_likelihood_fn(g.view(1, 1), x_obs).squeeze() for g in grid])
        ll_max = lls.max().item()

        def f(theta_val: float) -> float:
            theta = torch.tensor([[theta_val]], dtype=x_obs.dtype, device=x_obs.device)
            ll = self.log_likelihood_fn(theta, x_obs).item()
            return 2 * (ll_max - ll) - thresh

        grid_f = [2 * (ll_max - ll.item()) - thresh for ll in lls]
        signs = [g <= 0 for g in grid_f]
        try:
            i_first = signs.index(True)
            i_last = len(signs) - 1 - signs[::-1].index(True)
        except ValueError:
            return ConfidenceSet(
                contains=lambda th: False, boundary_repr=torch.tensor([0.0, 0.0]), alpha=alpha
            )
        gs = grid.flatten().tolist()
        left_lo, left_hi = gs[max(i_first - 1, 0)], gs[i_first]
        right_lo, right_hi = gs[i_last], gs[min(i_last + 1, len(gs) - 1)]
        left = bisect_1d(f, left_lo, left_hi, tol=1e-4) if i_first > 0 else gs[0]
        right = bisect_1d(f, right_lo, right_hi, tol=1e-4) if i_last < len(gs) - 1 else gs[-1]

        def contains(theta_val: float) -> bool:
            return left <= theta_val <= right

        return ConfidenceSet(
            contains=contains, boundary_repr=torch.tensor([left, right]), alpha=alpha
        )


class RatioBasedProcedure:
    """NRE: ratio thresholding — same shape as Likelihood but using log-ratio."""

    def __init__(self, log_ratio_fn: Callable, d_theta: int, theta_range: tuple = (-20.0, 20.0)):
        self.log_ratio_fn = log_ratio_fn
        self.d_theta = d_theta
        self.theta_range = theta_range

    def log_ratio(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.log_ratio_fn(theta, x)

    def confidence_set(self, x_obs: torch.Tensor, alpha: float) -> ConfidenceSet:
        # Use the same Wilks-style inversion as Likelihood, treating log-ratio as a log-likelihood.
        wrapper = LikelihoodBasedProcedure(self.log_ratio_fn, self.d_theta, self.theta_range)
        return wrapper.confidence_set(x_obs, alpha)
```

- [ ] **Step 5: Run tests to verify they pass** (will require Task 12's `root_find` first; see Task 12)

```bash
# Skip running the procedure tests until Task 12 lands the dependency.
```

- [ ] **Step 6: Commit (interfaces only, no tests run yet)**

```bash
git add src/cdsbi/confidence_set/__init__.py src/cdsbi/confidence_set/datatypes.py src/cdsbi/confidence_set/procedures.py tests/unit/test_confidence_procedures.py
git commit -m "feat(confidence_set): ConfidenceSet + 5 ConfidenceProcedure subtypes (tests pend Task 12)"
```

---

## Task 12: 1D root-finder + HPD extractor

**Files:**
- Create: `src/cdsbi/confidence_set/root_find.py`, `hpd.py`
- Test: `tests/unit/test_root_find.py`, `tests/unit/test_hpd.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_root_find.py
import math
from cdsbi.confidence_set.root_find import bisect_1d


def test_bisect_1d_finds_root():
    root = bisect_1d(lambda x: x - 3.0, 0.0, 10.0, tol=1e-6)
    assert abs(root - 3.0) < 1e-5


def test_bisect_1d_nonlinear():
    root = bisect_1d(lambda x: x**3 - 8.0, 0.0, 10.0, tol=1e-6)
    assert abs(root - 2.0) < 1e-5


def test_bisect_1d_no_sign_change_returns_midpoint():
    """If f doesn't change sign on [a, b], return midpoint with a warning (graceful)."""
    root = bisect_1d(lambda x: x**2 + 1.0, -1.0, 1.0, tol=1e-6)
    # Should not raise; should return something in [-1, 1]
    assert -1.0 <= root <= 1.0
```

```python
# tests/unit/test_hpd.py
import torch
from cdsbi.confidence_set.hpd import hpd_1d


def test_hpd_1d_standard_normal():
    torch.manual_seed(0)
    samples = torch.randn(20_000)
    cs = hpd_1d(samples, alpha=0.95)
    # 95% HPD of N(0,1) ≈ [−1.96, +1.96]
    lo, hi = cs.boundary_repr.tolist()
    assert abs(lo + 1.96) < 0.05
    assert abs(hi - 1.96) < 0.05


def test_hpd_1d_contains_works():
    torch.manual_seed(0)
    samples = torch.randn(10_000)
    cs = hpd_1d(samples, alpha=0.90)
    assert cs.contains(0.0)
    assert not cs.contains(5.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_root_find.py tests/unit/test_hpd.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement root_find**

```python
# src/cdsbi/confidence_set/root_find.py
"""1D bisection root finder."""
from __future__ import annotations

import warnings
from typing import Callable


def bisect_1d(
    f: Callable[[float], float], a: float, b: float, tol: float = 1e-4, max_iter: int = 100
) -> float:
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        warnings.warn(
            f"bisect_1d: f({a})={fa}, f({b})={fb} same sign; returning midpoint.",
            RuntimeWarning,
        )
        return 0.5 * (a + b)
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) < tol:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)
```

- [ ] **Step 4: Implement HPD**

```python
# src/cdsbi/confidence_set/hpd.py
"""1D highest-posterior-density region from samples."""
from __future__ import annotations

import torch

from cdsbi.confidence_set.datatypes import ConfidenceSet


def hpd_1d(samples: torch.Tensor, alpha: float) -> ConfidenceSet:
    """Smallest interval containing fraction α of the sample mass."""
    sorted_samples, _ = torch.sort(samples.flatten())
    n = sorted_samples.numel()
    window = int(n * alpha)
    if window < 2:
        v = float(sorted_samples[n // 2])
        return ConfidenceSet(
            contains=lambda th: th == v, boundary_repr=torch.tensor([v, v]), alpha=alpha
        )
    widths = sorted_samples[window:] - sorted_samples[: n - window]
    i_min = int(torch.argmin(widths))
    lo = float(sorted_samples[i_min])
    hi = float(sorted_samples[i_min + window])

    def contains(theta_val: float) -> bool:
        return lo <= theta_val <= hi

    return ConfidenceSet(contains=contains, boundary_repr=torch.tensor([lo, hi]), alpha=alpha)
```

- [ ] **Step 5: Run tests to verify they pass (including Task 11's deferred test)**

Run: `pytest tests/unit/test_root_find.py tests/unit/test_hpd.py tests/unit/test_confidence_procedures.py -v`
Expected: 7 passed (3 root_find + 2 hpd + 2 procedures).

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/confidence_set/root_find.py src/cdsbi/confidence_set/hpd.py tests/unit/test_root_find.py tests/unit/test_hpd.py
git commit -m "feat(confidence_set): 1D root-finder and HPD extractor"
```

---

## Task 13: Method.Runner base + TrainedModel + BudgetUnreachableError + budget enumeration

**Files:**
- Create: `src/cdsbi/methods/__init__.py`, `base.py`, `budget.py`
- Test: `tests/unit/test_budget.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_budget.py
import pytest
from cdsbi.methods.budget import build_from_budget, BudgetUnreachableError


def test_build_from_budget_picks_closest_within_tolerance():
    # Param-count function: 10 * width^2 (just a stand-in)
    def n_params_for_width(w: int) -> int:
        return 10 * w * w
    widths = [1, 2, 4, 8, 16, 32]
    # target 250 ⇒ widths 4 (160) or 8 (640): 160 closer ⇒ pick 4
    chosen, info = build_from_budget(
        target_params=250, n_params_fn=n_params_for_width, candidates=widths
    )
    assert chosen == 4
    assert info["actual_params"] == 160
    # 160 is 36% off from 250 ⇒ outside ±15% ⇒ should raise
    # Actually let me re-pick: target 170 ⇒ width 4 (160) is 5.9% off (within ±10%)
    chosen, info = build_from_budget(170, n_params_for_width, widths)
    assert chosen == 4
    assert info["status"] == "matched"


def test_build_from_budget_warning_band():
    def n_params_for_width(w: int) -> int:
        return 10 * w * w
    widths = [4, 8]  # 160 or 640
    # target 200: 160 is 20% off (between 10% and 15%? no, 20% is OUTSIDE 15%) ⇒ raise
    with pytest.raises(BudgetUnreachableError):
        build_from_budget(200, n_params_for_width, widths)


def test_build_from_budget_within_warning_band():
    def n_params_for_width(w: int) -> int:
        return 10 * w * w
    widths = [4, 8]  # 160 or 640
    # target 180: 160 is 11.1% off (between 10% and 15%) ⇒ matched_with_warning
    chosen, info = build_from_budget(180, n_params_for_width, widths)
    assert info["status"] == "matched_with_warning"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_budget.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement Methods base**

```python
# src/cdsbi/methods/__init__.py
```

```python
# src/cdsbi/methods/base.py
"""Method.Runner base + TrainedModel + n_params kind conventions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from cdsbi.confidence_set.procedures import ConfidenceProcedure


class BudgetUnreachableError(RuntimeError):
    """Raised when no candidate width lands within ±15% of target_params."""


@dataclass
class TrainedModel:
    procedure: ConfidenceProcedure
    state_dict: dict
    final_loss: float
    n_steps: int
    wall_clock_sec: float
    arch_metadata: dict = field(default_factory=dict)


@runtime_checkable
class Runner(Protocol):
    def fit(self, simulator, config, seed: int) -> TrainedModel: ...
    def n_params(self) -> dict:
        """{'backbone': int, 'head': int, 'calibration_stage': int, 'total': int, 'kind': str}.

        Every field always populated (zero when not applicable):
        - kind='flow' (CDSBI, NPE, NLE): backbone = flow weights; head = scalar α / scale params
        - kind='classifier' (NRE): head = MLP; backbone = 0
        - kind='two_stage' (LF2I): backbone = test-stat flow; calibration_stage = quantile head;
                                   head = 0
        """
        ...
```

- [ ] **Step 4: Implement budget enumeration**

```python
# src/cdsbi/methods/budget.py
"""Closest-candidate width enumeration for matched-budget config."""
from __future__ import annotations

import warnings
from typing import Callable, List, Tuple

from cdsbi.methods.base import BudgetUnreachableError


def build_from_budget(
    target_params: int,
    n_params_fn: Callable[[int], int],
    candidates: List[int],
) -> Tuple[int, dict]:
    """Pick the width whose `n_params_fn(width)` is closest to `target_params`.

    Status tiers:
    - within ±10%: 'matched'
    - within ±15%: 'matched_with_warning' (warning emitted)
    - outside ±15%: BudgetUnreachableError
    """
    counts = [(w, n_params_fn(w)) for w in candidates]
    best_width, best_count = min(counts, key=lambda wc: abs(wc[1] - target_params))
    rel_err = abs(best_count - target_params) / target_params

    if rel_err <= 0.10:
        status = "matched"
    elif rel_err <= 0.15:
        status = "matched_with_warning"
        warnings.warn(
            f"Budget {target_params} matched at {best_count} (width={best_width}, "
            f"rel_err={rel_err:.1%}) — within ±15% but outside ±10%.",
            RuntimeWarning,
        )
    else:
        raise BudgetUnreachableError(
            f"No width in {candidates} lands within ±15% of {target_params} params. "
            f"Closest: width={best_width} ⇒ {best_count} params ({rel_err:.1%} off)."
        )
    return best_width, {"actual_params": best_count, "status": status, "rel_err": rel_err}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_budget.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/methods/__init__.py src/cdsbi/methods/base.py src/cdsbi/methods/budget.py tests/unit/test_budget.py
git commit -m "feat(methods): Runner protocol + TrainedModel + build_from_budget enumeration"
```

---

## Task 14: CDSBIRunner

**Files:**
- Create: `src/cdsbi/methods/cd_sbi.py`
- Test: `tests/integration/test_cd_sbi_smoke.py`

- [ ] **Step 1: Write the failing integration test**

```python
# tests/integration/test_cd_sbi_smoke.py
import torch
from cdsbi.conditioners.identity import Identity
from cdsbi.flows.additive import AdditiveFlow1D
from cdsbi.losses.nfmle import NFMLELoss
from cdsbi.methods.cd_sbi import CDSBIRunner
from cdsbi.reproducibility.seeding import seed_everything
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def test_cdsbi_smoke_trains_briefly(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    flow = AdditiveFlow1D(hidden=8)
    runner = CDSBIRunner(flow=flow, conditioner=Identity(), loss=NFMLELoss())
    cfg = {"n_train": 200, "batch_size": 50, "lr": 3e-3, "n_steps": 50}
    trained = runner.fit(simulator=sim, config=cfg, seed=seed)
    assert trained.final_loss < 5.0
    assert trained.n_steps == 50
    assert trained.procedure is not None


def test_cdsbi_n_params_structured():
    flow = AdditiveFlow1D(hidden=8)
    runner = CDSBIRunner(flow=flow, conditioner=Identity(), loss=NFMLELoss())
    np_dict = runner.n_params()
    assert np_dict["kind"] == "flow"
    assert np_dict["total"] == np_dict["backbone"] + np_dict["head"] + np_dict["calibration_stage"]
    assert np_dict["head"] == 2  # the two α scalars
    assert np_dict["calibration_stage"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_cd_sbi_smoke.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement CDSBIRunner**

```python
# src/cdsbi/methods/cd_sbi.py
"""CDSBIRunner: trains a monotone pivot flow under NF-MLE."""
from __future__ import annotations

import time
from typing import Optional

import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import Runner, TrainedModel


class CDSBIRunner(Runner):
    def __init__(self, flow, conditioner, loss, allow_ablation: bool = False, device: str = "auto"):
        self.flow = flow
        self.conditioner = conditioner
        self.loss = loss
        self.allow_ablation = allow_ablation
        self.device = get_device(device)
        if not allow_ablation:
            self.loss.check_guarantees(flow)

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        from cdsbi.reproducibility.seeding import seed_everything
        rngs = seed_everything(seed)

        self.flow.to(self.device)
        opt = torch.optim.Adam(self.flow.parameters(), lr=config["lr"])

        # Pre-sample full training set
        theta_all, x_all = simulator.sample(config["n_train"], rngs.train)
        theta_all, x_all = theta_all.to(self.device), x_all.to(self.device)

        bs = config["batch_size"]
        n_steps = config["n_steps"]
        n_train = theta_all.shape[0]
        losses = []
        t0 = time.time()
        for step in range(n_steps):
            idx = torch.randint(0, n_train, (bs,), generator=torch.Generator(device="cpu"))
            theta_b, x_b = theta_all[idx], x_all[idx]
            context, log_det_contrib = self.conditioner.encode(x_b)
            r, log_det_flow = self.flow.forward(theta_b, context=context)
            log_det_total = log_det_flow + log_det_contrib
            loss_val = self.loss(r=r, log_det_jac_input=log_det_total)
            opt.zero_grad()
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=5.0)
            opt.step()
            losses.append(loss_val.item())
        wall = time.time() - t0

        # Build PivotBasedProcedure with a closure over the trained flow + conditioner
        flow = self.flow
        conditioner = self.conditioner
        device = self.device

        def pivot_fn(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            theta = theta.to(device)
            x = x.to(device)
            context, _ = conditioner.encode(x)
            r, _ = flow.forward(theta, context=context)
            return r

        procedure = PivotBasedProcedure(pivot_fn=pivot_fn, d_theta=simulator.d_theta)

        return TrainedModel(
            procedure=procedure,
            state_dict=self.flow.state_dict(),
            final_loss=losses[-1],
            n_steps=n_steps,
            wall_clock_sec=wall,
            arch_metadata={
                "flow_class": type(self.flow).__name__,
                "loss_class": type(self.loss).__name__,
                "loss_history_tail": losses[-min(100, len(losses)) :],
            },
        )

    def n_params(self) -> dict:
        backbone = self.flow.a.n_params() + self.flow.b.n_params()
        head = self.flow.n_params() - backbone  # the α scalars
        return {
            "backbone": backbone,
            "head": head,
            "calibration_stage": 0,
            "total": self.flow.n_params(),
            "kind": "flow",
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_cd_sbi_smoke.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/methods/cd_sbi.py tests/integration/test_cd_sbi_smoke.py
git commit -m "feat(methods): CDSBIRunner training loop + PivotBasedProcedure construction"
```

---

## Task 15: NPERunner + NLERunner (wrap sbi)

**Files:**
- Create: `src/cdsbi/methods/npe.py`, `nle.py`
- Test: `tests/integration/test_npe_smoke.py`, `tests/integration/test_nle_smoke.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/integration/test_npe_smoke.py
from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.methods.npe import NPERunner
from cdsbi.reproducibility.seeding import seed_everything
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def test_npe_smoke(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    runner = NPERunner(flow=flow)
    trained = runner.fit(simulator=sim, config={"n_train": 200, "n_epochs": 5}, seed=seed)
    assert trained.procedure is not None


def test_npe_n_params_structured():
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    runner = NPERunner(flow=flow)
    d = runner.n_params()
    assert d["kind"] == "flow"
    assert d["backbone"] > 0
    assert d["calibration_stage"] == 0
```

```python
# tests/integration/test_nle_smoke.py
from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.methods.nle import NLERunner
from cdsbi.reproducibility.seeding import seed_everything
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def test_nle_smoke(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    runner = NLERunner(flow=flow)
    trained = runner.fit(simulator=sim, config={"n_train": 200, "n_epochs": 5}, seed=seed)
    assert trained.procedure is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_npe_smoke.py tests/integration/test_nle_smoke.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement NPERunner**

```python
# src/cdsbi/methods/npe.py
"""NPERunner: wraps sbi.inference.SNPE_C with num_rounds=1 (amortized) and our MAFAdapter."""
from __future__ import annotations

import pickle
import time

import torch
from sbi.inference import SNPE_C
from sbi.utils import BoxUniform

from cdsbi.confidence_set.procedures import PosteriorBasedProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import Runner, TrainedModel
from cdsbi.reproducibility.seeding import seed_everything


def _density_estimator_builder(maf_adapter):
    """Factory that sbi's SNPE_C accepts as `density_estimator`."""
    def build(batch_theta, batch_x):
        return maf_adapter
    return build


class NPERunner(Runner):
    def __init__(self, flow, device: str = "auto"):
        self.flow = flow
        self.device = get_device(device)

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        seed_everything(seed)
        a, b = simulator.theta_range
        prior = BoxUniform(low=torch.tensor([a], device=self.device), high=torch.tensor([b], device=self.device))

        inferer = SNPE_C(
            prior=prior,
            density_estimator=_density_estimator_builder(self.flow.to(self.device)),
            device=str(self.device),
            show_progress_bars=False,
        )
        rngs = seed_everything(seed)
        theta, x = simulator.sample(config["n_train"], rngs.train)
        theta, x = theta.to(self.device), x.to(self.device)
        inferer.append_simulations(theta, x)

        t0 = time.time()
        density_estimator = inferer.train(
            max_num_epochs=config["n_epochs"], show_train_summary=False
        )
        wall = time.time() - t0
        posterior = inferer.build_posterior(density_estimator)

        def sample_fn(x_obs: torch.Tensor, n: int) -> torch.Tensor:
            return posterior.sample((n,), x=x_obs.squeeze(0), show_progress_bars=False)

        procedure = PosteriorBasedProcedure(sample_fn=sample_fn, d_theta=simulator.d_theta)
        final_loss = float(inferer.summary["training_log_probs"][-1]) if inferer.summary.get("training_log_probs") else 0.0

        return TrainedModel(
            procedure=procedure,
            state_dict={"pickle": pickle.dumps(posterior)},
            final_loss=final_loss,
            n_steps=config["n_epochs"],
            wall_clock_sec=wall,
            arch_metadata={"flow_class": "MAFAdapter", "method": "NPE_C"},
        )

    def n_params(self) -> dict:
        backbone = self.flow.n_params()
        return {
            "backbone": backbone,
            "head": 0,
            "calibration_stage": 0,
            "total": backbone,
            "kind": "flow",
        }
```

- [ ] **Step 4: Implement NLERunner**

```python
# src/cdsbi/methods/nle.py
"""NLERunner: wraps sbi.inference.SNLE_A with num_rounds=1 and our MAFAdapter."""
from __future__ import annotations

import pickle
import time

import torch
from sbi.inference import SNLE_A
from sbi.utils import BoxUniform

from cdsbi.confidence_set.procedures import LikelihoodBasedProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import Runner, TrainedModel
from cdsbi.methods.npe import _density_estimator_builder
from cdsbi.reproducibility.seeding import seed_everything


class NLERunner(Runner):
    def __init__(self, flow, device: str = "auto"):
        self.flow = flow
        self.device = get_device(device)

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        seed_everything(seed)
        a, b = simulator.theta_range
        prior = BoxUniform(low=torch.tensor([a], device=self.device), high=torch.tensor([b], device=self.device))

        inferer = SNLE_A(
            prior=prior,
            density_estimator=_density_estimator_builder(self.flow.to(self.device)),
            device=str(self.device),
            show_progress_bars=False,
        )
        rngs = seed_everything(seed)
        theta, x = simulator.sample(config["n_train"], rngs.train)
        theta, x = theta.to(self.device), x.to(self.device)
        inferer.append_simulations(theta, x)

        t0 = time.time()
        density_estimator = inferer.train(max_num_epochs=config["n_epochs"], show_train_summary=False)
        wall = time.time() - t0
        flow = self.flow

        def log_likelihood_fn(theta: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
            # MAFAdapter conditions on theta (the SBI convention: density of X given context=θ)
            return flow.log_prob(x=x_obs, context=theta)

        procedure = LikelihoodBasedProcedure(
            log_likelihood_fn=log_likelihood_fn,
            d_theta=simulator.d_theta,
            theta_range=simulator.theta_range,
        )

        return TrainedModel(
            procedure=procedure,
            state_dict={"flow": flow.state_dict()},
            final_loss=0.0,
            n_steps=config["n_epochs"],
            wall_clock_sec=wall,
            arch_metadata={"flow_class": "MAFAdapter", "method": "NLE_A"},
        )

    def n_params(self) -> dict:
        backbone = self.flow.n_params()
        return {
            "backbone": backbone,
            "head": 0,
            "calibration_stage": 0,
            "total": backbone,
            "kind": "flow",
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/integration/test_npe_smoke.py tests/integration/test_nle_smoke.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/methods/npe.py src/cdsbi/methods/nle.py tests/integration/test_npe_smoke.py tests/integration/test_nle_smoke.py
git commit -m "feat(methods): NPERunner + NLERunner wrapping sbi with MAFAdapter backbone"
```

---

## Task 16: NRERunner (wrap sbi)

**Files:**
- Create: `src/cdsbi/methods/nre.py`
- Test: `tests/integration/test_nre_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_nre_smoke.py
import torch.nn as nn
from cdsbi.methods.nre import NRERunner, build_classifier_mlp
from cdsbi.reproducibility.seeding import seed_everything
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def test_nre_smoke(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    runner = NRERunner(classifier_hidden=16, classifier_depth=2)
    trained = runner.fit(simulator=sim, config={"n_train": 200, "n_epochs": 5}, seed=seed)
    assert trained.procedure is not None


def test_nre_n_params_structured():
    runner = NRERunner(classifier_hidden=16, classifier_depth=2)
    d = runner.n_params()
    assert d["kind"] == "classifier"
    assert d["backbone"] == 0
    assert d["head"] > 0
    assert d["total"] == d["head"]


def test_build_classifier_mlp_param_count():
    mlp = build_classifier_mlp(input_dim=2, hidden=8, depth=2)
    # (2*8 + 8) + (8*8 + 8) + (8*1 + 1) = 24 + 72 + 9 = 105
    assert sum(p.numel() for p in mlp.parameters()) == 105
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_nre_smoke.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement NRERunner**

```python
# src/cdsbi/methods/nre.py
"""NRERunner: wraps sbi.inference.SNRE_B with num_rounds=1 and a controlled MLP classifier."""
from __future__ import annotations

import pickle
import time

import torch
import torch.nn as nn
from sbi.inference import SNRE_B
from sbi.utils import BoxUniform

from cdsbi.confidence_set.procedures import RatioBasedProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import Runner, TrainedModel
from cdsbi.reproducibility.seeding import seed_everything


def build_classifier_mlp(input_dim: int, hidden: int, depth: int) -> nn.Module:
    """MLP classifier head: [input_dim → hidden → hidden → … → 1]."""
    layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, 1))
    return nn.Sequential(*layers)


def _classifier_builder(hidden: int, depth: int):
    def build(batch_theta, batch_x):
        input_dim = batch_theta.shape[-1] + batch_x.shape[-1]
        return build_classifier_mlp(input_dim, hidden, depth)
    return build


class NRERunner(Runner):
    def __init__(self, classifier_hidden: int = 32, classifier_depth: int = 3, device: str = "auto"):
        self.classifier_hidden = classifier_hidden
        self.classifier_depth = classifier_depth
        self.device = get_device(device)

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        seed_everything(seed)
        a, b = simulator.theta_range
        prior = BoxUniform(low=torch.tensor([a], device=self.device), high=torch.tensor([b], device=self.device))

        inferer = SNRE_B(
            prior=prior,
            classifier=_classifier_builder(self.classifier_hidden, self.classifier_depth),
            device=str(self.device),
            show_progress_bars=False,
        )
        rngs = seed_everything(seed)
        theta, x = simulator.sample(config["n_train"], rngs.train)
        theta, x = theta.to(self.device), x.to(self.device)
        inferer.append_simulations(theta, x)

        t0 = time.time()
        ratio_estimator = inferer.train(max_num_epochs=config["n_epochs"], show_train_summary=False)
        wall = time.time() - t0

        def log_ratio_fn(theta: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
            # NRE's ratio estimator computes log r(x, θ); broadcast and return shape (n_theta,)
            n_th = theta.shape[0]
            x_rep = x_obs.expand(n_th, -1)
            return ratio_estimator(torch.cat([theta, x_rep], dim=-1)).squeeze(-1)

        procedure = RatioBasedProcedure(
            log_ratio_fn=log_ratio_fn,
            d_theta=simulator.d_theta,
            theta_range=simulator.theta_range,
        )

        n_class = sum(p.numel() for p in ratio_estimator.parameters())
        return TrainedModel(
            procedure=procedure,
            state_dict={"classifier": ratio_estimator.state_dict()},
            final_loss=0.0,
            n_steps=config["n_epochs"],
            wall_clock_sec=wall,
            arch_metadata={"method": "NRE_B", "classifier_params_actual": n_class},
        )

    def n_params(self) -> dict:
        # Build a dummy classifier to count
        # NRE concatenates [theta, x] ⇒ input_dim = d_theta + d_x = 1 + 1 = 2 in v0
        dummy = build_classifier_mlp(input_dim=2, hidden=self.classifier_hidden, depth=self.classifier_depth)
        head = sum(p.numel() for p in dummy.parameters())
        return {
            "backbone": 0,
            "head": head,
            "calibration_stage": 0,
            "total": head,
            "kind": "classifier",
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/integration/test_nre_smoke.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/methods/nre.py tests/integration/test_nre_smoke.py
git commit -m "feat(methods): NRERunner wrapping sbi with controlled MLP classifier"
```

---

## Task 17: LF2IRunner (two-stage: NLE backbone + quantile regression)

**Files:**
- Create: `src/cdsbi/methods/lf2i.py`
- Test: `tests/integration/test_lf2i_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_lf2i_smoke.py
import torch
from cdsbi.flows.maf_adapter import MAFAdapter
from cdsbi.methods.lf2i import LF2IRunner
from cdsbi.reproducibility.seeding import seed_everything
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def test_lf2i_smoke(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    runner = LF2IRunner(stat_flow=flow, quantile_hidden=16, quantile_depth=2)
    trained = runner.fit(
        simulator=sim,
        config={"n_train_stat": 200, "n_train_quantile": 200, "n_epochs_stat": 5, "n_epochs_quantile": 100, "alpha_grid": [0.5, 0.68, 0.9, 0.95]},
        seed=seed,
    )
    assert trained.procedure is not None
    # Test the procedure returns a confidence set
    cs = trained.procedure.confidence_set(torch.tensor([[0.0]]), alpha=0.9)
    assert cs.boundary_repr.shape == (2,)


def test_lf2i_n_params_structured():
    flow = MAFAdapter(features=1, context_features=1, hidden=8, num_layers=2)
    runner = LF2IRunner(stat_flow=flow, quantile_hidden=16, quantile_depth=2)
    d = runner.n_params()
    assert d["kind"] == "two_stage"
    assert d["backbone"] > 0
    assert d["calibration_stage"] > 0
    assert d["head"] == 0
    assert d["total"] == d["backbone"] + d["calibration_stage"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_lf2i_smoke.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement LF2IRunner**

```python
# src/cdsbi/methods/lf2i.py
"""LF2IRunner — two-stage: (1) NLE-style flow gives test statistic,
(2) pinball-loss MLP gives critical-value function c_α(θ).
"""
from __future__ import annotations

import time
from typing import List

import torch
import torch.nn as nn
from sbi.inference import SNLE_A
from sbi.utils import BoxUniform

from cdsbi.confidence_set.procedures import CriticalValueProcedure
from cdsbi.device import get_device
from cdsbi.methods.base import Runner, TrainedModel
from cdsbi.methods.nle import NLERunner
from cdsbi.methods.npe import _density_estimator_builder
from cdsbi.methods.nre import build_classifier_mlp
from cdsbi.reproducibility.seeding import seed_everything


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    diff = target - pred
    return torch.mean(torch.maximum(alpha * diff, (alpha - 1.0) * diff))


class LF2IRunner(Runner):
    def __init__(
        self,
        stat_flow,
        quantile_hidden: int = 32,
        quantile_depth: int = 3,
        theta_ref: float = None,  # default: prior median
        device: str = "auto",
    ):
        self.stat_flow = stat_flow
        self.quantile_hidden = quantile_hidden
        self.quantile_depth = quantile_depth
        self.theta_ref = theta_ref
        self.device = get_device(device)

    def fit(self, simulator, config: dict, seed: int) -> TrainedModel:
        rngs = seed_everything(seed)
        a, b = simulator.theta_range
        theta_ref = self.theta_ref if self.theta_ref is not None else 0.5 * (a + b)

        # === Stage 1: train NLE-style flow ===
        prior = BoxUniform(low=torch.tensor([a], device=self.device), high=torch.tensor([b], device=self.device))
        inferer = SNLE_A(
            prior=prior,
            density_estimator=_density_estimator_builder(self.stat_flow.to(self.device)),
            device=str(self.device),
            show_progress_bars=False,
        )
        theta, x = simulator.sample(config["n_train_stat"], rngs.train)
        theta, x = theta.to(self.device), x.to(self.device)
        inferer.append_simulations(theta, x)

        t0 = time.time()
        inferer.train(max_num_epochs=config["n_epochs_stat"], show_train_summary=False)
        flow = self.stat_flow

        theta_ref_t = torch.tensor([[theta_ref]], device=self.device, dtype=torch.float32)

        def test_stat_fn(theta: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
            # T(θ, X) = log p(X | θ) − log p(X | θ_ref)
            theta = theta.to(self.device)
            x_obs = x_obs.to(self.device)
            theta_ref_b = theta_ref_t.expand(theta.shape[0], -1)
            ll_th = flow.log_prob(x=x_obs.expand(theta.shape[0], -1), context=theta)
            ll_ref = flow.log_prob(x=x_obs.expand(theta.shape[0], -1), context=theta_ref_b)
            return ll_th - ll_ref

        # === Stage 2: pinball-loss MLP for c_α(θ) ===
        # Generate a held-out calibration set: for each θ, draw X from sim and record T(θ, X)
        theta_cal, x_cal = simulator.sample(config["n_train_quantile"], rngs.eval)
        theta_cal, x_cal = theta_cal.to(self.device), x_cal.to(self.device)
        with torch.no_grad():
            t_cal = torch.stack([
                test_stat_fn(theta_cal[i : i + 1], x_cal[i : i + 1]).squeeze() for i in range(theta_cal.shape[0])
            ])

        alpha_grid: List[float] = config["alpha_grid"]
        # One MLP per α — they're cheap; alternative is one MLP with α as input
        critical_nets = {}
        for alpha in alpha_grid:
            net = build_classifier_mlp(input_dim=simulator.d_theta, hidden=self.quantile_hidden, depth=self.quantile_depth).to(self.device)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            for _ in range(config["n_epochs_quantile"]):
                pred = net(theta_cal).squeeze(-1)
                loss = pinball_loss(pred, t_cal, alpha)
                opt.zero_grad()
                loss.backward()
                opt.step()
            critical_nets[alpha] = net
        wall = time.time() - t0

        def critical_value_fn(theta: torch.Tensor, alpha: float) -> torch.Tensor:
            net = critical_nets[alpha]
            return net(theta.to(self.device)).squeeze(-1)

        procedure = CriticalValueProcedure(
            test_stat_fn=test_stat_fn,
            critical_value_fn=critical_value_fn,
            d_theta=simulator.d_theta,
            theta_range=simulator.theta_range,
        )

        return TrainedModel(
            procedure=procedure,
            state_dict={
                "stat_flow": flow.state_dict(),
                "critical_nets": {a: net.state_dict() for a, net in critical_nets.items()},
            },
            final_loss=0.0,
            n_steps=config["n_epochs_stat"] + config["n_epochs_quantile"],
            wall_clock_sec=wall,
            arch_metadata={"method": "LF2I", "theta_ref": theta_ref, "alpha_grid": alpha_grid},
        )

    def n_params(self) -> dict:
        backbone = self.stat_flow.n_params()
        # Count one quantile net (others have same architecture); 4 α's in v0
        # Use d_theta=1 (v0 only); v1+ this becomes per-simulator
        dummy = build_classifier_mlp(input_dim=1, hidden=self.quantile_hidden, depth=self.quantile_depth)
        per_net = sum(p.numel() for p in dummy.parameters())
        # 4 α-grid points by v0 convention
        calibration_stage = 4 * per_net
        return {
            "backbone": backbone,
            "head": 0,
            "calibration_stage": calibration_stage,
            "total": backbone + calibration_stage,
            "kind": "two_stage",
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/integration/test_lf2i_smoke.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/methods/lf2i.py tests/integration/test_lf2i_smoke.py
git commit -m "feat(methods): LF2IRunner two-stage (NLE backbone + pinball-loss critical values)"
```

---

## Task 18: Diagnostic base + ks_noise_floor

**Files:**
- Create: `src/cdsbi/diagnostics/__init__.py`, `base.py`, `ks_floor.py`
- Test: `tests/diagnostics/test_ks_floor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/diagnostics/test_ks_floor.py
import math
from cdsbi.diagnostics.ks_floor import ks_noise_floor


def test_ks_noise_floor_marginal_N5000():
    expected = 1.628 / math.sqrt(5000)
    assert abs(ks_noise_floor(N=5000, n_bins=1) - expected) < 1e-12


def test_ks_noise_floor_per_bin_N5000_k5():
    expected = 1.628 / math.sqrt(5000 / 5)
    assert abs(ks_noise_floor(N=5000, n_bins=5) - expected) < 1e-12


def test_ks_noise_floor_per_bin_N6000_k7():
    expected = 1.628 / math.sqrt(6000 / 7)
    assert abs(ks_noise_floor(N=6000, n_bins=7) - expected) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/diagnostics/test_ks_floor.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement diagnostics base + ks_floor**

```python
# src/cdsbi/diagnostics/__init__.py
```

```python
# src/cdsbi/diagnostics/base.py
"""Diagnostic protocol + DiagnosticResult."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable, Union

import pandas as pd


@dataclass
class DiagnosticResult:
    name: str
    value: Union[float, pd.Series]
    passed: bool
    noise_floor: float
    n_samples: int
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Diagnostic(Protocol):
    name: str

    def __call__(self, trained, simulator, eval_data) -> DiagnosticResult: ...
```

```python
# src/cdsbi/diagnostics/ks_floor.py
"""KS-test noise floor: 1.628 / sqrt(N/k) for the per-bin case."""
from __future__ import annotations

import math


def ks_noise_floor(N: int, n_bins: int = 1) -> float:
    """Noise floor for a KS test at the 99% level: 1.628/√(N/k)."""
    return 1.628 / math.sqrt(N / n_bins)
```

- [ ] **Step 4: Create tests/diagnostics/__init__.py**

```python
# tests/diagnostics/__init__.py
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/diagnostics/test_ks_floor.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/diagnostics/__init__.py src/cdsbi/diagnostics/base.py src/cdsbi/diagnostics/ks_floor.py tests/diagnostics/test_ks_floor.py tests/diagnostics/__init__.py
git commit -m "feat(diagnostics): Diagnostic protocol + DiagnosticResult + ks_noise_floor"
```

---

## Task 19: PivotRMSE + MarginalPIT diagnostics

**Files:**
- Create: `src/cdsbi/diagnostics/pivot_rmse.py`, `marginal_pit.py`
- Test: `tests/diagnostics/test_pivot_rmse.py`, `tests/diagnostics/test_marginal_pit.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/diagnostics/test_pivot_rmse.py
import torch
from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.pivot_rmse import PivotRMSE
from cdsbi.methods.base import TrainedModel
from cdsbi.simulators.location_normal_1d import LocationNormal1D


def _make_oracle_trained(sim):
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: th - x, d_theta=1)
    return TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0
    )


def test_pivot_rmse_oracle_zero():
    sim = LocationNormal1D()
    trained = _make_oracle_trained(sim)
    theta = torch.linspace(-5, 5, 100).unsqueeze(-1)
    x = theta + 0.1 * torch.randn_like(theta)
    result = PivotRMSE()(trained, sim, eval_data=(theta, x))
    assert result.value < 1e-6  # oracle == truth ⇒ RMSE 0


def test_pivot_rmse_nonzero_on_perturbed_pivot():
    sim = LocationNormal1D()
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: 0.5 * (th - x), d_theta=1)
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0
    )
    theta = torch.linspace(-5, 5, 100).unsqueeze(-1)
    x = theta + 0.1 * torch.randn_like(theta)
    result = PivotRMSE()(trained, sim, eval_data=(theta, x))
    assert result.value > 0.1
```

```python
# tests/diagnostics/test_marginal_pit.py
import numpy as np
import torch
from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.marginal_pit import MarginalPIT
from cdsbi.methods.base import TrainedModel
from cdsbi.simulators.location_normal_1d import LocationNormal1D
from cdsbi.reproducibility.seeding import seed_everything


def test_marginal_pit_oracle_passes(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: th - x, d_theta=1)
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0
    )
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(5000, rng)
    result = MarginalPIT()(trained, sim, eval_data=(theta, x))
    assert result.passed
    assert result.value < result.noise_floor


def test_marginal_pit_catches_underdispersed(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: 0.5 * (th - x), d_theta=1)
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0
    )
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(5000, rng)
    result = MarginalPIT()(trained, sim, eval_data=(theta, x))
    assert not result.passed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/diagnostics/test_pivot_rmse.py tests/diagnostics/test_marginal_pit.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement PivotRMSE**

```python
# src/cdsbi/diagnostics/pivot_rmse.py
"""PivotRMSE: RMSE of trained pivot vs analytical r* on eval data."""
from __future__ import annotations

import torch

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.base import Diagnostic, DiagnosticResult


class PivotRMSE(Diagnostic):
    name = "pivot_rmse"

    def __call__(self, trained, simulator, eval_data) -> DiagnosticResult:
        if not isinstance(trained.procedure, PivotBasedProcedure):
            return DiagnosticResult(
                name=self.name, value=float("nan"), passed=True, noise_floor=0.0,
                n_samples=0, meta={"reason": "not a pivot-based procedure"},
            )
        if simulator.r_star is None:
            return DiagnosticResult(
                name=self.name, value=float("nan"), passed=True, noise_floor=0.0,
                n_samples=0, meta={"reason": "no analytical r*"},
            )
        theta, x = eval_data
        with torch.no_grad():
            r_hat = trained.procedure.pivot(theta, x)
            r_star = simulator.r_star(theta, x)
        rmse = (r_hat - r_star).pow(2).mean().sqrt().item()
        return DiagnosticResult(
            name=self.name, value=rmse, passed=rmse < 0.05,
            noise_floor=0.05, n_samples=theta.shape[0],
        )
```

- [ ] **Step 4: Implement MarginalPIT**

```python
# src/cdsbi/diagnostics/marginal_pit.py
"""MarginalPIT: KS test of Φ(r(θ, X)) vs U(0,1) on training-distribution samples."""
from __future__ import annotations

import torch
from scipy.stats import kstest, norm

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.base import Diagnostic, DiagnosticResult
from cdsbi.diagnostics.ks_floor import ks_noise_floor


class MarginalPIT(Diagnostic):
    name = "marginal_pit"

    def __call__(self, trained, simulator, eval_data) -> DiagnosticResult:
        if not isinstance(trained.procedure, PivotBasedProcedure):
            return DiagnosticResult(
                name=self.name, value=float("nan"), passed=True, noise_floor=0.0,
                n_samples=0, meta={"reason": "not a pivot-based procedure"},
            )
        theta, x = eval_data
        with torch.no_grad():
            r = trained.procedure.pivot(theta, x).flatten()
        u = norm.cdf(r.cpu().numpy())
        ks_stat, _ = kstest(u, "uniform")
        floor = ks_noise_floor(N=u.size, n_bins=1)
        return DiagnosticResult(
            name=self.name, value=float(ks_stat), passed=ks_stat <= floor,
            noise_floor=floor, n_samples=u.size,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/diagnostics/test_pivot_rmse.py tests/diagnostics/test_marginal_pit.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/diagnostics/pivot_rmse.py src/cdsbi/diagnostics/marginal_pit.py tests/diagnostics/test_pivot_rmse.py tests/diagnostics/test_marginal_pit.py
git commit -m "feat(diagnostics): PivotRMSE + MarginalPIT with KS-floor pass/fail"
```

---

## Task 20: ConditionalPIT diagnostic

**Files:**
- Create: `src/cdsbi/diagnostics/conditional_pit.py`
- Test: `tests/diagnostics/test_conditional_pit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/diagnostics/test_conditional_pit.py
import numpy as np
import torch
from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.conditional_pit import ConditionalPIT
from cdsbi.diagnostics.ks_floor import ks_noise_floor
from cdsbi.methods.base import TrainedModel
from cdsbi.simulators.location_normal_1d import LocationNormal1D
from cdsbi.reproducibility.seeding import seed_everything


def test_conditional_pit_per_bin_floor_calculation():
    diag = ConditionalPIT(n_bins=5)
    # The diagnostic's per-bin floor at N=5000 should equal ks_noise_floor(5000, 5)
    assert abs(diag._per_bin_floor(N=5000) - ks_noise_floor(5000, 5)) < 1e-12


def test_conditional_pit_oracle_passes(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: th - x, d_theta=1)
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0
    )
    rng = np.random.default_rng(seed)
    theta, x = sim.sample(5000, rng)
    result = ConditionalPIT(n_bins=5)(trained, sim, eval_data=(theta, x))
    # All per-bin KS should be ≤ per-bin floor for an oracle pivot
    assert result.passed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/diagnostics/test_conditional_pit.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement ConditionalPIT**

```python
# src/cdsbi/diagnostics/conditional_pit.py
"""ConditionalPIT: per-θ-bin KS test of Φ(r(θ, X)) | θ vs U(0,1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import kstest, norm

from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.base import Diagnostic, DiagnosticResult
from cdsbi.diagnostics.ks_floor import ks_noise_floor


class ConditionalPIT(Diagnostic):
    name = "conditional_pit"

    def __init__(self, n_bins: int):
        self.n_bins = n_bins

    def _per_bin_floor(self, N: int) -> float:
        return ks_noise_floor(N=N, n_bins=self.n_bins)

    def __call__(self, trained, simulator, eval_data) -> DiagnosticResult:
        if not isinstance(trained.procedure, PivotBasedProcedure):
            return DiagnosticResult(
                name=self.name, value=float("nan"), passed=True, noise_floor=0.0,
                n_samples=0, meta={"reason": "not a pivot-based procedure"},
            )
        theta, x = eval_data
        N = theta.shape[0]
        floor = self._per_bin_floor(N)
        with torch.no_grad():
            r = trained.procedure.pivot(theta, x).flatten().cpu().numpy()
        theta_np = theta.flatten().cpu().numpy()
        # Bin θ into equal-count bins
        edges = np.quantile(theta_np, np.linspace(0, 1, self.n_bins + 1))
        rows = []
        for k in range(self.n_bins):
            lo, hi = edges[k], edges[k + 1]
            mask = (theta_np >= lo) & (theta_np <= hi)
            r_bin = r[mask]
            if r_bin.size < 10:
                continue
            u_bin = norm.cdf(r_bin)
            ks_stat, _ = kstest(u_bin, "uniform")
            rows.append({
                "theta_0_bin": k,
                "theta_0_center_0": 0.5 * (lo + hi),
                "ks": ks_stat,
                "per_bin_noise_floor": floor,
                "n_per_bin": int(r_bin.size),
                "passed": ks_stat <= floor,
            })
        df = pd.DataFrame(rows)
        passed = bool(df["passed"].all())
        return DiagnosticResult(
            name=self.name, value=df, passed=passed, noise_floor=floor,
            n_samples=N, meta={"n_bins": self.n_bins},
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/diagnostics/test_conditional_pit.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/diagnostics/conditional_pit.py tests/diagnostics/test_conditional_pit.py
git commit -m "feat(diagnostics): ConditionalPIT with explicit n_bins arg and per-bin floor"
```

---

## Task 21: Coverage diagnostic

**Files:**
- Create: `src/cdsbi/diagnostics/coverage.py`
- Test: `tests/diagnostics/test_coverage.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/diagnostics/test_coverage.py
import numpy as np
import torch
from cdsbi.confidence_set.procedures import PivotBasedProcedure
from cdsbi.diagnostics.coverage import Coverage
from cdsbi.methods.base import TrainedModel
from cdsbi.simulators.location_normal_1d import LocationNormal1D
from cdsbi.reproducibility.seeding import seed_everything


def _oracle_trained():
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: th - x, d_theta=1)
    return TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0
    )


def test_coverage_oracle_within_mc_error(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    trained = _oracle_trained()
    diag = Coverage(theta_0_grid=[-3.0, 0.0, 3.0], alpha_grid=[0.5, 0.68, 0.9, 0.95], n_per_theta=2000)
    result = diag(trained, sim, eval_data=None)
    df = result.value  # DataFrame
    # All empirical within 0.03 of nominal (MC error ~ √(0.05·0.95/2000) ≈ 0.005)
    df["err"] = (df["empirical"] - df["nominal"]).abs()
    assert (df["err"] < 0.05).all()


def test_coverage_catches_underdispersed(seed):
    seed_everything(seed)
    sim = LocationNormal1D()
    proc = PivotBasedProcedure(pivot_fn=lambda th, x: 0.5 * (th - x), d_theta=1)
    trained = TrainedModel(
        procedure=proc, state_dict={}, final_loss=0.0, n_steps=0, wall_clock_sec=0.0
    )
    diag = Coverage(theta_0_grid=[0.0], alpha_grid=[0.9], n_per_theta=2000)
    result = diag(trained, sim, eval_data=None)
    df = result.value
    # Under-dispersed pivot ⇒ confidence set too narrow ⇒ empirical coverage < nominal
    assert df["empirical"].iloc[0] < 0.85
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/diagnostics/test_coverage.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement Coverage**

```python
# src/cdsbi/diagnostics/coverage.py
"""Coverage: empirical coverage of C_α(X_obs) at multiple (θ_0, α) — cross-method axis."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch

from cdsbi.diagnostics.base import Diagnostic, DiagnosticResult
from cdsbi.reproducibility.seeding import seed_everything


class Coverage(Diagnostic):
    name = "coverage"

    def __init__(self, theta_0_grid: List[float], alpha_grid: List[float], n_per_theta: int = 1000):
        self.theta_0_grid = theta_0_grid
        self.alpha_grid = alpha_grid
        self.n_per_theta = n_per_theta

    def __call__(self, trained, simulator, eval_data=None) -> DiagnosticResult:
        rng = np.random.default_rng(0)  # deterministic eval RNG; caller seeds outer
        rows = []
        for theta_0 in self.theta_0_grid:
            # Draw n_per_theta X | θ_0 (fix θ; vary X) — bypassing simulator.sample (uses θ ~ ρ)
            theta_t = torch.full((self.n_per_theta, 1), theta_0, dtype=torch.float32)
            eps = rng.standard_normal(size=(self.n_per_theta, 1))
            x = theta_t + torch.from_numpy(eps).float()
            for alpha in self.alpha_grid:
                inside = 0
                for i in range(self.n_per_theta):
                    cs = trained.procedure.confidence_set(x[i : i + 1], alpha=alpha)
                    if cs.contains(theta_0):
                        inside += 1
                empirical = inside / self.n_per_theta
                rows.append({
                    "theta_0_0": float(theta_0),
                    "alpha": float(alpha),
                    "nominal": float(alpha),
                    "empirical": float(empirical),
                    "n_eval": int(self.n_per_theta),
                    "passed": abs(empirical - alpha) <= 0.02,
                    "tolerance": 0.02,
                })
        df = pd.DataFrame(rows)
        passed = bool(df["passed"].all())
        return DiagnosticResult(
            name=self.name, value=df, passed=passed, noise_floor=0.02,
            n_samples=len(self.theta_0_grid) * self.n_per_theta,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/diagnostics/test_coverage.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/diagnostics/coverage.py tests/diagnostics/test_coverage.py
git commit -m "feat(diagnostics): Coverage — empirical coverage at (θ_0, α) grid, cross-method axis"
```

---

## Task 22: Hydra configs

**Files:**
- Create: `configs/config.yaml`, all group YAMLs, composites

- [ ] **Step 1: Create base config**

```yaml
# configs/config.yaml
defaults:
  - target: loc_normal_1d
  - flow: additive_umnn
  - conditioner: identity
  - method: cd_sbi
  - training: adam_3e-3_4k_steps
  - budget: medium
  - experiment: 8_1_replication
  - _self_

device: auto
seed: 0

run_dir: outputs/${experiment.name}/${now:%Y-%m-%d_%H-%M-%S}/method=${method.name},budget=${budget.name},seed=${seed}

hydra:
  run:
    dir: ${run_dir}
  sweep:
    dir: outputs/${experiment.name}/${now:%Y-%m-%d_%H-%M-%S}
    subdir: method=${method.name},budget=${budget.name},seed=${seed}
```

- [ ] **Step 2: Create target/loc_normal_1d.yaml**

```yaml
# configs/target/loc_normal_1d.yaml
name: loc_normal_1d
_target_: cdsbi.simulators.location_normal_1d.LocationNormal1D
theta_range: [-7.0, 7.0]
```

- [ ] **Step 3: Create flow configs**

```yaml
# configs/flow/additive_umnn.yaml
name: additive_umnn
_target_: cdsbi.flows.additive.AdditiveFlow1D
# hidden is set per-budget; placeholder below
hidden: ${budget.flow_hidden}
```

```yaml
# configs/flow/maf.yaml
name: maf
_target_: cdsbi.flows.maf_adapter.MAFAdapter
features: 1
context_features: 1
hidden: ${budget.flow_hidden}
num_layers: 2
```

- [ ] **Step 4: Create conditioner/identity.yaml**

```yaml
# configs/conditioner/identity.yaml
name: identity
_target_: cdsbi.conditioners.identity.Identity
```

- [ ] **Step 5: Create method configs**

```yaml
# configs/method/cd_sbi.yaml
name: cd_sbi
runner_class: cdsbi.methods.cd_sbi.CDSBIRunner
flow: additive_umnn
allow_ablation: false
```

```yaml
# configs/method/npe.yaml
name: npe
runner_class: cdsbi.methods.npe.NPERunner
flow: maf
```

```yaml
# configs/method/nle.yaml
name: nle
runner_class: cdsbi.methods.nle.NLERunner
flow: maf
```

```yaml
# configs/method/nre.yaml
name: nre
runner_class: cdsbi.methods.nre.NRERunner
classifier_hidden: ${budget.classifier_hidden}
classifier_depth: 2
```

```yaml
# configs/method/lf2i.yaml
name: lf2i
runner_class: cdsbi.methods.lf2i.LF2IRunner
stat_flow: maf
quantile_hidden: ${budget.quantile_hidden}
quantile_depth: 2
theta_ref: null  # defaults to prior median in runner
```

- [ ] **Step 6: Create training config**

```yaml
# configs/training/adam_3e-3_4k_steps.yaml
name: adam_3e-3_4k_steps
lr: 3.0e-3
batch_size: 512
n_steps: 4000           # used by CDSBI
n_epochs: 100           # used by sbi-wrapped methods (NPE/NLE/NRE)
n_train: 10000
```

- [ ] **Step 7: Create budget configs**

Each budget sets a target param count and per-method width hints (used as `${budget.flow_hidden}` etc. in flow/method configs).

```yaml
# configs/budget/small.yaml
name: small
target_params: 1000
flow_hidden: 16
classifier_hidden: 24
quantile_hidden: 8
```

```yaml
# configs/budget/medium.yaml
name: medium
target_params: 5000
flow_hidden: 32
classifier_hidden: 50
quantile_hidden: 16
```

```yaml
# configs/budget/large.yaml
name: large
target_params: 25000
flow_hidden: 80
classifier_hidden: 112
quantile_hidden: 40
```

```yaml
# configs/budget/xlarge.yaml
name: xlarge
target_params: 100000
flow_hidden: 168
classifier_hidden: 224
quantile_hidden: 80
```

- [ ] **Step 8: Create experiment composites**

```yaml
# configs/experiment/8_1_replication.yaml
name: 8_1_replication
n_eval: 5000
eval_thetas_interior: [-5.0, -3.0, 0.0, 3.0, 5.0]
eval_thetas_edge: [-7.0, 7.0]
alpha_grid: [0.5, 0.68, 0.9, 0.95]
n_eval_per_theta: 2000

# tells run.py to use this method and the standard medium budget by default
defaults:
  - override /method: cd_sbi
  - override /budget: medium
```

```yaml
# configs/experiment/8_1_baseline_sweep.yaml
name: 8_1_baseline_sweep
n_eval: 5000
eval_thetas_interior: [-5.0, -3.0, 0.0, 3.0, 5.0]
eval_thetas_edge: [-7.0, 7.0]
alpha_grid: [0.5, 0.68, 0.9, 0.95]
n_eval_per_theta: 2000

# Hydra multirun: 5 methods × 4 budgets × 5 seeds = 100 runs
hydra:
  mode: MULTIRUN
  sweeper:
    params:
      method: cd_sbi,npe,nle,nre,lf2i
      budget: small,medium,large,xlarge
      seed: 0,1,2,3,4
```

- [ ] **Step 9: Verify Hydra parses the configs**

Run: `python -c "from hydra import initialize, compose; initialize(version_base=None, config_path='configs'); cfg = compose(config_name='config'); print(cfg.method.name, cfg.budget.target_params)"`
Expected: prints `cd_sbi 5000`.

- [ ] **Step 10: Commit**

```bash
git add configs/
git commit -m "feat(configs): Hydra config groups + composites for §8.1 + baseline sweep"
```

---

## Task 23: CLI entrypoint + parquet writers + STATUS

**Files:**
- Create: `src/cdsbi/experiments/__init__.py`, `run.py`
- Test: `tests/integration/test_runner_e2e.py`

- [ ] **Step 1: Write the failing end-to-end test**

```python
# tests/integration/test_runner_e2e.py
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parents[2]


def test_cli_runs_cdsbi_minimal(tmp_path, repo_root):
    """End-to-end: invoke the CLI on a tiny budget; assert run dir is populated."""
    cmd = [
        "python", "-m", "cdsbi.experiments.run",
        f"hydra.run.dir={tmp_path}/run",
        "method=cd_sbi",
        "budget=small",
        "training.n_steps=20",      # super short for the test
        "training.n_train=200",
        "experiment.n_eval_per_theta=100",
        "experiment.eval_thetas_interior=[0.0]",
        "experiment.alpha_grid=[0.9]",
        "seed=0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    run_dir = tmp_path / "run"
    assert (run_dir / "STATUS").read_text().strip() == "OK"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "env.json").exists()
    env = json.loads((run_dir / "env.json").read_text())
    assert "git_sha" in env
    assert (run_dir / "diagnostics" / "coverage.parquet").exists()
    df = pd.read_parquet(run_dir / "diagnostics" / "coverage.parquet")
    assert set(df.columns) >= {"theta_0_0", "alpha", "nominal", "empirical", "n_eval", "passed", "tolerance"}
    assert (run_dir / "index_row.parquet").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_runner_e2e.py -v`
Expected: ModuleNotFoundError / cdsbi.experiments.run not found.

- [ ] **Step 3: Implement experiments package + CLI entrypoint**

```python
# src/cdsbi/experiments/__init__.py
```

```python
# src/cdsbi/experiments/run.py
"""Hydra CLI entrypoint: build → fit → diagnose → write parquets → STATUS."""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from cdsbi.device import get_device
from cdsbi.reproducibility.env import capture_env
from cdsbi.reproducibility.run_dir import RunDir, RunStatus
from cdsbi.reproducibility.seeding import seed_everything

log = logging.getLogger(__name__)


def _instantiate(target_path: str, **kwargs) -> Any:
    """Light-weight Hydra-style instantiation by import path."""
    module_path, cls_name = target_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)(**kwargs)


def _build_flow(cfg: DictConfig) -> Any:
    """Build a flow from the resolved flow sub-config."""
    flow_dict = OmegaConf.to_container(cfg.flow, resolve=True)
    target = flow_dict.pop("_target_")
    flow_dict.pop("name", None)
    return _instantiate(target, **flow_dict)


def _build_simulator(cfg: DictConfig) -> Any:
    sim_dict = OmegaConf.to_container(cfg.target, resolve=True)
    target = sim_dict.pop("_target_")
    sim_dict.pop("name", None)
    return _instantiate(target, **sim_dict)


def _build_method(cfg: DictConfig, simulator) -> Any:
    """Build a Method.Runner from cfg.method, attaching the flow if needed."""
    m = cfg.method
    runner_class = m.runner_class
    if m.name == "cd_sbi":
        from cdsbi.conditioners.identity import Identity
        from cdsbi.losses.nfmle import NFMLELoss
        flow = _build_flow(cfg)
        return _instantiate(
            runner_class,
            flow=flow,
            conditioner=Identity(),
            loss=NFMLELoss(),
            allow_ablation=m.get("allow_ablation", False),
            device=cfg.device,
        )
    if m.name in ("npe", "nle"):
        flow = _build_flow(cfg)
        return _instantiate(runner_class, flow=flow, device=cfg.device)
    if m.name == "nre":
        return _instantiate(
            runner_class,
            classifier_hidden=m.classifier_hidden,
            classifier_depth=m.classifier_depth,
            device=cfg.device,
        )
    if m.name == "lf2i":
        flow = _build_flow(cfg)
        return _instantiate(
            runner_class,
            stat_flow=flow,
            quantile_hidden=m.quantile_hidden,
            quantile_depth=m.quantile_depth,
            theta_ref=m.theta_ref,
            device=cfg.device,
        )
    raise ValueError(f"Unknown method: {m.name}")


def _fit_config(cfg: DictConfig, method_name: str) -> dict:
    """Translate training cfg into the runner's expected dict."""
    t = cfg.training
    if method_name == "cd_sbi":
        return {"lr": t.lr, "batch_size": t.batch_size, "n_steps": t.n_steps, "n_train": t.n_train}
    if method_name in ("npe", "nle", "nre"):
        return {"n_train": t.n_train, "n_epochs": t.n_epochs}
    if method_name == "lf2i":
        return {
            "n_train_stat": t.n_train,
            "n_train_quantile": t.n_train // 2,
            "n_epochs_stat": t.n_epochs,
            "n_epochs_quantile": 100,
            "alpha_grid": list(cfg.experiment.alpha_grid),
        }
    raise ValueError(method_name)


def _run_diagnostics(cfg: DictConfig, trained, simulator, eval_data, rd: RunDir):
    """Run the 4 v0 diagnostics, write per-diagnostic parquet."""
    from cdsbi.diagnostics.conditional_pit import ConditionalPIT
    from cdsbi.diagnostics.coverage import Coverage
    from cdsbi.diagnostics.marginal_pit import MarginalPIT
    from cdsbi.diagnostics.pivot_rmse import PivotRMSE

    n_bins = max(2, len(cfg.experiment.eval_thetas_interior))
    diagnostics = [
        ("pivot_rmse", PivotRMSE()),
        ("marginal_pit", MarginalPIT()),
        ("conditional_pit", ConditionalPIT(n_bins=n_bins)),
        ("coverage", Coverage(
            theta_0_grid=list(cfg.experiment.eval_thetas_interior),
            alpha_grid=list(cfg.experiment.alpha_grid),
            n_per_theta=cfg.experiment.n_eval_per_theta,
        )),
    ]
    diag_results = {}
    diag_dir = rd.path / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    for name, diag in diagnostics:
        result = diag(trained, simulator, eval_data=eval_data)
        if isinstance(result.value, pd.DataFrame):
            df = result.value
        else:
            df = pd.DataFrame([{
                "value": result.value, "passed": result.passed,
                "noise_floor": result.noise_floor, "n_samples": result.n_samples,
            }])
        df.to_parquet(diag_dir / f"{name}.parquet")
        diag_results[name] = result
    return diag_results


def _write_index_row(cfg: DictConfig, rd: RunDir, trained, diag_results, config_hash, env, n_params):
    """One-row parquet summary."""
    cov_df = pd.read_parquet(rd.path / "diagnostics" / "coverage.parquet")
    coverage_error_max = float((cov_df["empirical"] - cov_df["nominal"]).abs().max())
    marg = diag_results.get("marginal_pit")
    pivot = diag_results.get("pivot_rmse")
    row = {
        "config_hash": config_hash,
        "experiment": cfg.experiment.name,
        "method": cfg.method.name,
        "flow": cfg.flow.name,
        "target": cfg.target.name,
        "budget_name": cfg.budget.name,
        "target_params": int(cfg.budget.target_params),
        "actual_params_total": int(n_params["total"]),
        "actual_params_kind": n_params["kind"],
        "seed": int(cfg.seed),
        "device": env["device"],
        "git_sha": env["git_sha"],
        "dirty_tree": env["dirty_tree"],
        "final_loss": float(trained.final_loss),
        "wall_clock_sec": float(trained.wall_clock_sec),
        "coverage_error_max": coverage_error_max,
        "marginal_ks": float(marg.value) if marg and isinstance(marg.value, float) else None,
        "pivot_rmse": float(pivot.value) if pivot and isinstance(pivot.value, float) else None,
    }
    pd.DataFrame([row]).to_parquet(rd.path / "index_row.parquet")


def _config_hash(cfg: DictConfig) -> str:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    canonical = yaml.safe_dump(resolved, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir_path = Path(os.getcwd())  # Hydra cd's into the run dir
    rd = RunDir(run_dir_path)
    rd.set_status(RunStatus.RUNNING)

    try:
        # Snapshots
        rd.write_atomic(run_dir_path / "config.yaml", OmegaConf.to_yaml(cfg, resolve=True))
        env = capture_env()
        rd.write_atomic(run_dir_path / "env.json", json.dumps(env, indent=2))
        config_hash = _config_hash(cfg)
        rngs = seed_everything(int(cfg.seed))
        rd.write_atomic(run_dir_path / "seeds.json", json.dumps({
            "seed": int(cfg.seed), "config_hash": config_hash,
        }, indent=2))

        # Build + fit
        simulator = _build_simulator(cfg)
        runner = _build_method(cfg, simulator)
        n_params = runner.n_params()
        fit_cfg = _fit_config(cfg, cfg.method.name)
        trained = runner.fit(simulator=simulator, config=fit_cfg, seed=int(cfg.seed))

        # Eval data: training-distribution sample for diagnostics 1-3
        theta_eval, x_eval = simulator.sample(cfg.experiment.n_eval, rngs.eval)

        diag_results = _run_diagnostics(cfg, trained, simulator, (theta_eval, x_eval), rd)
        _write_index_row(cfg, rd, trained, diag_results, config_hash, env, n_params)

        # Don't pickle the procedure (closures) — store arch_metadata only for v0
        torch.save(
            {"arch_metadata": trained.arch_metadata, "final_loss": trained.final_loss},
            run_dir_path / "model.pt",
        )

        rd.set_status(RunStatus.OK)
    except Exception:
        rd.write_atomic(run_dir_path / "stdout.log", traceback.format_exc())
        rd.set_status(RunStatus.FAILED)
        raise


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_runner_e2e.py -v -s`
Expected: 1 passed (may take ~10–30 seconds).

- [ ] **Step 5: Commit**

```bash
git add src/cdsbi/experiments/ tests/integration/test_runner_e2e.py
git commit -m "feat(experiments): Hydra CLI entrypoint with full run-dir + diagnostics + index_row"
```

---

## Task 24: Analysis layer — loaders and paper_table_8_1

**Files:**
- Create: `src/cdsbi/analysis/__init__.py`, `loaders.py`, `paper_tables.py`
- Test: `tests/unit/test_loaders.py`, `tests/unit/test_paper_tables.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_loaders.py
from pathlib import Path
import pandas as pd
from cdsbi.analysis.loaders import load_runs


def test_load_runs_concatenates_index_rows(tmp_path):
    # Build fake run dirs
    for i in range(3):
        d = tmp_path / f"run{i}"
        d.mkdir()
        (d / "STATUS").write_text("OK")
        pd.DataFrame([{"config_hash": f"h{i}", "method": "cd_sbi", "seed": i, "coverage_error_max": 0.01 * i}]).to_parquet(d / "index_row.parquet")
    df = load_runs(str(tmp_path / "*"))
    assert len(df) == 3
    assert set(df["method"]) == {"cd_sbi"}


def test_load_runs_skips_non_ok(tmp_path):
    d_ok = tmp_path / "ok"; d_ok.mkdir()
    (d_ok / "STATUS").write_text("OK")
    pd.DataFrame([{"config_hash": "a", "method": "cd_sbi", "seed": 0}]).to_parquet(d_ok / "index_row.parquet")
    d_fail = tmp_path / "fail"; d_fail.mkdir()
    (d_fail / "STATUS").write_text("FAILED")
    pd.DataFrame([{"config_hash": "b", "method": "cd_sbi", "seed": 1}]).to_parquet(d_fail / "index_row.parquet")
    df = load_runs(str(tmp_path / "*"))
    assert len(df) == 1
    assert df["config_hash"].iloc[0] == "a"
```

```python
# tests/unit/test_paper_tables.py
import pandas as pd
from cdsbi.analysis.paper_tables import paper_table_8_1


def test_paper_table_8_1_pivots_method_x_budget():
    df = pd.DataFrame([
        {"method": "cd_sbi", "budget_name": "medium", "seed": 0, "coverage_error_max": 0.01, "marginal_ks": 0.008, "pivot_rmse": 0.03},
        {"method": "cd_sbi", "budget_name": "medium", "seed": 1, "coverage_error_max": 0.02, "marginal_ks": 0.009, "pivot_rmse": 0.04},
        {"method": "npe", "budget_name": "medium", "seed": 0, "coverage_error_max": 0.05, "marginal_ks": None, "pivot_rmse": None},
    ])
    out = paper_table_8_1(df)
    # Index = method, columns include 'coverage_error_max_mean'
    assert ("cd_sbi", "medium") in out.index
    assert "coverage_error_max_mean" in out.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_loaders.py tests/unit/test_paper_tables.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement loaders**

```python
# src/cdsbi/analysis/__init__.py
```

```python
# src/cdsbi/analysis/loaders.py
"""Run-dir loaders: glob over per-run index_row.parquet files, return one DataFrame."""
from __future__ import annotations

import glob as _glob
from pathlib import Path

import pandas as pd


def load_run(run_dir: str) -> dict:
    rd = Path(run_dir)
    status = (rd / "STATUS").read_text().strip() if (rd / "STATUS").exists() else "UNKNOWN"
    row = pd.read_parquet(rd / "index_row.parquet").iloc[0].to_dict()
    row["status"] = status
    row["run_dir"] = str(rd)
    return row


def load_runs(pattern: str) -> pd.DataFrame:
    """Glob pattern matches run dirs; concatenate their index_row.parquet (status=OK only)."""
    rows = []
    for path in _glob.glob(pattern):
        rd = Path(path)
        status_file = rd / "STATUS"
        if not status_file.exists() or status_file.read_text().strip() != "OK":
            continue
        index_row = rd / "index_row.parquet"
        if not index_row.exists():
            continue
        rows.append(pd.read_parquet(index_row))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
```

- [ ] **Step 4: Implement paper_tables**

```python
# src/cdsbi/analysis/paper_tables.py
"""Paper-table generators for v0 §8.1 comparison."""
from __future__ import annotations

import pandas as pd


def paper_table_8_1(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot (method × budget) → seed-averaged diagnostics."""
    metrics = ["coverage_error_max", "marginal_ks", "pivot_rmse", "actual_params_total"]
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby(["method", "budget_name"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_loaders.py tests/unit/test_paper_tables.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/cdsbi/analysis/ tests/unit/test_loaders.py tests/unit/test_paper_tables.py
git commit -m "feat(analysis): load_runs + paper_table_8_1"
```

---

## Task 25: Intensive replication test (§8.1)

**Files:**
- Create: `tests/intensive/__init__.py`, `tests/intensive/README.md`, `tests/intensive/test_replicate_8_1.py`

- [ ] **Step 1: Create the intensive directory + README**

```python
# tests/intensive/__init__.py
```

```markdown
<!-- tests/intensive/README.md -->
# Intensive replication tests

These are opt-in. Run with:

```
pytest -m intensive
```

The default `pytest` invocation skips them (configured in
`pyproject.toml` via `addopts = "-m 'not intensive'"`).

After non-trivial changes to the loss, flow, or diagnostics layers,
run `pytest -m intensive` locally to verify the §8 replication
numbers still land within tolerance. This typically takes several
minutes per experiment.

GPU note: small-scale runs (§8.1 scale) may be CPU-faster than GPU
due to kernel-launch overhead. The framework picks GPU by default;
override to `device=cpu` per-experiment for small workloads.
```

- [ ] **Step 2: Write the failing replication test**

```python
# tests/intensive/test_replicate_8_1.py
"""Full §8.1 replication: CDSBI on LocationNormal1D matches tolerance bands."""
from pathlib import Path
import subprocess

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.intensive
def test_replicate_8_1_cdsbi_matches_tolerance(tmp_path):
    """Run CDSBI at medium budget across 5 seeds; seed-averaged metrics within tolerance."""
    out_dir = tmp_path / "sweep"
    for seed in range(5):
        cmd = [
            "python", "-m", "cdsbi.experiments.run",
            f"hydra.run.dir={out_dir}/seed={seed}",
            "method=cd_sbi", "budget=medium",
            f"seed={seed}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        assert result.returncode == 0, f"seed {seed} failed:\n{result.stderr}"

    from cdsbi.analysis.loaders import load_runs
    df = load_runs(str(out_dir / "*"))
    assert len(df) == 5
    # Seed-averaged tolerance bands from spec §11(2)
    assert df["pivot_rmse"].mean() <= 0.05, f"pivot_rmse mean = {df['pivot_rmse'].mean()}"
    assert df["marginal_ks"].mean() <= 0.023, f"marginal_ks mean = {df['marginal_ks'].mean()}"
    assert df["coverage_error_max"].mean() <= 0.02, f"coverage_error mean = {df['coverage_error_max'].mean()}"
```

- [ ] **Step 3: Run with default invocation; verify it's skipped**

Run: `pytest tests/intensive/test_replicate_8_1.py -v`
Expected: 1 deselected (skipped because not marked).

- [ ] **Step 4: Run with `-m intensive`; verify it executes**

Run: `pytest -m intensive tests/intensive/test_replicate_8_1.py -v -s`
Expected: 1 passed (takes several minutes). If it fails, the tolerance bands are too tight or there's a bug — investigate before tightening.

- [ ] **Step 5: Commit**

```bash
git add tests/intensive/
git commit -m "test(intensive): §8.1 CDSBI replication against spec tolerance bands"
```

---

## Self-review (run inline after writing the plan)

**Spec coverage check** — each spec section maps to one or more tasks:

| Spec section | Task(s) |
|---|---|
| §3 package layout | Task 1 + per-module tasks 2–24 |
| §4.1 `ConfidenceProcedure` | Task 11 |
| §4.2 per-layer interfaces | Tasks 5 (Simulator), 6 (Flow), 8 (Conditioner), 9 (Loss), 13 (Runner), 18 (Diagnostic) |
| §4.3 dataclasses | Tasks 11 (`ConfidenceSet`), 13 (`TrainedModel`), 18 (`DiagnosticResult`) |
| §4.4 matched-budget rule | Task 13 (enumeration); Task 22 (per-method width configs) |
| §5 `simulators/` | Task 5 |
| §5 `flows/` | Tasks 6 (UMNN), 7 (Additive), 10 (MAF) |
| §5 `conditioners/` | Task 8 |
| §5 `losses/` | Task 9 |
| §5 `methods/` | Tasks 13 (base), 14 (CDSBI), 15 (NPE/NLE), 16 (NRE), 17 (LF2I) |
| §5 `confidence_set/` | Tasks 11 (procedures + datatypes), 12 (root_find + HPD) |
| §5 `diagnostics/` | Tasks 18 (base+ks_floor), 19 (PivotRMSE+MarginalPIT), 20 (ConditionalPIT), 21 (Coverage) |
| §5 `experiments/` | Tasks 22 (configs), 23 (CLI) |
| §5 `analysis/` | Task 24 |
| §6 data flow | Task 23 (CLI orchestrates: build → fit → diagnose → write) |
| §7 config groups | Task 22 |
| §8.1 run-dir layout | Task 23 |
| §8.2 parquet schemas | Tasks 20, 21, 23 (diagnostic writers); Task 23 (index_row) |
| §8.3 reproducibility | Tasks 2 (seeding), 3 (env/RunDir/STATUS), 23 (eval-seed derivation, config_hash) |
| §9 testing | All tasks (TDD); Task 25 (intensive) |
| §10 reproducibility module | Tasks 2, 3, 4 |
| §11 v0 done criteria | Task 25 verifies tolerance bands |

No gaps.

**Placeholder scan** — searched plan for: TBD, TODO, "implement later", "similar to Task N", "add appropriate error handling". None found.

**Type consistency check** — spot-checked:
- `TrainedModel` fields used in Task 14+ match definition in Task 13 ✓
- `DiagnosticResult` fields used in Tasks 19–21 match definition in Task 18 ✓
- `ConfidenceSet.boundary_repr` is `torch.Tensor` in both Task 11 definition and Task 21 usage ✓
- `n_params()` returns dict with same keys (`backbone`, `head`, `calibration_stage`, `total`, `kind`) across all method runners ✓
- `seed_everything` returns `SeededRNGs` with `.train/.eval/.init` everywhere it's called ✓

**One known approximation:** Task 23's `_fit_config` for LF2I sets `n_train_quantile = n_train // 2` and `n_epochs_quantile = 100` as hard-coded heuristics rather than dedicated config fields. If you want these as explicit Hydra config knobs, add them to `configs/method/lf2i.yaml` and read them through.

---

## Execution choice

Plan complete and saved to
`docs/superpowers/plans/2026-05-25-cd-sbi-v0-experiment-infrastructure.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per
task, two-stage review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session, batch
execution with checkpoints for review.

**Which approach?**
