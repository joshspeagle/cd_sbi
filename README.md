# CD-SBI

**Calibrated Confidence Distributions via Simulation-Based Inference.**

A research project exploring an SBI framework whose training target is a
frequentist *confidence distribution* (CD) with pointwise coverage by
construction, rather than a Bayesian posterior, likelihood, or likelihood
ratio.

## Status

- **Manuscript** (`cd_sbi_v7.tex`, 47 pages) — post-round-3, suitable for
  sharing with collaborators. Three completed vetting rounds (factual
  accuracy, citation hygiene, pedagogy) documented in `reviews/round{1,2,3}/`.
- **v0 experiment infrastructure** (`src/cdsbi/`) — Python package
  implementing the §8.1 baseline sweep (CD-SBI vs NPE/NLE/NRE/LF2I at
  matched parameter budget on `LocationNormal1D`). 70 tests passing;
  single-seed §8.1 validation lands inside the spec's tolerance bands
  (pivot_rmse ≈ 0.043, marginal_ks ≈ 0.012, coverage_error_max ≈ 0.018).
- **v0.3 follow-ups** (before publishing §8.1 numbers): restore the
  finite-sample training regime (currently fresh-batch-per-step only);
  re-tune per-method budget widths.
- **v1+** (multivariate flows, §8.4 ablation, real-data targets, etc.) —
  scheduled per the v0 spec's roadmap; not yet built.

## Contents

- `cd_sbi_v7.tex` — working draft of the framework. Build cycle:
  `pdflatex → bibtex → pdflatex → pdflatex`.
- `cd_sbi.bib` — 45 BibTeX entries; `references/<bibkey>.md` paper notes.
- `src/cdsbi/` — Python package (six layers: `simulators/`, `flows/`,
  `conditioners/`, `losses/`, `methods/`, `diagnostics/` + `confidence_set/`,
  `experiments/`, `analysis/`, `reproducibility/`).
- `configs/` — Hydra config groups (target / flow / conditioner / method /
  training / budget / experiment).
- `tests/` — unit, integration, diagnostics test suites; opt-in `intensive/`
  replication tests.
- `docs/superpowers/specs/` — design specs.
- `docs/superpowers/plans/` — implementation plans.
- `reviews/round{1,2,3}/` — manuscript vetting audit trail.
- `CLAUDE.md` — orientation notes for Claude Code agents working in this
  repository.

## Quickstart

```bash
# Install the package + dev dependencies (Python ≥ 3.10)
pip install -e ".[dev]"

# Run the fast test suite (~30 s)
pytest

# Run a single CD-SBI experiment
python -m cdsbi.experiments.run experiment=8_1_replication seed=0

# Run the full §8.1 baseline sweep (5 methods × 4 budgets × 5 seeds = 100 runs)
python -m cdsbi.experiments.run -m experiment=8_1_baseline_sweep

# Opt-in: full replication of §8.1 numbers against tolerance bands (several minutes)
pytest -m intensive
```

Outputs are written to `outputs/<experiment>/<timestamp>/<run-dir>/`. The
`cdsbi.analysis` module loads run dirs and produces comparison tables.

## Author

Josh Speagle. Comments, counter-examples, and pushback welcome.

## License

See `LICENSE`.
