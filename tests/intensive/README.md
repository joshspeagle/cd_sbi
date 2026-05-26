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
