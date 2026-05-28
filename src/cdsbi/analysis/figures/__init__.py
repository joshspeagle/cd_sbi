"""CD-SBI visualization suite: style, data loading, figure builders, render CLI.

Layers:
- style:    shared mplstyle, method colour/marker convention, size presets
- data_io:  aggregates (parquet) and checkpoints (model.pt) loaders
- panels:   reusable Axes-level chart primitives (populated in F1)
- figures:  figure-level builders, one per catalogue entry
- render:   CLI that reads the manifest and writes figures/<id>.{pdf,png}
"""
