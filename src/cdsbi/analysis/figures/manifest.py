"""Manifest parsing: configs/figures/manifest.yaml -> {id: FigureSpec}."""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable

import yaml


@dataclass
class FigureSpec:
    """One figure's binding: data sources, builder, output paths."""
    id: str
    description: str
    section: str
    builder: str
    output_pdf: str
    output_png: str
    source_runs: list[str] = field(default_factory=list)
    checkpoint_runs: list[str] = field(default_factory=list)

    def resolve_builder(self) -> Callable[["FigureSpec"], object]:
        """Import the `module:function` builder reference and return the callable."""
        module_path, fn_name = self.builder.split(":")
        module = importlib.import_module(module_path)
        return getattr(module, fn_name)


def load_manifest(path: str) -> dict[str, FigureSpec]:
    """Parse a manifest YAML into FigureSpec objects keyed by figure id.

    Raises KeyError if a required field (builder/description/section/output)
    is missing for any entry.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    specs: dict[str, FigureSpec] = {}
    for fig_id, entry in raw.items():
        output = entry["output"]  # KeyError if missing -> surfaced to caller
        specs[fig_id] = FigureSpec(
            id=fig_id,
            description=entry["description"],
            section=entry["section"],
            builder=entry["builder"],
            output_pdf=output["pdf"],
            output_png=output["png"],
            source_runs=list(entry.get("source_runs") or []),
            checkpoint_runs=list(entry.get("checkpoint_runs") or []),
        )
    return specs
