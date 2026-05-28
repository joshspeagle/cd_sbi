"""Shared figure style: method convention, size presets, headless style application."""
from __future__ import annotations

from importlib import resources

import matplotlib

# Canonical narrative order (best -> worst), used for legend and category ordering.
CANONICAL_ORDER = ["cd_sbi", "lf2i_bff", "nre", "nle", "npe"]

# Method colour/marker convention. Keys match the `method` column in index_row.parquet.
METHOD_STYLE = {
    "cd_sbi":   {"color": "#1f2d5a", "marker": "o", "label": "CD-SBI"},
    "lf2i_bff": {"color": "#e8743b", "marker": "s", "label": "LF2I-BFF"},
    "nre":      {"color": "#7f7f7f", "marker": "X", "label": "NRE"},
    "nle":      {"color": "#2ca02c", "marker": "D", "label": "NLE"},
    "npe":      {"color": "#6baed6", "marker": "^", "label": "NPE"},
}

# Figure-size presets in inches (width, height).
SIZES = {
    "single_column": (3.4, 2.6),
    "double_column": (7.0, 3.0),
    "square":        (3.4, 3.4),
    "wide":          (7.0, 2.4),
}


def apply_style() -> None:
    """Force the Agg backend and apply the packaged cdsbi.mplstyle. Idempotent."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (ensure pyplot binds to Agg)
    style_path = resources.files("cdsbi.analysis.figures") / "cdsbi.mplstyle"
    # Fail loudly and clearly if the style file isn't shipped, rather than
    # surfacing a baffling matplotlib OSError mid-figure-build.
    assert style_path.is_file(), f"cdsbi.mplstyle not found at {style_path}"
    matplotlib.style.use(str(style_path))
