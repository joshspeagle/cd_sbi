"""Render CLI: read the manifest, build each figure, save PDF + PNG.

Builders are pure (return a Figure); this module owns all disk IO.

Usage:
    python -m cdsbi.analysis.figures.render --fig e8_headline_summary
    python -m cdsbi.analysis.figures.render --all
    python -m cdsbi.analysis.figures.render --all --section 8.4
    python -m cdsbi.analysis.figures.render --gallery        # (added in Task 9)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cdsbi.analysis.figures.manifest import FigureSpec, load_manifest

DEFAULT_MANIFEST = "configs/figures/manifest.yaml"


def render_one(spec: FigureSpec) -> None:
    """Build one figure and save it to its PDF + PNG paths.

    Suppresses the timestamp metadata matplotlib otherwise embeds, so a
    re-render of an unchanged figure produces byte-identical output and does
    not create spurious git diffs on the tracked figures/ artifacts.
    """
    builder = spec.resolve_builder()
    fig = builder(spec)
    for out in (spec.output_pdf, spec.output_png):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    # PDF embeds a CreationDate by default; None drops it. PNG embeds a
    # Software tEXt chunk; None drops it.
    fig.savefig(spec.output_pdf, metadata={"CreationDate": None})
    fig.savefig(spec.output_png, metadata={"Software": None})
    import matplotlib.pyplot as plt
    plt.close(fig)


def render_all(manifest_path: str = DEFAULT_MANIFEST, section: str | None = None) -> list[str]:
    """Render every figure (optionally filtered by section). Returns rendered ids."""
    specs = load_manifest(manifest_path)
    rendered = []
    for fig_id, spec in specs.items():
        if section is not None and spec.section != section:
            continue
        render_one(spec)
        rendered.append(fig_id)
    return rendered


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render CD-SBI figures from the manifest.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--fig", help="Render a single figure by id.")
    parser.add_argument("--all", action="store_true", help="Render all figures.")
    parser.add_argument("--section", help="Filter --all to one manuscript section.")
    parser.add_argument("--gallery", action="store_true", help="Regenerate figures/README.md.")
    args = parser.parse_args(argv)

    if args.fig:
        specs = load_manifest(args.manifest)
        render_one(specs[args.fig])
    if args.all:
        rendered = render_all(args.manifest, section=args.section)
        print(f"Rendered {len(rendered)} figure(s): {', '.join(rendered)}")
    if args.gallery:
        from cdsbi.analysis.figures.gallery import write_gallery
        path = write_gallery(args.manifest)
        print(f"Wrote gallery: {path}")
    if not (args.fig or args.all or args.gallery):
        parser.error("nothing to do: pass --fig <id>, --all, and/or --gallery")


if __name__ == "__main__":
    main()
