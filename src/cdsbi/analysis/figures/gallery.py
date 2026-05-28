"""Auto-generate figures/README.md from the manifest: a browsable gallery."""
from __future__ import annotations

from pathlib import Path

from cdsbi.analysis.figures.manifest import load_manifest

DEFAULT_OUT = "figures/README.md"


def write_gallery(manifest_path: str, out_path: str = DEFAULT_OUT) -> str:
    """Render a Markdown gallery grouped by manuscript section. Returns out_path."""
    specs = load_manifest(manifest_path)

    sections: dict[str, list] = {}
    for spec in specs.values():
        sections.setdefault(spec.section, []).append(spec)

    lines = [
        "# CD-SBI Figure Gallery",
        "",
        "_Auto-generated from `configs/figures/manifest.yaml` by "
        "`python -m cdsbi.analysis.figures.render --gallery`. Do not edit by hand._",
        "",
    ]
    for section in sorted(sections):
        lines.append(f"## {section}")
        lines.append("")
        for spec in sorted(sections[section], key=lambda s: s.id):
            png_name = Path(spec.output_png).name
            lines.append(f"### {spec.id} — {spec.description}")
            lines.append("")
            lines.append(f"![{spec.id}]({png_name})")
            lines.append("")
            if spec.source_runs:
                lines.append("Source runs:")
                for run in spec.source_runs:
                    lines.append(f"- `{run}`")
                lines.append("")
    text = "\n".join(lines)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out_path
