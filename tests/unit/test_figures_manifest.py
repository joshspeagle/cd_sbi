"""Manifest schema: parse manifest.yaml into FigureSpec objects."""
from __future__ import annotations

import pytest

MANIFEST_YAML = """\
_hello:
  description: "Hello-world synthetic figure"
  section: "infra"
  source_runs: []
  checkpoint_runs: []
  builder: "os.path:join"
  output:
    pdf: figures/_hello.pdf
    png: figures/_hello.png
e8_demo:
  description: "Demo with sources"
  section: "8.5"
  source_runs:
    - outputs/a
    - outputs/b
  builder: "os.path:join"
  output:
    pdf: figures/e8_demo.pdf
    png: figures/e8_demo.png
"""


def _write(tmp_path, text):
    p = tmp_path / "manifest.yaml"
    p.write_text(text)
    return str(p)


def test_load_manifest_returns_specs_keyed_by_id(tmp_path):
    from cdsbi.analysis.figures.manifest import load_manifest
    specs = load_manifest(_write(tmp_path, MANIFEST_YAML))
    assert set(specs) == {"_hello", "e8_demo"}


def test_figurespec_fields(tmp_path):
    from cdsbi.analysis.figures.manifest import load_manifest
    spec = load_manifest(_write(tmp_path, MANIFEST_YAML))["e8_demo"]
    assert spec.id == "e8_demo"
    assert spec.section == "8.5"
    assert spec.source_runs == ["outputs/a", "outputs/b"]
    assert spec.checkpoint_runs == []
    assert spec.builder == "os.path:join"
    assert spec.output_pdf == "figures/e8_demo.pdf"
    assert spec.output_png == "figures/e8_demo.png"


def test_checkpoint_runs_defaults_to_empty_when_absent(tmp_path):
    from cdsbi.analysis.figures.manifest import load_manifest
    text = MANIFEST_YAML.replace("  checkpoint_runs: []\n", "", 1)  # drop _hello's line
    spec = load_manifest(_write(tmp_path, text))["_hello"]
    assert spec.checkpoint_runs == []


def test_resolve_builder_imports_callable(tmp_path):
    from cdsbi.analysis.figures.manifest import load_manifest
    spec = load_manifest(_write(tmp_path, MANIFEST_YAML))["_hello"]
    fn = spec.resolve_builder()
    assert callable(fn)


def test_missing_required_field_raises(tmp_path):
    from cdsbi.analysis.figures.manifest import load_manifest
    bad = "x:\n  description: no builder here\n"
    with pytest.raises(KeyError):
        load_manifest(_write(tmp_path, bad))


def test_shipped_manifest_builders_resolve():
    """The real configs/figures/manifest.yaml must parse and every builder
    reference must import to a real callable. The MANIFEST_YAML tests above use
    os.path:join as a parse-only stand-in; this test guards the actual config so
    a renamed/typo'd builder string is caught at the manifest layer, not only
    in the render integration test."""
    from cdsbi.analysis.figures.manifest import load_manifest
    specs = load_manifest("configs/figures/manifest.yaml")
    assert specs, "shipped manifest is empty"
    for fig_id, spec in specs.items():
        fn = spec.resolve_builder()
        assert callable(fn), f"{fig_id} builder {spec.builder!r} did not resolve to a callable"
