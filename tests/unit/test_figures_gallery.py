"""Gallery generator: manifest -> figures/README.md with PNG embeds."""
from __future__ import annotations

MANIFEST = """\
_hello:
  description: "Hello-world synthetic figure"
  section: "infra"
  source_runs: []
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures._hello:render"
  output:
    pdf: figures/_hello.pdf
    png: figures/_hello.png
e8_demo:
  description: "Cross-method summary demo"
  section: "8.5"
  source_runs:
    - outputs/8_1_baseline_sweep/run
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures._hello:render"
  output:
    pdf: figures/e8_demo.pdf
    png: figures/e8_demo.png
"""


def test_write_gallery_creates_readme(tmp_path):
    from cdsbi.analysis.figures.gallery import write_gallery
    mpath = tmp_path / "manifest.yaml"
    mpath.write_text(MANIFEST)
    out = tmp_path / "figures" / "README.md"
    path = write_gallery(str(mpath), out_path=str(out))
    text = out.read_text()
    assert "Hello-world synthetic figure" in text
    assert "Cross-method summary demo" in text
    # PNG embed for each figure, by BASENAME (link is relative to figures/README.md).
    # Assert the full markdown embed, and that the full path is NOT used — so a
    # regression from Path(...).name back to the full "figures/..." path is caught.
    assert "![_hello](_hello.png)" in text
    assert "![e8_demo](e8_demo.png)" in text
    assert "figures/_hello.png" not in text
    # Source run listed for the data-bound figure
    assert "outputs/8_1_baseline_sweep/run" in text
    assert str(out) == path


def test_gallery_groups_by_section(tmp_path):
    from cdsbi.analysis.figures.gallery import write_gallery
    mpath = tmp_path / "manifest.yaml"
    mpath.write_text(MANIFEST)
    out = tmp_path / "figures" / "README.md"
    write_gallery(str(mpath), out_path=str(out))
    text = out.read_text()
    assert "## infra" in text
    assert "## 8.5" in text
