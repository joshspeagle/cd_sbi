"""Render CLI: manifest -> builder -> saved PDF + PNG. Section/all filtering."""
from __future__ import annotations

MANIFEST = """\
_hello:
  description: "Hello-world synthetic figure"
  section: "infra"
  source_runs: []
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures._hello:render"
  output:
    pdf: {pdf}
    png: {png}
second:
  description: "Second figure, different section"
  section: "8.5"
  source_runs: []
  checkpoint_runs: []
  builder: "cdsbi.analysis.figures.figures._hello:render"
  output:
    pdf: {pdf2}
    png: {png2}
"""


def _manifest(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    text = MANIFEST.format(
        pdf=out / "_hello.pdf", png=out / "_hello.png",
        pdf2=out / "second.pdf", png2=out / "second.png",
    )
    mpath = tmp_path / "manifest.yaml"
    mpath.write_text(text)
    return mpath, out


def test_render_one_writes_pdf_and_png(tmp_path):
    # Import load_manifest from its home module, not via render's namespace,
    # so the test doesn't break if render.py ever switches to a lazy import.
    from cdsbi.analysis.figures.render import render_one
    from cdsbi.analysis.figures.manifest import load_manifest
    mpath, out = _manifest(tmp_path)
    specs = load_manifest(str(mpath))
    render_one(specs["_hello"])
    assert (out / "_hello.pdf").stat().st_size > 0
    assert (out / "_hello.png").stat().st_size > 0


def test_render_all_writes_every_figure(tmp_path):
    from cdsbi.analysis.figures.render import render_all
    mpath, out = _manifest(tmp_path)
    render_all(str(mpath))
    for stem in ("_hello", "second"):
        assert (out / f"{stem}.pdf").stat().st_size > 0
        assert (out / f"{stem}.png").stat().st_size > 0


def test_render_all_section_filter(tmp_path):
    from cdsbi.analysis.figures.render import render_all
    mpath, out = _manifest(tmp_path)
    render_all(str(mpath), section="8.5")
    assert (out / "second.pdf").exists()
    assert not (out / "_hello.pdf").exists()
