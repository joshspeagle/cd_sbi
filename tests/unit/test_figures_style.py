"""Style layer: method convention, size presets, headless Agg style application."""
from __future__ import annotations


def test_canonical_order_is_best_to_worst():
    from cdsbi.analysis.figures import style
    assert style.CANONICAL_ORDER == ["cd_sbi", "lf2i_bff", "nre", "nle", "npe"]


def test_every_method_has_colour_and_marker():
    from cdsbi.analysis.figures import style
    for method in style.CANONICAL_ORDER:
        entry = style.METHOD_STYLE[method]
        assert entry["color"].startswith("#")
        assert isinstance(entry["marker"], str) and len(entry["marker"]) >= 1


def test_cdsbi_is_navy():
    from cdsbi.analysis.figures import style
    assert style.METHOD_STYLE["cd_sbi"]["color"] == "#1f2d5a"


def test_sizes_presets_exist_and_are_width_height_tuples():
    from cdsbi.analysis.figures import style
    for name in ("single_column", "double_column", "square", "wide"):
        w, h = style.SIZES[name]
        assert w > 0 and h > 0


def test_apply_style_sets_agg_backend_and_returns_none():
    import matplotlib
    from cdsbi.analysis.figures import style
    assert style.apply_style() is None
    assert matplotlib.get_backend().lower() == "agg"


def test_apply_style_loads_mplstyle_rcparams():
    import matplotlib as mpl
    from cdsbi.analysis.figures import style
    style.apply_style()
    # cdsbi.mplstyle sets these; assert they took effect.
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False
    assert mpl.rcParams["mathtext.fallback"] == "cm"
