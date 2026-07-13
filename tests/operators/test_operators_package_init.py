"""Tests for tud_lbm.operators package bootstrap helpers."""

from __future__ import annotations


def test_load_all_imports_only_subpackages(monkeypatch):
    from tud_lbm import operators

    loaded: list[str] = []
    monkeypatch.setattr(
        operators.pkgutil,
        "iter_modules",
        lambda _: [(None, "collision", True), (None, "README", False), (None, "wetting", True)],
    )
    monkeypatch.setattr(operators, "auto_load_operators", loaded.append)

    operators.load_all()

    assert loaded == ["tud_lbm.operators.collision", "tud_lbm.operators.wetting"]


def test_plotting_public_exports_alias_io_plotting_symbols():
    from tud_lbm.io.plotting import Animator
    from tud_lbm.io.plotting import Animator as IOAnimator
    from tud_lbm.io.plotting import FigureBuilder
    from tud_lbm.io.plotting import FigureBuilder as IOFigureBuilder
    from tud_lbm.io.plotting import PlotOperator
    from tud_lbm.io.plotting import PlotOperator as IOPlotOperator

    assert Animator is IOAnimator
    assert FigureBuilder is IOFigureBuilder
    assert PlotOperator is IOPlotOperator
