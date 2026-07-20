"""Tests for src.operators package bootstrap helpers."""

from __future__ import annotations


def test_load_all_imports_only_subpackages(monkeypatch):
    from src import operators

    loaded: list[str] = []
    monkeypatch.setattr(
        operators.pkgutil,
        "iter_modules",
        lambda _: [(None, "collision", True), (None, "README", False), (None, "wetting", True)],
    )
    monkeypatch.setattr(operators, "auto_load_operators", loaded.append)

    operators.load_all()

    assert loaded == ["src.operators.collision", "src.operators.wetting"]


def test_plotting_public_exports_alias_io_plotting_symbols():
    from src.simulation_io.plotting import Animator
    from src.simulation_io.plotting import Animator as IOAnimator
    from src.simulation_io.plotting import FigureBuilder
    from src.simulation_io.plotting import FigureBuilder as IOFigureBuilder
    from src.simulation_io.plotting import PlotOperator
    from src.simulation_io.plotting import PlotOperator as IOPlotOperator

    assert Animator is IOAnimator
    assert FigureBuilder is IOFigureBuilder
    assert PlotOperator is IOPlotOperator
