"""Plotting — figures, animations and the plot operators they are built from.

Public API: FigureBuilder, Animator, build_plot_operator(), build_analysis_plot(),
the PlotOperator / AnalysisPlot base classes, and FigureStyle / DEFAULT_STYLE.

Implementation modules (every ``_*.py``) are operator modules: auto-discovered,
internal, and reached by registered name through the two factories. Public
modules are library API and are never scanned, which is what keeps
``regime_map_plot`` (and its scipy import) out of a bare package import.

Example:
    from src.simulation_io.plotting import FigureBuilder, build_plot_operator

    FigureBuilder(config, run_dir).build_all()

    density = build_plot_operator("density")(config)
    density(ax, {"rho": rho}, timestep)
"""

from __future__ import annotations
from src.operators._loader import auto_load_operators
from src.registry import get_operators
from src.simulation_io.plotting.animator import Animator
from src.simulation_io.plotting.base import AnalysisPlot
from src.simulation_io.plotting.base import PlotOperator
from src.simulation_io.plotting.figure_builder import FigureBuilder
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE
from src.simulation_io.plotting.figure_config import FigureStyle

# Auto-discover and import private operator modules for registry registration
auto_load_operators("src.simulation_io.plotting")


def build_plot_operator(name: str) -> type[PlotOperator]:
    """Return the per-timestep panel operator class registered as *name*.

    Args:
        name: A ``plotting`` operator name, e.g. ``"density"`` or ``"pressure"``.

    Returns:
        The operator class; instantiate it with the run's config.
    """
    return get_operators("plotting")[name].target


def build_analysis_plot(name: str) -> type[AnalysisPlot]:
    """Return the snapshot-history plot class registered as *name*.

    Args:
        name: An ``analysis`` operator name, e.g. ``"max_velocity"``.

    Returns:
        The analysis plot class; instantiate it with the run's config.
    """
    return get_operators("analysis")[name].target


__all__ = [
    "DEFAULT_STYLE",
    "AnalysisPlot",
    "Animator",
    "FigureBuilder",
    "FigureStyle",
    "PlotOperator",
    "build_analysis_plot",
    "build_plot_operator",
]
