"""Plotting utilities for TUD-LBM.

Public surface:

- ``FigureBuilder`` -- assembles per-timestep composite figures from config.
- ``Animator``      -- encodes saved snapshots into an mp4 or gif.
- ``PlotOperator`` -- abstract base class for individual panel operators.
- ``AnalysisPlot``  -- abstract base class for analysis plot operators.
- ``FigureStyle`` / ``DEFAULT_STYLE`` -- centralized figure styling.

Operator modules are auto-discovered, following the repo-wide convention: every
``_*.py`` here is an operator implementation imported by
:func:`~src.operators._loader.auto_load_operators`, and every *public* module is
library API that the scan skips. Adding a plot operator is adding a ``_*.py``
file -- there is no list to edit.

That split is load-bearing rather than cosmetic. ``regime_map_plot`` registers no
operators and pulls in scipy, which doubles this package's import time (~1.5s ->
~3.1s); keeping it public excludes it by the very rule that performs the
discovery, so no deny-list is needed. ``run_comparison`` is likewise deliberately
not an operator (see :mod:`src.registry`). The one exception to "private means
operator" is ``_analysis_common``, shared helpers that register nothing -- it is
imported by the operator modules anyway, so the scan reaching it is a no-op.

The scan runs *last*, after every re-exported name below is bound on the package,
so a scanned module may reach back through the package attribute form.
"""

# ``noqa: I001`` exempts this block from import sorting. Without it the
# formatter and the isort autofix disagree about the blank line below and
# rewrite the file on alternate runs, which has already silently dropped the
# ``auto_load_operators`` call once.
from __future__ import annotations  # noqa: I001

from src.operators._loader import auto_load_operators
from ._ca_theta_plot import plot_contact_angle_vs_capillary_number
from ._ca_theta_plot import plot_dual_axis_ca_theta
from ._ca_theta_plot import save_figure
from ._density import DensityPlotOperator
from ._pressure import BulkPressurePlotOperator
from ._pressure import TotalPressurePlotOperator
from ._scalar_history_plot import MaxVelocityPlot
from ._simulation_csv import build_simulation_csv
from .animator import Animator
from .base import AnalysisPlot
from .base import PlotOperator
from .figure_builder import FigureBuilder
from .figure_config import DEFAULT_STYLE
from .figure_config import FigureStyle

# Registers every remaining operator module. Runs *last*, after the re-exports
# above are bound, so a scanned module may reach back through the package
# attribute form. Removing this line silently empties the plotting and analysis
# operator kinds; tests/io/test_plotting_auto_discovery.py guards it.
auto_load_operators("src.simulation_io.plotting")

__all__ = [
    "DEFAULT_STYLE",
    "AnalysisPlot",
    "Animator",
    "BulkPressurePlotOperator",
    "DensityPlotOperator",
    "FigureBuilder",
    "FigureStyle",
    "MaxVelocityPlot",
    "PlotOperator",
    "TotalPressurePlotOperator",
    "build_simulation_csv",
    "plot_contact_angle_vs_capillary_number",
    "plot_dual_axis_ca_theta",
    "save_figure",
]
