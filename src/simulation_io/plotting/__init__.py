"""Plotting utilities for TUD-LBM.

Public surface:

- ``FigureBuilder`` -- assembles per-timestep composite figures from config.
- ``Animator``      -- encodes saved snapshots into an mp4 or gif.
- ``PlotOperator`` -- abstract base class for individual panel operators.
- ``AnalysisPlot``  -- abstract base class for analysis plot operators.
- ``FigureStyle`` / ``DEFAULT_STYLE`` -- centralized figure styling.
"""

from __future__ import annotations  # noqa: I001

# Trigger operator self-registration at import time.
#
# NOT auto-discovered: unlike the operator subpackages, plot operators live in
# public modules, and a blanket public-module scan also pulls in
# ``regime_map_plot`` -> ``analysis.accelerations`` -> ``plotting.figure_config``,
# which re-enters this package before it is initialised. Breaking that cycle
# means moving ``figure_config`` out of the plotting package first.
from . import ca_theta_plot as _ca_theta_plot_mod  # noqa: F401
from . import contact_angle_plot as _contact_angle_plot_mod  # noqa: F401
from . import contact_line_speed_plot as _contact_line_speed_plot_mod  # noqa: F401
from . import density as _density_mod  # noqa: F401
from . import force as _force_mod  # noqa: F401
from . import overview_simulation_inc_snapshots as _overview_mod  # noqa: F401
from . import pressure as _pressure_mod  # noqa: F401
from . import run_comparison as _run_comparison_mod  # noqa: F401
from . import scalar_history_plot as _scalar_history_plot_mod  # noqa: F401
from . import simulation_csv as _simulation_csv_mod  # noqa: F401
from . import velocity as _velocity_mod  # noqa: F401
from .animator import Animator
from .base import AnalysisPlot
from .base import PlotOperator
from .ca_theta_plot import plot_contact_angle_vs_capillary_number
from .ca_theta_plot import plot_dual_axis_ca_theta
from .ca_theta_plot import save_figure
from .figure_builder import FigureBuilder
from .figure_config import DEFAULT_STYLE
from .figure_config import FigureStyle

__all__ = [
    "DEFAULT_STYLE",
    "AnalysisPlot",
    "Animator",
    "FigureBuilder",
    "FigureStyle",
    "PlotOperator",
    "plot_contact_angle_vs_capillary_number",
    "plot_dual_axis_ca_theta",
    "save_figure",
]
