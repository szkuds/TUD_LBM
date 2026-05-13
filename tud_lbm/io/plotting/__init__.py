"""Plotting utilities for TUD-LBM.

Public surface:

- ``FigureBuilder`` -- assembles per-timestep composite figures from config.
- ``Animator``      -- encodes saved snapshots into an mp4 or gif.
- ``PlotOperator`` -- abstract base class for individual panel operators.
- ``AnalysisPlot``  -- abstract base class for analysis plot operators.
"""

from __future__ import annotations  # noqa: I001

# Trigger operator self-registration at import time.
from . import analysis as _analysis_mod  # noqa: F401
from . import density as _density_mod  # noqa: F401
from . import force as _force_mod  # noqa: F401
from . import velocity as _velocity_mod  # noqa: F401
from .animator import Animator
from .base import AnalysisPlot
from .base import PlotOperator
from .figure_builder import FigureBuilder

__all__ = ["AnalysisPlot", "Animator", "FigureBuilder", "PlotOperator"]
