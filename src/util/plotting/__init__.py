"""Plotting utilities for TUD-LBM.

Public surface:

- ``FigureBuilder`` -- assembles per-timestep composite figures from config.
- ``PlotOperator`` -- abstract base class for individual panel operators.
- ``visualise`` -- backward-compatible entry point for simulation objects
  or run directories.
"""

from __future__ import annotations

import json
from pathlib import Path

# Trigger operator self-registration at import time.
from . import analysis as _analysis_mod  # noqa: F401
from . import density as _density_mod  # noqa: F401
from . import force as _force_mod  # noqa: F401
from . import velocity as _velocity_mod  # noqa: F401
from .base import PlotOperator
from .figure_builder import FigureBuilder


def visualise(run_dir, title: str = "LBM Simulation Results", skip: int = 0) -> None:
    """Plotting entry point.

    Accepts a run-directory
    path. In both cases, figures are rendered by :class:`FigureBuilder`.

    Note:
        The *title* parameter is retained for backward compatibility but
        is no longer used.  The figure title is sourced from
        ``SimulationConfig.simulation_name`` instead.
    """
    # TODO: Need make this function with adapters
    from config import from_dict

    with Path(run_dir + "/config.json").open() as _fh:
        raw = json.load(_fh)

    config = from_dict(raw)

    builder = FigureBuilder(config=config, run_dir=run_dir)
    builder.build_all(skip=skip)


__all__ = ["FigureBuilder", "PlotOperator", "visualise"]
