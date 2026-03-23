"""Plotting utilities for TUD-LBM.

Public surface
--------------
FigureBuilder  — assembles per-timestep composite figures from config.
PlotOperator   — abstract base class for individual panel operators.
visualise      — backward-compatible entry point for simulation objects
                 or run directories.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

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
    """
    # TODO: Need make this function with adapters
    config = json.load(open(run_dir + "/config.json"))
    if title and "plot_title" not in config:
        config["plot_title"] = title

    builder = FigureBuilder(config=config, run_dir=run_dir)
    builder.build_all(skip=skip)


__all__ = ["FigureBuilder", "PlotOperator", "visualise"]
