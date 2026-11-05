"""Values of the post-processing options the CLI offers: one definition each.

These are not :class:`~src.config.simulation_config.SimulationConfig` fields; a
run never reads them. They live here so the click choices and the functions
that take them cannot list different values.
"""

from __future__ import annotations
from enum import StrEnum


class Smoothing(StrEnum):
    """How the acceleration series is smoothed before the regime fit."""

    RAW = "raw"
    SAVGOL = "savgol"


class AxisScale(StrEnum):
    """Scale of a regime-map axis; the values are matplotlib's scale names."""

    LINEAR = "linear"
    LOG = "log"
