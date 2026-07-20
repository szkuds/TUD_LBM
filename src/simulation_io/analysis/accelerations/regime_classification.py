"""Classify a droplet run into pinning / viscous / inertial / unknown.

Built on top of :mod:`src.simulation_io.analysis.accelerations.acceleration_analysis`:
pinning is decided directly from the centroid excursion, while
viscous/inertial/unknown are decided from the sign of the Ca(t) slope within
the acceleration-derived slope window.
"""

from __future__ import annotations
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import NamedTuple
import numpy as np
from src.simulation_io.analysis.accelerations.acceleration_analysis import find_slope_window

if TYPE_CHECKING:
    from src.simulation_io.analysis.accelerations.acceleration_analysis import AccelerationResult


class Regime(StrEnum):
    """Centralized regime labels used by acceleration-based classification."""

    PINNING = "Pinning"
    DISSIPATIVE = "Dissipative"
    INERTIAL = "Inertial"
    UNKNOWN = "unknown"


_PINNING_FRACTION_OF_R0 = 0.5


class RegimeResult(NamedTuple):
    """Outcome of regime classification for one run."""

    regime: Regime
    slope: float | None
    is_pinned: bool
    window: tuple[int, int] | None


def is_pinned(cm_x: np.ndarray, r_zero: float) -> bool:
    """True when the centroid excursion is below half the initial radius."""
    return bool(np.max(cm_x) - np.min(cm_x) < _PINNING_FRACTION_OF_R0 * r_zero)


def classify_regime(cm_x: np.ndarray, r_zero: float, accel_result: AccelerationResult) -> RegimeResult:
    """Classify a run as pinning, viscous, inertial, or unknown.

    Pinning takes priority and does not require acceleration data. Otherwise,
    the slope of ``Ca`` vs iteration is fit within the acceleration-derived
    slope window: negative slope is viscous, non-negative is inertial. No
    usable window (no clean peak pair, or fewer than two points) is unknown.
    """
    pinned = is_pinned(cm_x, r_zero)
    if pinned:
        return RegimeResult(regime=Regime.PINNING, slope=None, is_pinned=True, window=None)

    window = find_slope_window(accel_result)
    if window is None:
        return RegimeResult(regime=Regime.UNKNOWN, slope=None, is_pinned=False, window=None)

    start, end = window
    slope, _intercept = np.polyfit(
        accel_result.iteration[start : end + 1],
        accel_result.ca[start : end + 1],
        1,
    )
    regime = Regime.DISSIPATIVE if slope < 0 else Regime.INERTIAL
    return RegimeResult(regime=regime, slope=float(slope), is_pinned=False, window=window)
