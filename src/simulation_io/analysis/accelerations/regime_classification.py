"""Classify a droplet run into pinning / dissipative / capillary / steady / unknown.

Built on top of :mod:`src.simulation_io.analysis.accelerations.acceleration_analysis`:
pinning is decided directly from the centroid excursion, while the remaining
labels come from the trend of ``Ca(t)`` within the acceleration-derived slope
window.

The trend is only believed when it is large enough to matter *and* larger than
its own uncertainty. A droplet crossing a chemical step typically reaches a
terminal velocity on the step, so the window lands on a plateau where the
least-squares slope is set by scatter rather than by physics -- measured over
eighteen chemical-step runs, every moving one drifted by at most 4.4% of its
mean ``Ca`` while individual fitted slopes changed sign under a four-sample
shift of the window. Branching on the bare sign of that slope therefore
labelled physically identical runs differently. ``Regime.STEADY`` names the
plateau instead of forcing it to one side.
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
    CAPILLARY = "Capillary"
    STEADY = "Steady"
    UNKNOWN = "unknown"


_PINNING_FRACTION_OF_R0 = 0.5

#: Minimum ``|slope * window_span| / mean(Ca)`` for a trend to count. Chemical-step
#: runs that reach a terminal velocity on the step drift by only a few percent
#: across the window, and that residual drift is monotone in ``Bo_parallel``
#: (-0.26% at 0.21, -2.26% at 0.26, -4.36% at 0.32) -- real, but far smaller than
#: the accelerating and decelerating phases this classifier is meant to separate.
#: Lower this to resolve that fine structure; the value is reported on the
#: diagnostic plot so the choice stays visible per run.
_MIN_RELATIVE_DRIFT = 0.05

#: Minimum ``|slope| / stderr(slope)``. Guards the short windows that a coarse
#: ``save_interval`` produces, where a few points of scatter can look like a trend.
_MIN_SLOPE_T_STATISTIC = 2.0

#: Perturbations of the window *start* used to check that a significant trend is
#: not an artefact of where the window happens to begin. The window start is a
#: fixed offset past the acceleration peak, so it can still catch the tail of the
#: ramp into the plateau; a label that flips under these shifts is not reported.
_ROBUSTNESS_START_OFFSETS = (-4, 0, 4)

_MIN_POINTS_FOR_TREND = 3


class TrendFit(NamedTuple):
    """Linear trend of ``Ca`` over a window, with its significance."""

    slope: float
    drift: float
    t_statistic: float

    @property
    def is_significant(self) -> bool:
        """True when the trend is both large enough and larger than its own error."""
        return abs(self.drift) >= _MIN_RELATIVE_DRIFT and abs(self.t_statistic) >= _MIN_SLOPE_T_STATISTIC


class RegimeResult(NamedTuple):
    """Outcome of regime classification for one run."""

    regime: Regime
    slope: float | None
    is_pinned: bool
    window: tuple[int, int] | None
    drift: float | None = None
    t_statistic: float | None = None
    is_robust: bool = True


def is_pinned(cm_x: np.ndarray, r_zero: float) -> bool:
    """True when the centroid excursion is below half the initial radius."""
    return bool(np.max(cm_x) - np.min(cm_x) < _PINNING_FRACTION_OF_R0 * r_zero)


def fit_trend(x: np.ndarray, y: np.ndarray) -> TrendFit | None:
    """Least-squares slope of *y* over *x*, its relative drift and t-statistic.

    ``drift`` is the total change the fit predicts across the window as a
    fraction of ``mean(y)`` -- the scale-free measure of whether the trend
    matters. ``t_statistic`` is the slope over its own standard error. Returns
    ``None`` when the fit is not defined: fewer than three points (no residual
    degrees of freedom), a degenerate *x*, or a zero mean to normalise by. An
    exact fit has no residual and so gives an infinite t-statistic, which is the
    right answer for a noiseless straight line.
    """
    if x.size < _MIN_POINTS_FOR_TREND or np.std(x) == 0.0:
        return None
    mean_y = float(np.mean(y))
    if mean_y == 0.0:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    drift = float(slope) * float(x[-1] - x[0]) / mean_y

    residual_std = float(np.std(y - (slope * x + intercept), ddof=_MIN_POINTS_FOR_TREND - 1))
    if residual_std == 0.0:
        t_statistic = float(np.inf) * np.sign(slope) if slope != 0 else 0.0
    else:
        stderr = residual_std / (float(np.std(x)) * np.sqrt(x.size))
        t_statistic = float(slope) / stderr
    return TrendFit(slope=float(slope), drift=drift, t_statistic=float(t_statistic))


def _sign_is_stable(accel_result: AccelerationResult, window: tuple[int, int], slope: float) -> bool:
    """True when every significant refit at a shifted window start agrees in sign."""
    _start, end = window
    reference = slope >= 0
    for offset in _ROBUSTNESS_START_OFFSETS:
        start = _start + offset
        if start < 0 or end - start + 1 < _MIN_POINTS_FOR_TREND:
            continue
        fit = fit_trend(accel_result.iteration[start : end + 1], accel_result.ca[start : end + 1])
        if fit is not None and fit.is_significant and (fit.slope >= 0) != reference:
            return False
    return True


def classify_regime(cm_x: np.ndarray, r_zero: float, accel_result: AccelerationResult) -> RegimeResult:
    """Classify a run as pinning, dissipative, capillary, steady, or unknown.

    Pinning takes priority and does not require acceleration data. Otherwise the
    trend of ``Ca`` vs iteration is fit within the acceleration-derived slope
    window. A trend that is smaller than ``_MIN_RELATIVE_DRIFT`` or than its own
    standard error is a plateau, reported as steady; a significant trend whose
    sign flips under a shift of the window start is not trusted and reported as
    unknown; otherwise a negative slope is dissipative and a non-negative one
    capillary. No usable window (no clean peak pair, or too few points) is also
    unknown.
    """
    pinned = is_pinned(cm_x, r_zero)
    if pinned:
        return RegimeResult(regime=Regime.PINNING, slope=None, is_pinned=True, window=None)

    window = find_slope_window(accel_result)
    if window is None:
        return RegimeResult(regime=Regime.UNKNOWN, slope=None, is_pinned=False, window=None)

    start, end = window
    fit = fit_trend(accel_result.iteration[start : end + 1], accel_result.ca[start : end + 1])
    if fit is None:
        return RegimeResult(regime=Regime.UNKNOWN, slope=None, is_pinned=False, window=window)

    if not fit.is_significant:
        regime = Regime.STEADY
        robust = True
    elif not _sign_is_stable(accel_result, window, fit.slope):
        regime = Regime.UNKNOWN
        robust = False
    else:
        regime = Regime.DISSIPATIVE if fit.slope < 0 else Regime.CAPILLARY
        robust = True

    return RegimeResult(
        regime=regime,
        slope=fit.slope,
        is_pinned=False,
        window=window,
        drift=fit.drift,
        t_statistic=fit.t_statistic,
        is_robust=robust,
    )
