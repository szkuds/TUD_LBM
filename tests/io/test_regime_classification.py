"""Tests for src.simulation_io.analysis.accelerations.regime_classification."""

from __future__ import annotations
import numpy as np
import pytest
from src.simulation_io.analysis.accelerations import AccelerationResult
from src.simulation_io.analysis.accelerations import classify_regime
from src.simulation_io.analysis.accelerations import fit_trend
from src.simulation_io.analysis.accelerations import is_pinned


def _accel_result(
    iteration: np.ndarray, ca: np.ndarray, peak_accel_idx: int, peak_decel_idx: int
) -> AccelerationResult:
    accel = np.full_like(ca, np.nan)
    return AccelerationResult(
        iteration=iteration,
        ca=ca,
        accel=accel,
        peak_accel_idx=peak_accel_idx,
        peak_decel_idx=peak_decel_idx,
        has_peak_pair=True,
    )


def test_is_pinned_true_when_excursion_below_half_r0():
    cm_x = np.array([10.0, 10.2, 9.9, 10.1])
    assert is_pinned(cm_x, r_zero=10.0)


def test_is_pinned_false_when_excursion_above_half_r0():
    cm_x = np.array([10.0, 20.0, 10.0])
    assert not is_pinned(cm_x, r_zero=10.0)


def test_classify_regime_pinning_takes_priority_over_acceleration_data():
    cm_x = np.array([10.0, 10.1, 9.9])
    iteration = np.arange(20, dtype=float)
    ca = iteration.copy()
    accel_result = _accel_result(iteration, ca, peak_accel_idx=4, peak_decel_idx=14)

    result = classify_regime(cm_x, r_zero=10.0, accel_result=accel_result)

    assert result.regime == "Pinning"
    assert result.is_pinned
    assert result.slope is None
    assert result.window is None


def test_classify_regime_viscous_when_slope_negative_in_window():
    cm_x = np.array([0.0, 5.0, 30.0])
    iteration = np.arange(30, dtype=float)
    ca = 30.0 - iteration  # strictly decreasing everywhere, and positive: Ca is a magnitude
    accel_result = _accel_result(iteration, ca, peak_accel_idx=4, peak_decel_idx=24)

    result = classify_regime(cm_x, r_zero=10.0, accel_result=accel_result)

    assert result.regime == "Dissipative"
    assert result.slope is not None
    assert result.slope < 0
    assert result.window == (16, 20)


def test_classify_regime_capillary_when_slope_nonnegative_in_window():
    cm_x = np.array([0.0, 5.0, 30.0])
    iteration = np.arange(30, dtype=float)
    ca = iteration.copy()  # strictly increasing everywhere, including the window
    accel_result = _accel_result(iteration, ca, peak_accel_idx=4, peak_decel_idx=24)

    result = classify_regime(cm_x, r_zero=10.0, accel_result=accel_result)

    assert result.regime == "Capillary"
    assert result.slope is not None
    assert result.slope >= 0
    assert result.window == (16, 20)


def test_classify_regime_steady_when_plateau_trend_is_within_the_noise():
    """A flat plateau is named, not forced onto one side of a zero slope.

    Chemical-step runs reach a terminal velocity on the step, so the window
    lands on a plateau where the fitted slope is set by scatter. This is the
    case that used to split physically identical runs between Dissipative and
    Capillary depending on which side of zero the noise fell.
    """
    cm_x = np.array([0.0, 5.0, 30.0])
    iteration = np.arange(40, dtype=float)
    rng = np.random.default_rng(0)
    ca = 0.14 + 0.001 * rng.standard_normal(iteration.size)
    accel_result = _accel_result(iteration, ca, peak_accel_idx=0, peak_decel_idx=39)

    result = classify_regime(cm_x, r_zero=10.0, accel_result=accel_result)

    assert result.regime == "Steady"
    assert result.slope is not None  # the fit is still reported
    assert result.drift is not None
    assert abs(result.drift) < 0.05
    assert result.is_robust


def test_classify_regime_unknown_when_significant_sign_flips_with_the_window():
    """A trend that only exists because of where the window starts is not trusted."""
    cm_x = np.array([0.0, 5.0, 30.0])
    iteration = np.arange(40, dtype=float)
    # A jump followed by a gentle decline. The window is [12, 35]; shifting its
    # start back to 8 pulls in pre-jump samples and makes the fit significantly
    # positive, while starting at 12 or 16 sees only the decline and fits
    # significantly negative.
    ca = np.concatenate([np.zeros(11), np.linspace(1.0, 0.90, 29)])
    accel_result = _accel_result(iteration, ca, peak_accel_idx=0, peak_decel_idx=39)

    result = classify_regime(cm_x, r_zero=10.0, accel_result=accel_result)

    assert result.regime == "unknown"
    assert not result.is_robust


def test_classify_regime_unknown_when_window_is_too_short_to_fit():
    """A window that a line fits exactly has no residual to test the slope against."""
    cm_x = np.array([0.0, 5.0, 30.0])
    iteration = np.arange(20, dtype=float)
    ca = iteration.copy()
    accel_result = _accel_result(iteration, ca, peak_accel_idx=0, peak_decel_idx=15)

    result = classify_regime(cm_x, r_zero=10.0, accel_result=accel_result)

    assert result.regime == "unknown"
    assert result.window is None


def test_fit_trend_none_for_fewer_than_three_points():
    assert fit_trend(np.array([0.0, 1.0]), np.array([1.0, 2.0])) is None


def test_fit_trend_reports_drift_relative_to_the_mean():
    x = np.arange(11, dtype=float)
    y = 1.0 + 0.1 * x  # mean 1.5, total rise 1.0
    fit = fit_trend(x, y)

    assert fit is not None
    assert fit.slope == pytest.approx(0.1)
    assert fit.drift == pytest.approx(1.0 / 1.5)
    assert fit.is_significant


def test_fit_trend_normalises_by_the_magnitude_of_the_mean():
    """``Ca`` is signed: a run driven in -x must classify like its mirror image.

    ``Ca = avg_u_x * nu / sigma`` and ``avg_u_x`` is the inclusion's signed mean
    x-velocity, so a negative-``Ca`` run is a frame choice, not a degenerate fit.
    Its drift must have the same magnitude as the mirrored run and must follow
    its own slope, rather than being flipped by the sign of the mean.
    """
    x = np.arange(11, dtype=float)
    rising = fit_trend(x, 1.0 + 0.1 * x)
    mirrored = fit_trend(x, -(1.0 + 0.1 * x))

    assert rising is not None
    assert mirrored is not None
    assert mirrored.slope == pytest.approx(-rising.slope)
    assert mirrored.drift == pytest.approx(-rising.drift)
    assert abs(mirrored.drift) == pytest.approx(abs(rising.drift))
    assert mirrored.is_significant == rising.is_significant


def test_classify_regime_survives_a_negative_capillary_number_window():
    """A window of negative ``Ca`` classifies, rather than collapsing to unknown."""
    cm_x = np.array([0.0, 5.0, 30.0])
    iteration = np.arange(30, dtype=float)
    ca = -(30.0 - iteration)  # strictly increasing, negative throughout
    accel_result = _accel_result(iteration, ca, peak_accel_idx=4, peak_decel_idx=24)

    result = classify_regime(cm_x, r_zero=10.0, accel_result=accel_result)

    assert result.regime == "Capillary"
    assert result.slope is not None
    assert result.slope > 0


def test_fit_trend_none_when_the_mean_vanishes():
    x = np.arange(11, dtype=float)
    y = x - float(np.mean(x))  # symmetric about zero: no scale to normalise by

    assert fit_trend(x, y) is None


def test_classify_regime_unknown_when_no_usable_window():
    cm_x = np.array([0.0, 5.0, 30.0])
    iteration = np.arange(20, dtype=float)
    ca = iteration.copy()
    accel_result = AccelerationResult(
        iteration=iteration,
        ca=ca,
        accel=np.full_like(ca, np.nan),
        peak_accel_idx=4,
        peak_decel_idx=None,
        has_peak_pair=False,
    )

    result = classify_regime(cm_x, r_zero=10.0, accel_result=accel_result)

    assert result.regime == "unknown"
    assert result.slope is None
    assert not result.is_pinned
    assert result.window is None
