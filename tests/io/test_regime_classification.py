"""Tests for src.simulation_io.analysis.accelerations.regime_classification."""

from __future__ import annotations
import numpy as np
from src.simulation_io.analysis.accelerations import AccelerationResult
from src.simulation_io.analysis.accelerations import classify_regime
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
    ca = 10.0 - iteration  # strictly decreasing everywhere, including the window
    accel_result = _accel_result(iteration, ca, peak_accel_idx=4, peak_decel_idx=24)

    result = classify_regime(cm_x, r_zero=10.0, accel_result=accel_result)

    assert result.regime == "Dissipative"
    assert result.slope is not None
    assert result.slope < 0
    assert result.window == (16, 20)


def test_classify_regime_inertial_when_slope_nonnegative_in_window():
    cm_x = np.array([0.0, 5.0, 30.0])
    iteration = np.arange(30, dtype=float)
    ca = iteration.copy()  # strictly increasing everywhere, including the window
    accel_result = _accel_result(iteration, ca, peak_accel_idx=4, peak_decel_idx=24)

    result = classify_regime(cm_x, r_zero=10.0, accel_result=accel_result)

    assert result.regime == "Inertial"
    assert result.slope is not None
    assert result.slope >= 0
    assert result.window == (16, 20)


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
