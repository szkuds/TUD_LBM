"""Tests for tud_lbm.io.analysis.accelerations.acceleration_analysis."""

from __future__ import annotations
import numpy as np
import pandas as pd
from tud_lbm.io.analysis.accelerations import compute_acceleration
from tud_lbm.io.analysis.accelerations import find_slope_window
from tud_lbm.io.analysis.accelerations import save_diagnostic_plot


def _ramp_up_then_down_df(n: int = 20) -> pd.DataFrame:
    """Build a Ca(t) curve via double-integration of a known accel impulse pair.

    accel_true has a single positive impulse at index 2 (peak acceleration)
    followed by a single negative impulse at index 12 (peak deceleration),
    so the recovered peak indices are known analytically: integration and
    second-differencing are exact inverses for this construction.
    """
    accel_true = np.zeros(n)
    accel_true[2] = 3.0
    accel_true[12] = -4.0
    vel = np.concatenate([[0.0], np.cumsum(accel_true)])
    ca = np.concatenate([[0.0], np.cumsum(vel)])
    return pd.DataFrame({"normalised_iteration": np.arange(ca.size), "Ca": ca})


def test_compute_acceleration_recovers_known_peak_indices():
    df = _ramp_up_then_down_df()

    result = compute_acceleration(df)

    assert result.has_peak_pair
    assert result.peak_accel_idx == 4
    assert result.peak_decel_idx == 14
    assert np.isnan(result.accel[0])
    assert np.isnan(result.accel[1])


def test_compute_acceleration_no_finite_tail_after_peak_has_no_pair():
    df = pd.DataFrame({"normalised_iteration": [0, 1, 2], "Ca": [0.0, 0.0, 5.0]})

    result = compute_acceleration(df)

    assert result.peak_accel_idx == 2
    assert result.peak_decel_idx is None
    assert not result.has_peak_pair


def test_find_slope_window_returns_window_with_default_margin():
    df = _ramp_up_then_down_df()
    result = compute_acceleration(df)

    window = find_slope_window(result)

    assert window == (7, 11)


def test_find_slope_window_none_when_window_too_narrow():
    df = _ramp_up_then_down_df()
    result = compute_acceleration(df)

    window = find_slope_window(result, margin=8)

    assert window is None


def test_find_slope_window_none_when_no_peak_pair():
    df = pd.DataFrame({"normalised_iteration": [0, 1, 2], "Ca": [0.0, 0.0, 5.0]})
    result = compute_acceleration(df)

    assert find_slope_window(result) is None


def test_save_diagnostic_plot_writes_file_with_window(tmp_path):
    df = _ramp_up_then_down_df()
    result = compute_acceleration(df)
    window = find_slope_window(result)

    out_path = save_diagnostic_plot(result, window, tmp_path / "acceleration_analysis.png")

    assert out_path.exists()


def test_save_diagnostic_plot_writes_file_without_window(tmp_path):
    df = pd.DataFrame({"normalised_iteration": [0, 1, 2], "Ca": [0.0, 0.0, 5.0]})
    result = compute_acceleration(df)

    out_path = save_diagnostic_plot(result, None, tmp_path / "acceleration_analysis.png")

    assert out_path.exists()


def test_compute_acceleration_raw_is_default_and_unchanged():
    df = _ramp_up_then_down_df()

    raw_default = compute_acceleration(df)
    raw_explicit = compute_acceleration(df, smoothing="raw")

    np.testing.assert_array_equal(raw_default.accel, raw_explicit.accel)


def test_compute_acceleration_savgol_smooths_noisy_accel():
    rng = np.random.default_rng(0)
    n = 40
    accel_true = np.zeros(n)
    accel_true[5] = 3.0
    accel_true[30] = -4.0
    vel = np.concatenate([[0.0], np.cumsum(accel_true)])
    ca = np.concatenate([[0.0], np.cumsum(vel)])
    ca = ca + rng.normal(scale=0.05, size=ca.size)
    df = pd.DataFrame({"normalised_iteration": np.arange(ca.size), "Ca": ca})

    raw_result = compute_acceleration(df, smoothing="raw")
    smoothed_result = compute_acceleration(df, smoothing="savgol")

    raw_variation = np.nansum(np.abs(np.diff(raw_result.accel)))
    smoothed_variation = np.nansum(np.abs(np.diff(smoothed_result.accel)))
    assert smoothed_variation < raw_variation


def test_compute_acceleration_savgol_falls_back_to_raw_when_too_few_points():
    df = pd.DataFrame({"normalised_iteration": [0, 1, 2], "Ca": [0.0, 0.0, 5.0]})

    raw_result = compute_acceleration(df, smoothing="raw")
    savgol_result = compute_acceleration(df, smoothing="savgol", savgol_window=5)

    np.testing.assert_array_equal(raw_result.accel, savgol_result.accel)
