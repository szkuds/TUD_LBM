"""Acceleration-peak analysis of the Ca(t) curve for regime classification.

Computes the raw second backward difference of capillary number ``Ca`` over
normalised iteration, locates the peak-acceleration / peak-deceleration pair,
and derives the slope window used by regime classification (see
:mod:`tud_lbm.io.analysis.accelerations.regime_classification`).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal
import numpy as np
from scipy.signal import savgol_filter
from tud_lbm.io.plotting.figure_config import DEFAULT_STYLE

if TYPE_CHECKING:
    import pandas as pd

_SLOPE_WINDOW_MARGIN = 4
_MIN_SLOPE_WINDOW_POINTS = 2
_MIN_POINTS_FOR_SECOND_DIFFERENCE = 3
_DEFAULT_SAVGOL_WINDOW = 5
_DEFAULT_SAVGOL_POLYORDER = 2

_CA_COLOR = "tab:blue"
_ACCEL_COLOR = "tab:orange"

Smoothing = Literal["raw", "savgol"]


@dataclass(frozen=True)
class AccelerationResult:
    """Ca(t), its raw second difference, and the located peak pair."""

    iteration: np.ndarray
    ca: np.ndarray
    accel: np.ndarray
    peak_accel_idx: int | None
    peak_decel_idx: int | None
    has_peak_pair: bool


def compute_acceleration(
    df: pd.DataFrame,
    *,
    x_col: str = "normalised_iteration",
    smoothing: Smoothing = "raw",
    savgol_window: int = _DEFAULT_SAVGOL_WINDOW,
    savgol_polyorder: int = _DEFAULT_SAVGOL_POLYORDER,
) -> AccelerationResult:
    """Locate the peak-acceleration / peak-deceleration pair in ``df["Ca"]``.

    ``accel`` is the second backward difference of ``Ca``, NaN-padded at
    indices 0 and 1 so it aligns with ``iteration``/``ca``. By default
    (``smoothing="raw"``) it is unsmoothed. With ``smoothing="savgol"``, a
    Savitzky-Golay filter (``savgol_window``, ``savgol_polyorder``) is applied
    to the raw second difference to remove spurious spikes before peak
    detection; this falls back to the raw curve when there are fewer points
    than ``savgol_window``. Peak acceleration is ``argmax(accel)``; peak
    deceleration is ``argmin(accel)`` searched strictly after the
    peak-acceleration index.
    """
    iteration = df[x_col].to_numpy(dtype=float)
    ca = df["Ca"].to_numpy(dtype=float)

    accel = np.full_like(ca, np.nan)
    if ca.size >= _MIN_POINTS_FOR_SECOND_DIFFERENCE:
        accel[2:] = np.diff(np.diff(ca))
        if smoothing == "savgol" and accel[2:].size >= savgol_window:
            accel[2:] = savgol_filter(accel[2:], window_length=savgol_window, polyorder=savgol_polyorder)

    valid_accel = np.where(np.isnan(accel), -np.inf, accel)
    if not np.any(np.isfinite(valid_accel)):
        return AccelerationResult(iteration, ca, accel, None, None, has_peak_pair=False)

    peak_accel_idx = int(np.argmax(valid_accel))

    tail = valid_accel[peak_accel_idx + 1 :]
    if tail.size == 0 or not np.any(np.isfinite(tail)):
        return AccelerationResult(iteration, ca, accel, peak_accel_idx, None, has_peak_pair=False)

    peak_decel_idx = peak_accel_idx + 1 + int(np.argmin(tail))
    return AccelerationResult(iteration, ca, accel, peak_accel_idx, peak_decel_idx, has_peak_pair=True)


def find_slope_window(result: AccelerationResult, *, margin: int = _SLOPE_WINDOW_MARGIN) -> tuple[int, int] | None:
    """Return ``[peak_accel_idx+margin, peak_decel_idx-margin]``, or ``None``.

    ``None`` when no peak pair was found, or fewer than two indices remain in
    the window.
    """
    if not result.has_peak_pair or result.peak_accel_idx is None or result.peak_decel_idx is None:
        return None
    start = result.peak_accel_idx + 3 * margin
    end = result.peak_decel_idx - margin
    if end - start + 1 < _MIN_SLOPE_WINDOW_POINTS:
        return None
    return start, end


def save_diagnostic_plot(result: AccelerationResult, window: tuple[int, int] | None, out_path: str | Path) -> Path:
    """Save Ca(t) + twin-axis accel(t) with peak markers and the slope window."""
    import matplotlib.pyplot as plt

    fig, ax_ca = plt.subplots(figsize=DEFAULT_STYLE.analysis_figsize)
    ax_ca.plot(result.iteration, result.ca, color=_CA_COLOR, label="Ca")
    ax_ca.set_xlabel("Normalised iteration", fontsize=DEFAULT_STYLE.axis_label_fontsize)
    ax_ca.set_ylabel("Ca", color=_CA_COLOR, fontsize=DEFAULT_STYLE.axis_label_fontsize)
    ax_ca.tick_params(axis="y", labelcolor=_CA_COLOR)

    ax_accel = ax_ca.twinx()
    ax_accel.plot(result.iteration, result.accel, color=_ACCEL_COLOR, alpha=0.6, label="accel")
    ax_accel.set_ylabel("Acceleration", color=_ACCEL_COLOR, fontsize=DEFAULT_STYLE.axis_label_fontsize)
    ax_accel.tick_params(axis="y", labelcolor=_ACCEL_COLOR)

    if result.peak_accel_idx is not None:
        ax_accel.scatter(
            result.iteration[result.peak_accel_idx],
            result.accel[result.peak_accel_idx],
            color="tab:green",
            marker="^",
            zorder=5,
            label="peak accel",
        )
    if result.peak_decel_idx is not None:
        ax_accel.scatter(
            result.iteration[result.peak_decel_idx],
            result.accel[result.peak_decel_idx],
            color="tab:red",
            marker="v",
            zorder=5,
            label="peak decel",
        )

    if window is not None:
        ax_ca.axvspan(
            result.iteration[window[0]],
            result.iteration[window[1]],
            color="tab:gray",
            alpha=0.2,
            label="slope window",
        )
    else:
        ax_ca.text(
            0.5,
            0.95,
            "slope window unavailable",
            transform=ax_ca.transAxes,
            ha="center",
            va="top",
            fontsize=DEFAULT_STYLE.empty_state_fontsize,
            color="tab:red",
        )

    ax_ca.set_title("Ca(t) acceleration analysis", fontsize=DEFAULT_STYLE.title_fontsize)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_STYLE.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
