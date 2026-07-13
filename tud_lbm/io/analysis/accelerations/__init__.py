"""Acceleration-based regime analysis for contact-line dynamics."""

from tud_lbm.io.analysis.accelerations.acceleration_analysis import AccelerationResult
from tud_lbm.io.analysis.accelerations.acceleration_analysis import Smoothing
from tud_lbm.io.analysis.accelerations.acceleration_analysis import compute_acceleration
from tud_lbm.io.analysis.accelerations.acceleration_analysis import find_slope_window
from tud_lbm.io.analysis.accelerations.acceleration_analysis import save_diagnostic_plot
from tud_lbm.io.analysis.accelerations.regime_classification import Regime
from tud_lbm.io.analysis.accelerations.regime_classification import RegimeResult
from tud_lbm.io.analysis.accelerations.regime_classification import classify_regime
from tud_lbm.io.analysis.accelerations.regime_classification import is_pinned

__all__ = [
    "AccelerationResult",
    "Regime",
    "RegimeResult",
    "Smoothing",
    "classify_regime",
    "compute_acceleration",
    "find_slope_window",
    "is_pinned",
    "save_diagnostic_plot",
]
