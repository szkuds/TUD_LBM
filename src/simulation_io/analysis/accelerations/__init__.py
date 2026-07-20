"""Acceleration-based regime analysis for contact-line dynamics."""

from src.simulation_io.analysis.accelerations.acceleration_analysis import AccelerationResult
from src.simulation_io.analysis.accelerations.acceleration_analysis import Smoothing
from src.simulation_io.analysis.accelerations.acceleration_analysis import compute_acceleration
from src.simulation_io.analysis.accelerations.acceleration_analysis import find_slope_window
from src.simulation_io.analysis.accelerations.acceleration_analysis import save_diagnostic_plot
from src.simulation_io.analysis.accelerations.regime_classification import Regime
from src.simulation_io.analysis.accelerations.regime_classification import RegimeResult
from src.simulation_io.analysis.accelerations.regime_classification import classify_regime
from src.simulation_io.analysis.accelerations.regime_classification import is_pinned

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
