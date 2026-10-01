"""Numerical surface-tension calibration for EOS without a closed form.

Equations of state without a closed-form surface tension expression
(Carnahan-Starling) need it measured numerically instead.

The measurement is a sweep of ordinary runs: :func:`calibration_configs`
produces them, :func:`collect_calibration` fits their results into the cache,
and :func:`cached_surface_tension` is how every reader gets the value back.

Public API::

    from src.simulation_io.analysis.surface_tension import cached_surface_tension
    from src.simulation_io.analysis.surface_tension import record_surface_tension
"""

from src.simulation_io.analysis.surface_tension.surface_tension import cached_surface_tension
from src.simulation_io.analysis.surface_tension.surface_tension import calibrate_surface_tension
from src.simulation_io.analysis.surface_tension.surface_tension import calibrated_digests
from src.simulation_io.analysis.surface_tension.surface_tension import calibration_configs
from src.simulation_io.analysis.surface_tension.surface_tension import collect_calibration
from src.simulation_io.analysis.surface_tension.surface_tension import find_sweep_runs
from src.simulation_io.analysis.surface_tension.surface_tension import is_calibrated
from src.simulation_io.analysis.surface_tension.surface_tension import needs_calibration
from src.simulation_io.analysis.surface_tension.surface_tension import record_surface_tension
from src.simulation_io.analysis.surface_tension.surface_tension import surface_tension_dir

__all__ = [
    "cached_surface_tension",
    "calibrate_surface_tension",
    "calibrated_digests",
    "calibration_configs",
    "collect_calibration",
    "find_sweep_runs",
    "is_calibrated",
    "needs_calibration",
    "record_surface_tension",
    "surface_tension_dir",
]
