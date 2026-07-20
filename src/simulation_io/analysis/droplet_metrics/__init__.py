"""Shared droplet metric computation for saved simulation snapshots.

Invariant: :func:`compute_droplet_series` is the only place that reads ``.npz``
snapshots for droplet metrics, and
:func:`src.simulation_io.plotting.simulation_csv.build_simulation_csv` is the only
place that decides whether a run gets a ``simulation_data.csv``.
"""

from src.simulation_io.analysis.droplet_metrics._scales import MetricScales
from src.simulation_io.analysis.droplet_metrics._scales import RZero
from src.simulation_io.analysis.droplet_metrics._scales import analytical_sigma_lg
from src.simulation_io.analysis.droplet_metrics._scales import inclination_angle_deg
from src.simulation_io.analysis.droplet_metrics._scales import measured_sigma_lg
from src.simulation_io.analysis.droplet_metrics._scales import resolve_r_zero
from src.simulation_io.analysis.droplet_metrics._scales import resolve_scales
from src.simulation_io.analysis.droplet_metrics._scales import resolve_step_x
from src.simulation_io.analysis.droplet_metrics.series import CSV_COLUMNS
from src.simulation_io.analysis.droplet_metrics.series import DropletSeries
from src.simulation_io.analysis.droplet_metrics.series import backward_diff
from src.simulation_io.analysis.droplet_metrics.series import clear_series_cache
from src.simulation_io.analysis.droplet_metrics.series import compute_droplet_series
from src.simulation_io.analysis.droplet_metrics.series import droplet_series_for_run
from src.simulation_io.analysis.droplet_metrics.series import series_for_files

__all__ = [
    "CSV_COLUMNS",
    "DropletSeries",
    "MetricScales",
    "RZero",
    "analytical_sigma_lg",
    "backward_diff",
    "clear_series_cache",
    "compute_droplet_series",
    "droplet_series_for_run",
    "inclination_angle_deg",
    "measured_sigma_lg",
    "resolve_r_zero",
    "resolve_scales",
    "resolve_step_x",
    "series_for_files",
]
