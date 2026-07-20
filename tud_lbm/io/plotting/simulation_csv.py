"""Per-timestep droplet-metric CSV export.

This module is purely a serializer: every metric it writes comes from
:func:`tud_lbm.io.analysis.droplet_metrics.droplet_series_for_run`. It is also
the only place that decides whether a run gets a ``simulation_data.csv``.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from tud_lbm.io.analysis.droplet_metrics import droplet_series_for_run
from tud_lbm.io.plotting._analysis_common import _set_empty_state
from tud_lbm.io.plotting.base import AnalysisPlot
from tud_lbm.registry import analysis_operator

if TYPE_CHECKING:
    import matplotlib.axes
    from tud_lbm.config import SimulationConfig
    from tud_lbm.io.analysis.droplet_metrics import MetricScales

_SUPPORTED_SIM_TYPES = {
    "multiphase_wetting",
    "multiphase_hysteresis",
    "multiphase_hysteresis_chemical_step",
}
_CSV_FILENAME = "simulation_data.csv"


def _warn_r_zero_fallback(config: SimulationConfig, scales: MetricScales) -> None:
    """Warn that R₀ came from the nominal radius rather than a measured length.

    Emitted here rather than in the metric layer so that plotting operators
    sharing the same series stay quiet.
    """
    print(
        f"WARNING: R_0 fell back to nominal radius ({scales.r_zero:.4g} lu); "
        f"init file not found (init_dir={config.init_dir!r}). "
        "avg_x_location_norm uses radii*min_dim, not measured L/2.",
        file=sys.stderr,
    )


def build_simulation_csv(run_dir: str | Path, config: SimulationConfig) -> Path | None:
    """Compute per-timestep metrics and write ``simulation_data.csv``.

    Skips silently (returns ``None``) when *config.sim_type* is not a supported
    wetting variant, or when the run has no usable snapshots.

    Args:
        run_dir: Run directory (contains ``data/timestep_*.npz``).
        config:  :class:`~tud_lbm.config.simulation_config.SimulationConfig`.

    Returns:
        Path to the written CSV, or ``None`` when skipped.
    """
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        print("pandas is required for CSV export. Install with: pip install pandas")
        return None

    if config.sim_type not in _SUPPORTED_SIM_TYPES:
        return None

    run_dir = Path(run_dir)
    series = droplet_series_for_run(run_dir, config)
    if series is None:
        return None

    if series.scales.r_zero_is_fallback:
        _warn_r_zero_fallback(config, series.scales)

    csv_path = run_dir / _CSV_FILENAME
    series.to_dataframe().to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    return csv_path


@analysis_operator(name="simulation_csv")
class SimulationCsvExport(AnalysisPlot):
    """Export per-timestep droplet metrics to ``simulation_data.csv``."""

    name = "simulation_csv"
    required_keys = ("rho", "u")
    export_only = True

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:  # noqa: ARG002
        """Return an empty payload because this operator is export-only."""
        return {
            "iters": np.asarray([], dtype=int),
            "values": np.asarray([], dtype=float),
        }

    def render(self, ax: matplotlib.axes.Axes, precomputed: dict[str, np.ndarray]) -> None:  # noqa: ARG002
        """Render a placeholder panel when this operator is selected for plotting."""
        _set_empty_state(
            ax,
            title="CSV export operator",
            ylabel="N/A",
            required_keys=self.required_keys,
        )

    def export(self, run_dir: Path) -> Path | None:
        """Write ``simulation_data.csv`` for the configured run directory."""
        if self.config is None:
            return None
        return build_simulation_csv(run_dir, self.config)
