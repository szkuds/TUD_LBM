"""Scalar-history analysis plots: a single value vs. timestep."""

from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from src.registry import analysis_operator
from src.simulation_io.analysis.droplet_metrics import extract_rho_2d
from src.simulation_io.plotting._analysis_common import _BaseAnalysisPlot
from src.simulation_io.plotting._analysis_common import _extract_u_mag_2d
from src.simulation_io.plotting._analysis_common import _reduce_timesteps
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE

if TYPE_CHECKING:
    from pathlib import Path

_MIN_POSITIVE_RHO = 1e-30


class _ScalarHistoryPlot(_BaseAnalysisPlot):
    """One scalar per snapshot, reduced while streaming the run's ``.npz`` files.

    Subclasses supply ``required_keys`` and :meth:`reduce`; the shared
    :meth:`compute` never holds more than one snapshot's fields in memory.
    """

    @abstractmethod
    def reduce(self, snapshot: dict[str, np.ndarray]) -> float:
        """Collapse one snapshot's required fields to the plotted scalar."""

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Reduce every parseable snapshot in *files* to one scalar."""
        iters, values = _reduce_timesteps(files, self.required_keys, self.reduce)
        return {"iters": iters, "values": values}


@analysis_operator(name="max_velocity")
class MaxVelocityPlot(_ScalarHistoryPlot):
    """Plot maximum velocity magnitude over time."""

    name = "max_velocity"
    title = "Maximum velocity vs timestep"
    ylabel = "max(|u|)"
    color = DEFAULT_STYLE.colors["max_velocity"]
    required_keys = ("u",)

    def reduce(self, snapshot: dict[str, np.ndarray]) -> float:
        """Maximum velocity magnitude in the snapshot."""
        return float(np.max(_extract_u_mag_2d(snapshot["u"])))


@analysis_operator(name="density_ratio")
class DensityRatioPlot(_ScalarHistoryPlot):
    """Plot max/min density ratio over time."""

    name = "density_ratio"
    title = "Density ratio vs timestep"
    ylabel = "max(rho) / min(rho)"
    color = DEFAULT_STYLE.colors["density_ratio"]
    ylog = True
    required_keys = ("rho",)

    def reduce(self, snapshot: dict[str, np.ndarray]) -> float:
        """Density ratio, guarding against a non-positive minimum."""
        rho = extract_rho_2d(snapshot["rho"])
        min_rho = float(np.min(rho))
        safe_min = min_rho if min_rho > 0 else max(min_rho, _MIN_POSITIVE_RHO)
        return float(np.max(rho)) / safe_min if safe_min != 0 else float(np.inf)


@analysis_operator(name="avg_density")
class AvgDensityPlot(_ScalarHistoryPlot):
    """Plot average density over time."""

    name = "avg_density"
    title = "Average density vs timestep"
    ylabel = "mean(rho)"
    color = DEFAULT_STYLE.colors["avg_density"]
    required_keys = ("rho",)

    def reduce(self, snapshot: dict[str, np.ndarray]) -> float:
        """Mean density over the snapshot."""
        return float(np.mean(extract_rho_2d(snapshot["rho"])))


@analysis_operator(name="total_mass")
class TotalMassPlot(_ScalarHistoryPlot):
    """Plot total domain mass (sum of rho) over time."""

    name = "total_mass"
    title = "Total mass vs timestep"
    ylabel = "sum(rho)"
    color = DEFAULT_STYLE.colors["total_mass"]
    required_keys = ("rho",)

    def reduce(self, snapshot: dict[str, np.ndarray]) -> float:
        """Total mass in the snapshot."""
        return float(np.sum(extract_rho_2d(snapshot["rho"])))
