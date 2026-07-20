"""Scalar-history analysis plots: a single value vs. timestep."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from src.registry import analysis_operator
from src.simulation_io.plotting._analysis_common import _BaseAnalysisPlot
from src.simulation_io.plotting._analysis_common import _extract_rho_2d
from src.simulation_io.plotting._analysis_common import _extract_u_mag_2d
from src.simulation_io.plotting._analysis_common import _load_timesteps
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE

if TYPE_CHECKING:
    from pathlib import Path


@analysis_operator(name="max_velocity")
class MaxVelocityPlot(_BaseAnalysisPlot):
    """Plot maximum velocity magnitude over time."""

    name = "max_velocity"
    title = "Maximum velocity vs timestep"
    ylabel = "max(|u|)"
    color = DEFAULT_STYLE.colors["max_velocity"]
    required_keys = ("u",)

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute maximum velocity values for each timestep file."""
        iters, snapshots = _load_timesteps(files, ("u",))
        vals = np.asarray([float(np.max(_extract_u_mag_2d(snap["u"]))) for snap in snapshots], dtype=float)
        return {"iters": iters, "values": vals}


@analysis_operator(name="density_ratio")
class DensityRatioPlot(_BaseAnalysisPlot):
    """Plot max/min density ratio over time."""

    name = "density_ratio"
    title = "Density ratio vs timestep"
    ylabel = "max(rho) / min(rho)"
    color = DEFAULT_STYLE.colors["density_ratio"]
    ylog = True
    required_keys = ("rho",)

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute density ratio values for each timestep file."""
        iters, snapshots = _load_timesteps(files, ("rho",))
        vals = []
        for snap in snapshots:
            rho = _extract_rho_2d(snap["rho"])
            min_rho = float(np.min(rho))
            safe_min = min_rho if min_rho > 0 else max(min_rho, 1e-30)
            vals.append(float(np.max(rho)) / safe_min if safe_min != 0 else np.inf)
        return {"iters": iters, "values": np.asarray(vals, dtype=float)}


@analysis_operator(name="avg_density")
class AvgDensityPlot(_BaseAnalysisPlot):
    """Plot average density over time."""

    name = "avg_density"
    title = "Average density vs timestep"
    ylabel = "mean(rho)"
    color = DEFAULT_STYLE.colors["avg_density"]
    required_keys = ("rho",)

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute average density values for each timestep file."""
        iters, snapshots = _load_timesteps(files, ("rho",))
        vals = np.asarray([float(np.mean(_extract_rho_2d(snap["rho"]))) for snap in snapshots], dtype=float)
        return {"iters": iters, "values": vals}


@analysis_operator(name="total_mass")
class TotalMassPlot(_BaseAnalysisPlot):
    """Plot total domain mass (sum of rho) over time."""

    name = "total_mass"
    title = "Total mass vs timestep"
    ylabel = "sum(rho)"
    color = DEFAULT_STYLE.colors["total_mass"]
    required_keys = ("rho",)

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute total mass values for each timestep file."""
        iters, snapshots = _load_timesteps(files, ("rho",))
        vals = np.asarray([float(np.sum(_extract_rho_2d(snap["rho"]))) for snap in snapshots], dtype=float)
        return {"iters": iters, "values": vals}
