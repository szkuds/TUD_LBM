"""Interface contour plots: an overlay, a standalone panel and an evolution figure.

All three draw the ``rho == level`` contour for the interface markers of
:mod:`src.simulation_io.analysis.interface_contour` (``config`` and
``measured``), so the markers can be compared on one figure.

``interface`` (plotting operator)
    One operator in two roles. Named in ``plot_fields`` it renders its own
    panel; named in ``overlay_fields`` its :meth:`draw_overlay` draws only the
    contour on top of every other field panel (density, velocity, ...). It is
    opt-in, so it never joins the default figure uninvited.

``interface_evolution_config`` / ``interface_evolution_measured`` (analysis)
    Every snapshot's contour at one marker, coloured by timestep. One figure per
    marker, because many timesteps times two markers in one axes is unreadable.

Contours are a spatial map, like the ``imshow`` field panels, so they are
exempt from the scatter-only rule for data-series plots.
"""

from __future__ import annotations
from itertools import pairwise
from typing import TYPE_CHECKING
import numpy as np
from matplotlib.collections import LineCollection
from src.registry import analysis_operator
from src.registry import plotting_operator
from src.simulation_io.analysis.droplet_metrics import extract_rho_2d
from src.simulation_io.analysis.droplet_metrics import parse_timestep
from src.simulation_io.analysis.interface_contour import LEVEL_CONFIG
from src.simulation_io.analysis.interface_contour import LEVEL_MEASURED
from src.simulation_io.analysis.interface_contour import interface_lines
from src.simulation_io.analysis.interface_contour import level_value
from src.simulation_io.analysis.interface_contour import resolve_interface_levels
from src.simulation_io.plotting._analysis_common import _set_empty_state
from src.simulation_io.plotting.base import AnalysisPlot
from src.simulation_io.plotting.base import PlotOperator
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE

if TYPE_CHECKING:
    from pathlib import Path
    import matplotlib.axes
    from src.config import SimulationConfig

_AXIS_X = "x"
_AXIS_Y = "y"


def _domain_limits(config: SimulationConfig) -> tuple[tuple[float, float], tuple[float, float]]:
    """The ``(xlim, ylim)`` an ``imshow(rho.T, origin="lower")`` panel spans."""
    nx, ny = config.grid_shape[0], config.grid_shape[1]
    return (-0.5, nx - 0.5), (-0.5, ny - 0.5)


def _set_domain_axes(ax: matplotlib.axes.Axes, config: SimulationConfig) -> None:
    """Match a blank axes to the extent and aspect of the field panels."""
    xlim, ylim = _domain_limits(config)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel(_AXIS_X)
    ax.set_ylabel(_AXIS_Y)


def _draw_interface(
    ax: matplotlib.axes.Axes,
    data: dict[str, np.ndarray],
    config: SimulationConfig,
    levels: tuple[str, ...],
) -> None:
    """Draw one styled contour per interface marker, with a legend naming its density.

    Autoscaling is switched off first so the host panel keeps its extent. A
    marker this snapshot cannot supply (no config densities, a uniform field) is
    skipped rather than drawn at a meaningless level.
    """
    ax.autoscale(enable=False)
    rho_2d = extract_rho_2d(data["rho"])
    drawn = False
    for name in levels:
        value = level_value(name, config, rho_2d)
        if value is None:
            continue
        color, linestyle = DEFAULT_STYLE.interface_level_styles[name]
        collection = LineCollection(
            interface_lines(rho_2d, value),
            colors=color,
            linestyles=linestyle,
            linewidths=DEFAULT_STYLE.interface_linewidth,
            label=f"{name} ρ={value:.4g}",
        )
        ax.add_collection(collection)
        drawn = True
    if drawn:
        ax.legend(loc="upper right", fontsize=DEFAULT_STYLE.panel_legend_fontsize)


@plotting_operator(name="interface")
class InterfacePlotOperator(PlotOperator):
    """The interface contour, as a standalone panel or as an overlay on any field panel."""

    name = "interface"
    opt_in = True
    supports_overlay = True
    accepts_overlays = False

    def __init__(self, config: SimulationConfig, data_dir: str | Path | None = None) -> None:
        """Validate the configured interface markers up front, not inside a panel."""
        super().__init__(config, data_dir=data_dir)
        self.levels = resolve_interface_levels(config)

    def is_available(self, data: dict[str, np.ndarray]) -> bool:
        """A contour needs a density field."""
        return "rho" in data

    def __call__(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        timestep: int,
    ) -> None:
        """Render the interface contour on its own panel."""
        _set_domain_axes(ax, self.config)
        _draw_interface(ax, data, self.config, self.levels)
        ax.set_title(f"Interface  t={timestep}")

    def draw_overlay(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        timestep: int,  # noqa: ARG002
    ) -> None:
        """Draw only the contour on top of another field panel."""
        _draw_interface(ax, data, self.config, self.levels)


class _InterfaceEvolutionPlot(AnalysisPlot):
    """Every snapshot's contour at one interface marker, coloured by timestep."""

    level: str

    @property
    def title(self) -> str:
        """Figure title naming the marker."""
        return f"Interface evolution ({self.level} level)"

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Stream the snapshots into flat contour arrays.

        One snapshot's field is resident at a time; only its contour vertices are
        kept. Lines are packed flat — ``vertices`` split at ``line_offsets``, and
        ``line_snapshot`` indexing into ``timesteps``/``levels`` — so the result
        stays a plain ``dict[str, ndarray]``.
        """
        timesteps: list[int] = []
        levels: list[float] = []
        lines: list[np.ndarray] = []
        line_snapshot: list[int] = []
        for fp in files:
            step = parse_timestep(fp.stem)
            if step is None or self.config is None:
                continue
            with np.load(fp) as raw:
                if "rho" not in raw:
                    continue
                rho_2d = extract_rho_2d(np.asarray(raw["rho"]))
            value = level_value(self.level, self.config, rho_2d)
            if value is None:
                continue
            index = len(timesteps)
            timesteps.append(step)
            levels.append(value)
            for line in interface_lines(rho_2d, value):
                lines.append(line)
                line_snapshot.append(index)

        offsets = np.concatenate(([0], np.cumsum([len(line) for line in lines], dtype=int)))
        return {
            "timesteps": np.asarray(timesteps, dtype=int),
            "levels": np.asarray(levels, dtype=float),
            "vertices": np.concatenate(lines) if lines else np.empty((0, 2)),
            "line_offsets": offsets.astype(int),
            "line_snapshot": np.asarray(line_snapshot, dtype=int),
        }

    def render(self, ax: matplotlib.axes.Axes, precomputed: dict[str, np.ndarray]) -> None:
        """Draw every contour, coloured by timestep, on the domain's axes."""
        ax.clear()
        timesteps = precomputed["timesteps"]
        if timesteps.size == 0 or self.config is None:
            _set_empty_state(ax, title=self.title, ylabel=_AXIS_Y, required_keys=("rho", self.level))
            return

        offsets = precomputed["line_offsets"]
        vertices = precomputed["vertices"]
        segments = [vertices[start:stop] for start, stop in pairwise(offsets)]
        collection = LineCollection(
            segments,
            cmap=DEFAULT_STYLE.colormap_interface_time,
            linewidths=DEFAULT_STYLE.interface_linewidth,
        )
        collection.set_array(timesteps[precomputed["line_snapshot"]])
        collection.set_clim(float(timesteps.min()), float(timesteps.max()))
        ax.add_collection(collection)
        ax.figure.colorbar(collection, ax=ax, fraction=0.046, pad=0.04, label="Timestep")

        _set_domain_axes(ax, self.config)
        levels = precomputed["levels"]
        lo, hi = float(levels.min()), float(levels.max())
        span = f"ρ={lo:.4g}" if np.isclose(lo, hi) else f"ρ={lo:.4g}–{hi:.4g}"
        ax.set_title(f"{self.title}  {span}")


@analysis_operator(name="interface_evolution_config")
class InterfaceEvolutionConfigPlot(_InterfaceEvolutionPlot):
    """Interface evolution at the config ``(rho_l + rho_v) / 2``."""

    name = "interface_evolution_config"
    level = LEVEL_CONFIG


@analysis_operator(name="interface_evolution_measured")
class InterfaceEvolutionMeasuredPlot(_InterfaceEvolutionPlot):
    """Interface evolution at each snapshot's measured bulk-phase midpoint."""

    name = "interface_evolution_measured"
    level = LEVEL_MEASURED
