"""Shared helpers for snapshot-history analysis plots.

Plays the same role for ``AnalysisPlot`` subclasses that ``base.py`` plays
for the package as a whole: a common foundation that the per-plot-type
modules (``scalar_history_plot.py``, ``contact_angle_plot.py``,
``contact_line_speed_plot.py``) build on.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from src.simulation_io.analysis.droplet_metrics import extract_velocity_components_2d
from src.simulation_io.analysis.droplet_metrics import parse_timestep
from src.simulation_io.analysis.droplet_metrics import series_for_files
from src.simulation_io.plotting.base import AnalysisPlot
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    import matplotlib.axes
    from src.config import SimulationConfig
    from src.simulation_io.analysis.droplet_metrics import DropletSeries

_X_LABEL_TIMESTEP = "Timestep"
_EMPTY_DATA_TEXT = "No data"
_REQUIRES_PREFIX = "(Requires: "
_CONTACT_ANGLE_Y_LABEL = "Contact angle (deg)"
_CONTACT_LINE_SPEED_Y_LABEL = "d(cll)/dt"
_COLOR_TAB_BLUE = DEFAULT_STYLE.colors["max_velocity"]


def _empty_data_message(required_keys: tuple[str, ...] | None = None) -> str:
    msg = _EMPTY_DATA_TEXT
    if required_keys:
        msg += f"\n{_REQUIRES_PREFIX}{', '.join(required_keys)})"
    return msg


def _set_empty_state(
    ax: matplotlib.axes.Axes,
    *,
    title: str,
    ylabel: str,
    required_keys: tuple[str, ...] | None = None,
) -> None:
    ax.text(
        0.5,
        0.5,
        _empty_data_message(required_keys),
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=DEFAULT_STYLE.empty_state_fontsize,
    )
    ax.set_title(title)
    ax.set_xlabel(_X_LABEL_TIMESTEP)
    ax.set_ylabel(ylabel)


def load_snapshot(path: Path) -> dict[str, np.ndarray]:
    """Read every array in a ``.npz`` snapshot into a plain dict."""
    with np.load(path) as raw:
        return {key: np.asarray(raw[key]) for key in raw.files}


def _extract_u_mag_2d(u: np.ndarray) -> np.ndarray:
    """Velocity magnitude over the 2-D slice of *u*."""
    u_x, u_y = extract_velocity_components_2d(u)
    return np.hypot(u_x, u_y)


def _reduce_timesteps(
    files: list[Path],
    required: tuple[str, ...],
    reduce_fn: Callable[[dict[str, np.ndarray]], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Stream *files*, reducing each snapshot to one scalar via *reduce_fn*.

    The reduction runs per snapshot so only one snapshot's fields are resident
    at a time — the whole-run field arrays are never accumulated.
    """
    iters: list[int] = []
    values: list[float] = []
    for fp in files:
        step = parse_timestep(fp.stem)
        if step is None:
            continue
        with np.load(fp) as raw:
            if not all(key in raw for key in required):
                continue
            values.append(reduce_fn({key: np.asarray(raw[key]) for key in required}))
            iters.append(step)
    return np.asarray(iters, dtype=int), np.asarray(values, dtype=float)


def _droplet_series(config: SimulationConfig | None, files: list[Path]) -> DropletSeries | None:
    """Shared droplet series for *files*, or ``None`` when unavailable.

    Operators derive contact angles and contact-line positions from the density
    field rather than from stored ``.npz`` keys, so panels render even for runs
    that did not save those scalars.
    """
    if config is None:
        return None
    return series_for_files(files, config)


def _empty_series_arrays(*keys: str) -> dict[str, np.ndarray]:
    """Empty arrays under *keys*, for operators with no computable series."""
    return {key: np.asarray([], dtype=float) for key in keys}


def _render_scatter(
    ax: matplotlib.axes.Axes,
    iters: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    ylabel: str,
    color: str = _COLOR_TAB_BLUE,
    ylog: bool = False,
    required_keys: tuple[str, ...] | None = None,
) -> None:
    ax.clear()
    if len(iters) == 0:
        _set_empty_state(ax, title=title, ylabel=ylabel, required_keys=required_keys)
        return
    ax.scatter(
        iters,
        values,
        s=DEFAULT_STYLE.scatter_marker_size,
        alpha=DEFAULT_STYLE.scatter_alpha,
        color=color,
        edgecolors="none",
    )
    ax.set_xlabel(_X_LABEL_TIMESTEP)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)
    if ylog:
        ax.set_yscale("log")


class _BaseAnalysisPlot(AnalysisPlot):
    title: str
    ylabel: str
    color: str = _COLOR_TAB_BLUE
    ylog: bool = False
    required_keys: tuple[str, ...] = ()

    def render(self, ax: matplotlib.axes.Axes, precomputed: dict[str, np.ndarray]) -> None:
        _render_scatter(
            ax,
            precomputed["iters"],
            precomputed["values"],
            title=self.title,
            ylabel=self.ylabel,
            color=self.color,
            ylog=self.ylog,
            required_keys=self.required_keys,
        )


class _SeriesAttrPlot(_BaseAnalysisPlot):
    """One :class:`DropletSeries` attribute plotted against iteration."""

    #: Attribute on :class:`DropletSeries` holding the plotted values.
    series_attr: str

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """The configured series attribute per snapshot, from the shared series."""
        series = _droplet_series(self.config, files)
        if series is None:
            return _empty_series_arrays("iters", "values")
        return {"iters": series.iteration, "values": getattr(series, self.series_attr)}


class _PairSeriesPlot(AnalysisPlot):
    """Left/right pair of :class:`DropletSeries` attributes on shared axes."""

    title: str
    ylabel: str
    left_attr: str
    right_attr: str
    left_color: str
    right_color: str
    legend_kwargs: dict[str, object] = {}  # noqa: RUF012
    required_keys: tuple[str, ...] = ()

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Both sides' values per snapshot, from the shared series."""
        series = _droplet_series(self.config, files)
        if series is None:
            return _empty_series_arrays("iters", "left", "right")
        return {
            "iters": series.iteration,
            "left": getattr(series, self.left_attr),
            "right": getattr(series, self.right_attr),
        }

    def render(self, ax: matplotlib.axes.Axes, precomputed: dict[str, np.ndarray]) -> None:
        """Draw both sides as labelled scatter series."""
        ax.clear()
        iters = precomputed["iters"]
        if len(iters) == 0:
            _set_empty_state(ax, title=self.title, ylabel=self.ylabel, required_keys=self.required_keys)
            return
        for key, color, label in (
            ("left", self.left_color, "Left"),
            ("right", self.right_color, "Right"),
        ):
            ax.scatter(
                iters,
                precomputed[key],
                s=DEFAULT_STYLE.scatter_marker_size,
                color=color,
                alpha=DEFAULT_STYLE.scatter_alpha,
                edgecolors="none",
                label=label,
            )
        ax.set_title(self.title)
        ax.set_xlabel(_X_LABEL_TIMESTEP)
        ax.set_ylabel(self.ylabel)
        ax.grid(False)
        ax.legend(**self.legend_kwargs)
