"""Shared helpers for snapshot-history analysis plots.

Plays the same role for ``AnalysisPlot`` subclasses that ``base.py`` plays
for the package as a whole: a common foundation that the per-plot-type
modules (``scalar_history_plot.py``, ``contact_angle_plot.py``,
``contact_line_speed_plot.py``) build on.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from tud_lbm.io.analysis.droplet_metrics import series_for_files
from tud_lbm.io.plotting.base import AnalysisPlot
from tud_lbm.io.plotting.figure_config import DEFAULT_STYLE

if TYPE_CHECKING:
    from pathlib import Path
    import matplotlib.axes
    from tud_lbm.config import SimulationConfig
    from tud_lbm.io.analysis.droplet_metrics import DropletSeries

_NDIM_2D = 2
_NDIM_3D = 3
_NDIM_4D = 4
_NDIM_5D = 5

_X_LABEL_TIMESTEP = "Timestep"
_EMPTY_DATA_TEXT = "No data"
_REQUIRES_PREFIX = "(Requires: "
_CONTACT_ANGLE_Y_LABEL = "Contact angle (deg)"
_CONTACT_LINE_SPEED_Y_LABEL = "d(cll)/dt"
_COLOR_TAB_BLUE = DEFAULT_STYLE.colors["max_velocity"]
_COLOR_TAB_PURP = DEFAULT_STYLE.colors["contact_angle_left"]
_COLOR_TAB_RED = DEFAULT_STYLE.colors["contact_angle_right"]
_COLOR_TAB_ORANGE = DEFAULT_STYLE.colors["density_ratio"]


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


def _parse_timestep(stem: str) -> int | None:
    try:
        return int(stem.rsplit("_", maxsplit=1)[-1])
    except ValueError:
        return None


def _extract_rho_2d(rho: np.ndarray) -> np.ndarray:
    arr = np.asarray(rho)
    if arr.ndim >= _NDIM_5D:
        return arr[:, :, 0, 0, 0]
    if arr.ndim == _NDIM_4D:
        return arr[:, :, 0, 0]
    if arr.ndim == _NDIM_3D:
        return arr[:, :, 0]
    if arr.ndim == _NDIM_2D:
        return arr
    msg = f"Unsupported rho shape: {arr.shape}"
    raise ValueError(msg)


def _extract_u_mag_2d(u: np.ndarray) -> np.ndarray:
    arr = np.asarray(u)
    if arr.ndim >= _NDIM_5D:
        ux = arr[:, :, 0, 0, 0]
        uy = arr[:, :, 0, 0, 1]
    elif arr.ndim == _NDIM_4D:
        ux = arr[:, :, 0, 0]
        uy = arr[:, :, 0, 1]
    elif arr.ndim == _NDIM_3D:
        ux = arr[:, :, 0]
        uy = arr[:, :, 1]
    else:
        msg = f"Unsupported u shape: {arr.shape}"
        raise ValueError(msg)
    return np.sqrt(ux**2 + uy**2)


def _load_timesteps(files: list[Path], required: tuple[str, ...]) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    iters: list[int] = []
    snapshots: list[dict[str, np.ndarray]] = []
    for fp in files:
        step = _parse_timestep(fp.stem)
        if step is None:
            continue
        with np.load(fp) as raw:
            if not all(key in raw for key in required):
                continue
            snapshots.append({key: np.asarray(raw[key]) for key in required})
            iters.append(step)
    return np.asarray(iters, dtype=int), snapshots


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
