"""Analysis plot operators for saved simulation history."""

from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from matplotlib.colors import TABLEAU_COLORS
from tud_lbm.io.physical_parameters import _get_setup_contact_line_length
from tud_lbm.io.plotting.base import AnalysisPlot
from tud_lbm.registry import analysis_operator

if TYPE_CHECKING:
    import matplotlib.axes
    import pandas as pd
    from tud_lbm.config import SimulationConfig

_NDIM_2D = 2
_NDIM_3D = 3
_NDIM_4D = 4
_NDIM_5D = 5

_X_LABEL_TIMESTEP = "Timestep"
_EMPTY_DATA_TEXT = "No data"
_REQUIRES_PREFIX = "(Requires: "
_CONTACT_ANGLE_Y_LABEL = "Contact angle (deg)"
_CONTACT_ANGLES_TITLE = "Contact angles vs timestep"
_CONTACT_LINE_SPEEDS_TITLE = "Contact-line speeds vs timestep"
_CONTACT_LINE_SPEED_Y_LABEL = "d(cll)/dt"
_CAPILLARY_NUMBERS_TITLE = "Capillary numbers vs timestep"
_CAPILLARY_NUMBER_Y_LABEL = "Ca [nu*U/sigma]"
_NORMALIZED_CLL_TITLE = "Normalised contact-line position vs timestep"
_NORMALIZED_CLL_Y_LABEL = "(x - x_step) / R"
_COLOR_TAB_BLUE = "tab:blue"
_COLOR_TAB_PURP = "tab:purple"
_COLOR_TAB_RED = "tab:red"
_COLOR_TAB_ORANGE = "tab:orange"

_CS2 = 1.0 / 3.0
_WIDTH_EPS = 1e-15
_DIR_SPLIT_PARTS = 2

_LABEL_CA = r"$\mathrm{Ca}$"
_LABEL_IT_NORM = r"$\mathrm{it}/\mathrm{it}_{\mathrm{max}}$"
_LABEL_X_AVG_NORM = r"$X_{\mathrm{avg}}/R_0$"
_CONFIG_TOML = "config.toml"


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
        fontsize=9,
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
    ax.scatter(iters, values, s=16, alpha=0.8, color=color, edgecolors="none")
    ax.set_xlabel(_X_LABEL_TIMESTEP)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if ylog:
        ax.set_yscale("log")


def _derive_surface_tension(config: SimulationConfig) -> float | None:
    """Compute lattice surface tension gamma=(2/3)*(kappa/W)*(delta_rho**2)."""
    kappa = getattr(config, "kappa", None)
    width = getattr(config, "interface_width", None)
    rho_l = getattr(config, "rho_l", None)
    rho_v = getattr(config, "rho_v", None)
    width_val = float(width) if width is not None else 0.0
    if any(x is None for x in (kappa, width, rho_l, rho_v)) or abs(width_val) < _WIDTH_EPS:
        return None
    drho = float(rho_l) - float(rho_v)
    return (2.0 / 3.0) * (float(kappa) / float(width)) * drho**2


def _resolve_initial_radius(config: SimulationConfig) -> float | None:
    """Return initial drop radius in lattice units from config.initialisation."""
    init = getattr(config, "initialisation", None)
    if not init or not isinstance(init, dict):
        return None
    try:
        radii = init.get("radii", [])
        if not radii:
            return None
        nx = float(config.grid_shape[0])
        ny = float(config.grid_shape[1])
        return float(radii[0]) * min(nx, ny)
    except (IndexError, ValueError, TypeError):
        return None


def _resolve_step_x(config: SimulationConfig) -> float | None:
    """Return the chemical step x-position in lattice units from config."""
    chem = getattr(config, "chemical_step_config", None)
    if not chem or not isinstance(chem, dict):
        return None
    loc = chem.get("chemical_step_location")
    if loc is None:
        return None
    return float(loc) * float(config.grid_shape[0])


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


@analysis_operator(name="max_velocity")
class MaxVelocityPlot(_BaseAnalysisPlot):
    """Plot maximum velocity magnitude over time."""

    name = "max_velocity"
    title = "Maximum velocity vs timestep"
    ylabel = "max(|u|)"
    color = _COLOR_TAB_BLUE
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
    color = _COLOR_TAB_ORANGE
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
    color = "tab:green"
    required_keys = ("rho",)

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute average density values for each timestep file."""
        iters, snapshots = _load_timesteps(files, ("rho",))
        vals = np.asarray([float(np.mean(_extract_rho_2d(snap["rho"]))) for snap in snapshots], dtype=float)
        return {"iters": iters, "values": vals}


@analysis_operator(name="contact_angle_left")
class ContactAngleLeftPlot(_BaseAnalysisPlot):
    """Plot left contact angle over time."""

    name = "contact_angle_left"
    title = "Left contact angle vs timestep"
    ylabel = _CONTACT_ANGLE_Y_LABEL
    color = _COLOR_TAB_PURP
    required_keys = ("ca_left",)

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute left contact angle values for each timestep file."""
        iters, snapshots = _load_timesteps(files, ("ca_left",))
        vals = np.asarray([float(np.asarray(s["ca_left"]).squeeze()) for s in snapshots], dtype=float)
        return {"iters": iters, "values": vals}


@analysis_operator(name="contact_angle_right")
class ContactAngleRightPlot(_BaseAnalysisPlot):
    """Plot right contact angle over time."""

    name = "contact_angle_right"
    title = "Right contact angle vs timestep"
    ylabel = _CONTACT_ANGLE_Y_LABEL
    color = _COLOR_TAB_RED
    required_keys = ("ca_right",)

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute right contact angle values for each timestep file."""
        iters, snapshots = _load_timesteps(files, ("ca_right",))
        vals = np.asarray([float(np.asarray(s["ca_right"]).squeeze()) for s in snapshots], dtype=float)
        return {"iters": iters, "values": vals}


class _ContactLineSpeedBase(_BaseAnalysisPlot):
    cl_key: str

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        iters, snapshots = _load_timesteps(files, (self.cl_key,))
        cl = np.asarray([float(np.asarray(s[self.cl_key]).squeeze()) for s in snapshots], dtype=float)
        if len(cl) == 0:
            return {"iters": iters, "values": cl}
        if len(cl) == 1:
            return {"iters": iters, "values": np.asarray([0.0], dtype=float)}
        d_iter = np.diff(iters).astype(float)
        d_iter[d_iter == 0] = np.nan
        speeds = np.diff(cl) / d_iter
        vals = np.concatenate(([0.0], speeds))
        vals = np.nan_to_num(vals, nan=0.0)
        return {"iters": iters, "values": vals}


@analysis_operator(name="contact_line_speed_left")
class ContactLineSpeedLeftPlot(_ContactLineSpeedBase):
    """Plot left contact-line speed over time."""

    name = "contact_line_speed_left"
    title = "Left contact-line speed vs timestep"
    ylabel = "d(cll_left)/dt"
    color = "tab:brown"
    cl_key = "cll_left"
    required_keys = ("cll_left",)


@analysis_operator(name="contact_line_speed_right")
class ContactLineSpeedRightPlot(_ContactLineSpeedBase):
    """Plot right contact-line speed over time."""

    name = "contact_line_speed_right"
    title = "Right contact-line speed vs timestep"
    ylabel = "d(cll_right)/dt"
    color = "tab:pink"
    cl_key = "cll_right"
    required_keys = ("cll_right",)


@analysis_operator(name="contact_angles_pair")
class ContactAnglesPairPlot(AnalysisPlot):
    """Render paired left/right contact-angle history."""

    name = "contact_angles_pair"
    required_keys = ("ca_left", "ca_right")

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute left/right contact-angle arrays for all snapshots."""
        iters, snapshots = _load_timesteps(files, ("ca_left", "ca_right"))
        left = np.asarray([float(np.asarray(s["ca_left"]).squeeze()) for s in snapshots], dtype=float)
        right = np.asarray([float(np.asarray(s["ca_right"]).squeeze()) for s in snapshots], dtype=float)
        return {"iters": iters, "left": left, "right": right}

    def render(self, ax: matplotlib.axes.Axes, precomputed: dict[str, np.ndarray]) -> None:
        """Draw the paired contact-angle scatter plot."""
        ax.clear()
        iters = precomputed["iters"]
        if len(iters) == 0:
            _set_empty_state(
                ax,
                title=_CONTACT_ANGLES_TITLE,
                ylabel=_CONTACT_ANGLE_Y_LABEL,
                required_keys=self.required_keys,
            )
            return
        ax.scatter(iters, precomputed["left"], s=16, color=_COLOR_TAB_PURP, alpha=0.8, edgecolors="none", label="Left")
        ax.scatter(iters, precomputed["right"], s=16, color=_COLOR_TAB_RED, alpha=0.8, edgecolors="none", label="Right")
        ax.set_title(_CONTACT_ANGLES_TITLE)
        ax.set_xlabel(_X_LABEL_TIMESTEP)
        ax.set_ylabel(_CONTACT_ANGLE_Y_LABEL)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)


@analysis_operator(name="contact_line_speeds_pair")
class ContactLineSpeedsPairPlot(AnalysisPlot):
    """Render paired left/right contact-line speed history."""

    name = "contact_line_speeds_pair"
    required_keys = ("cll_left", "cll_right")

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute left/right contact-line speed arrays for all snapshots."""
        left = ContactLineSpeedLeftPlot().compute(files)
        right = ContactLineSpeedRightPlot().compute(files)
        iters = left["iters"] if len(left["iters"]) >= len(right["iters"]) else right["iters"]
        return {"iters": iters, "left": left["values"], "right": right["values"]}

    def render(self, ax: matplotlib.axes.Axes, precomputed: dict[str, np.ndarray]) -> None:
        """Draw the paired contact-line speed scatter plot."""
        ax.clear()
        iters = precomputed["iters"]
        left = precomputed["left"]
        right = precomputed["right"]
        if len(iters) == 0:
            _set_empty_state(
                ax,
                title=_CONTACT_LINE_SPEEDS_TITLE,
                ylabel=_CONTACT_LINE_SPEED_Y_LABEL,
                required_keys=self.required_keys,
            )
            return
        ax.scatter(iters[: len(left)], left, s=16, color="tab:brown", alpha=0.8, edgecolors="none", label="Left")
        ax.scatter(iters[: len(right)], right, s=16, color="tab:pink", alpha=0.8, edgecolors="none", label="Right")
        ax.set_title(_CONTACT_LINE_SPEEDS_TITLE)
        ax.set_xlabel(_X_LABEL_TIMESTEP)
        ax.set_ylabel(_CONTACT_LINE_SPEED_Y_LABEL)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)


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


# -- Droplet CSV export -------------------------------------------------------

_SUPPORTED_SIM_TYPES = {
    "multiphase_wetting",
    "multiphase_hysteresis",
    "multiphase_hysteresis_chemical_step",
}
_CSV_FILENAME = "simulation_data.csv"
_COMPARISON_DIR = "comparison_analysis"


def _resolve_r_zero(config: SimulationConfig) -> float:
    """Initial droplet radius in lattice units.

    Derived from half of the setup contact-line length if available.
    Otherwise, defaults to ``initialisation.radii[0] * min(nx, ny)``, matching the
    computation used when the contact line length cannot be determined.
    Falls back to 27.0 when all keys are absent.
    """
    length = _get_setup_contact_line_length(config)
    if length is not None:
        return length / 2.0

    init = config.initialisation
    radii = init.get("radii", []) if isinstance(init, dict) else []
    if radii:
        return float(radii[0]) * float(min(config.grid_shape[0], config.grid_shape[1]))
    return 27.0


def _inclination_angle_deg(config: SimulationConfig) -> float:
    gf = config.gravity_force
    if gf and isinstance(gf, dict):
        return float(gf.get("inclination_angle_deg", 0.0))
    return 0.0


def _sigma_lg(config: SimulationConfig) -> float:
    return (
        (2.0 / 3.0)
        * (float(config.kappa) / float(config.interface_width))
        * (float(config.rho_l) - float(config.rho_v)) ** 2
    )


def _interpolate_interface(row: np.ndarray, rho_mean: float) -> tuple[float, float]:
    """Sub-cell left/right interface positions via linear interpolation."""
    mask = (row < rho_mean).astype(int)
    diff = np.diff(mask)
    li = int(np.nonzero(diff == -1)[0][0])
    ri = int(np.nonzero(diff == 1)[0][0]) + 1
    x_left = li + (rho_mean - row[li]) / (row[li + 1] - row[li])
    x_right = ri - (rho_mean - row[ri]) / (row[ri - 1] - row[ri])
    return x_left, x_right


def _ca_from_rho(rho_2d: np.ndarray, rho_mean: float) -> tuple[float, float]:
    xl0, xr0 = _interpolate_interface(rho_2d[:, 1], rho_mean)
    xl1, xr1 = _interpolate_interface(rho_2d[:, 2], rho_mean)
    ca_left = float(np.rad2deg(math.pi / 2.0 + np.arctan(xl0 - xl1)))
    ca_right = float(np.rad2deg(math.pi / 2.0 + np.arctan(xr1 - xr0)))
    return ca_left, ca_right


def _cll_from_rho(rho_2d: np.ndarray, rho_mean: float) -> tuple[float, float]:
    return _interpolate_interface(rho_2d[:, 1], rho_mean)


def _center_of_mass(rho_2d: np.ndarray, rho_mean: float) -> tuple[float, float]:
    mask = rho_2d > rho_mean
    xi, yi = np.indices(rho_2d.shape)
    total = np.sum(mask * rho_2d)
    return float(np.sum(xi * mask * rho_2d) / total), float(np.sum(yi * mask * rho_2d) / total)


def _avg_x_location(rho_2d: np.ndarray, rho_mean: float, offset_x: float) -> float:
    nx = rho_2d.shape[0]
    mask = rho_2d > rho_mean
    x_idx = np.arange(nx, dtype=float) - offset_x
    return float(np.sum(x_idx[:, None] * mask) / np.sum(mask))


def _backward_diff(arr: np.ndarray, interval: int) -> np.ndarray:
    shifted = np.roll(arr, 1)
    shifted[0] = arr[0]
    return (arr - shifted) / max(interval, 1)


def _empty_csv_rows() -> dict[str, list[float | int]]:
    keys = (
        "iteration",
        "avg_u_x",
        "avg_u_y",
        "avg_x_location",
        "cll_left",
        "cll_right",
        "ca_left",
        "ca_right",
        "cm_x",
        "cm_y",
    )
    return {k: [] for k in keys}


def _append_csv_row(rows: dict[str, list[float | int]], row_data: dict[str, float | int]) -> None:
    for key, val in row_data.items():
        rows[key].append(val)


def _extract_optional_contact_metrics(
    raw: np.lib.npyio.NpzFile,
) -> tuple[float | None, float | None, float | None, float | None]:
    ca_l = float(np.asarray(raw["ca_left"]).squeeze()) if "ca_left" in raw else None
    ca_r = float(np.asarray(raw["ca_right"]).squeeze()) if "ca_right" in raw else None
    cll_l = float(np.asarray(raw["cll_left"]).squeeze()) if "cll_left" in raw else None
    cll_r = float(np.asarray(raw["cll_right"]).squeeze()) if "cll_right" in raw else None
    return ca_l, ca_r, cll_l, cll_r


def _collect_csv_rows(files: list[Path], rho_mean: float, offset_x: float) -> dict[str, list[float | int]]:
    rows = _empty_csv_rows()
    for fp in files:
        it = _parse_timestep(fp.stem)
        if it is None:
            continue
        with np.load(fp) as raw:
            rho = np.asarray(raw["rho"])
            u = np.asarray(raw["u"])
            ca_l, ca_r, cll_l, cll_r = _extract_optional_contact_metrics(raw)

        rho_2d = rho[:, :, 0, 0, 0]
        u_x = u[:, :, 0, 0, 0]
        u_y = u[:, :, 0, 0, 1]
        mask = rho_2d > rho_mean
        n_liq = np.sum(mask)
        avg_ux = float(np.sum(u_x * mask) / n_liq) if n_liq > 0 else 0.0
        avg_uy = float(np.sum(u_y * mask) / n_liq) if n_liq > 0 else 0.0

        if cll_l is None:
            cll_l, cll_r = _cll_from_rho(rho_2d, rho_mean)
        if ca_l is None:
            ca_l, ca_r = _ca_from_rho(rho_2d, rho_mean)

        cm_x, cm_y = _center_of_mass(rho_2d, rho_mean)
        avg_x = _avg_x_location(rho_2d, rho_mean, offset_x)
        _append_csv_row(
            rows,
            {
                "iteration": it,
                "avg_u_x": avg_ux,
                "avg_u_y": avg_uy,
                "avg_x_location": avg_x,
                "cll_left": cll_l,
                "cll_right": cll_r,
                "ca_left": ca_l,
                "ca_right": ca_r,
                "cm_x": cm_x,
                "cm_y": cm_y,
            },
        )
    return rows


def _finalize_csv_dataframe(
    df: pd.DataFrame,
    *,
    r_zero: float,
    nu: float,
    sigma_lg: float,
    incl_deg: float,
    save_iv: int,
) -> pd.DataFrame:
    it_max = df["iteration"].max()
    df["normalised_iteration"] = df["iteration"] / it_max
    df["avg_x_location_norm"] = df["avg_x_location"] / r_zero
    df["v_left"] = _backward_diff(df["cll_left"].to_numpy(), save_iv)
    df["v_right"] = _backward_diff(df["cll_right"].to_numpy(), save_iv)
    df["v_cm"] = _backward_diff(df["cm_x"].to_numpy(), save_iv)
    df["Ca"] = (df["avg_u_x"] * nu) / sigma_lg
    df["Ca_cll_left"] = (df["v_left"] * nu) / sigma_lg
    df["Ca_cll_right"] = (df["v_right"] * nu) / sigma_lg
    df["Ca_cm"] = (df["v_cm"] * nu) / sigma_lg
    df["Ca_norm"] = df["Ca"] / math.sin(math.radians(incl_deg)) if incl_deg > 0 else df["Ca"]
    return df[
        [
            "iteration",
            "normalised_iteration",
            "avg_x_location",
            "avg_x_location_norm",
            "avg_u_x",
            "avg_u_y",
            "cll_left",
            "cll_right",
            "v_left",
            "v_right",
            "v_cm",
            "ca_left",
            "ca_right",
            "cm_x",
            "cm_y",
            "Ca",
            "Ca_cll_left",
            "Ca_cll_right",
            "Ca_cm",
            "Ca_norm",
        ]
    ]


def build_simulation_csv(run_dir: str | Path, config: SimulationConfig) -> Path | None:
    """Compute per-timestep metrics and write ``simulation_data.csv``.

    Skips silently (returns ``None``) when *config.sim_type* is not a
    supported wetting variant. Saves the CSV directly to *run_dir*.

    Args:
        run_dir: Run directory (contains ``data/timestep_*.npz``).
        config:  :class:`~tud_lbm.config.simulation_config.SimulationConfig`.

    Returns:
        Path to the written CSV, or ``None`` when skipped.
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas is required for CSV export. Install with: pip install pandas")
        return None

    if config.sim_type not in _SUPPORTED_SIM_TYPES:
        return None

    run_dir = Path(run_dir)
    data_dir = run_dir / "data"
    if not data_dir.exists():
        return None

    files = sorted(data_dir.glob("timestep_*.npz"), key=_parse_timestep_from_path)
    if not files:
        return None

    rho_mean = (float(config.rho_l) + float(config.rho_v)) / 2.0
    sigma_lg = _sigma_lg(config)
    nu = (float(config.tau) - 0.5) / 3.0
    r_zero = _resolve_r_zero(config)
    incl_deg = _inclination_angle_deg(config)
    save_iv = max(config.save_interval, 1)

    step_x = _resolve_step_x(config)
    offset_x = step_x if step_x is not None else float(config.grid_shape[0] // 2)

    rows = _collect_csv_rows(files, rho_mean, offset_x)

    df = pd.DataFrame(rows)
    if df.empty:
        return None
    df = _finalize_csv_dataframe(
        df,
        r_zero=r_zero,
        nu=nu,
        sigma_lg=sigma_lg,
        incl_deg=incl_deg,
        save_iv=save_iv,
    )

    csv_path = run_dir / _CSV_FILENAME
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    return csv_path


def _parse_timestep_from_path(p: Path) -> int:
    try:
        return int(p.stem.rsplit("_", maxsplit=1)[-1])
    except ValueError:
        return -1


# -- Comparison plots ---------------------------------------------------------

_COMPARISON_PLOT_CONFIGS = [
    {
        "filename": "01_Ca_vs_iteration.png",
        "x": "normalised_iteration",
        "y": "Ca",
        "xlabel": _LABEL_IT_NORM,
        "ylabel": _LABEL_CA,
    },
    {
        "filename": "02_Ca_vs_x_avg.png",
        "x": "avg_x_location_norm",
        "y": "Ca",
        "xlabel": _LABEL_X_AVG_NORM,
        "ylabel": _LABEL_CA,
    },
    {
        "filename": "03_Ca_contact_line_vs_iteration.png",
        "x": "normalised_iteration",
        "y_pair": ("Ca_cll_left", "Ca_cll_right"),
        "pair_labels": ("Trailing", "Leading"),
        "xlabel": _LABEL_IT_NORM,
        "ylabel": _LABEL_CA,
    },
    {
        "filename": "04_Ca_contact_line_vs_x_avg.png",
        "x": "avg_x_location_norm",
        "y_pair": ("Ca_cll_left", "Ca_cll_right"),
        "pair_labels": ("Trailing", "Leading"),
        "xlabel": _LABEL_X_AVG_NORM,
        "ylabel": _LABEL_CA,
    },
    {
        "filename": "05_Ca_cm_vs_iteration.png",
        "x": "normalised_iteration",
        "y": "Ca_cm",
        "xlabel": _LABEL_IT_NORM,
        "ylabel": _LABEL_CA,
    },
    {
        "filename": "06_Ca_cm_vs_x_avg.png",
        "x": "avg_x_location_norm",
        "y": "Ca_cm",
        "xlabel": _LABEL_X_AVG_NORM,
        "ylabel": _LABEL_CA,
    },
    {
        "filename": "07_contact_angles_vs_iteration.png",
        "x": "normalised_iteration",
        "y_pair": ("ca_left", "ca_right"),
        "pair_labels": ("Trailing", "Leading"),
        "xlabel": _LABEL_IT_NORM,
        "ylabel": "Contact angle (degrees)",
    },
    {
        "filename": "08_contact_angles_vs_x_avg.png",
        "x": "avg_x_location_norm",
        "y_pair": ("ca_left", "ca_right"),
        "pair_labels": ("Trailing", "Leading"),
        "xlabel": _LABEL_X_AVG_NORM,
        "ylabel": r"$\theta$ (degrees)",
    },
]

_COMPARISON_MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "+", "x"]


def _safe_load_config(toml_path: Path) -> SimulationConfig | None:
    from tud_lbm.io.readers import TomlAdapter

    try:
        return TomlAdapter().load(str(toml_path))
    except Exception as exc:  # noqa: BLE001
        print(f"  Skipped {toml_path.parent}: {exc}")
        return None


def _load_comparison_entries(parent_dir: Path) -> list[dict]:
    """Collect all processed run CSVs under *parent_dir*, sorted by advancing CA."""
    try:
        import pandas as pd
    except ImportError:
        return []

    entries = []
    for csv_path in sorted(parent_dir.rglob(_CSV_FILENAME)):
        run_dir = csv_path.parent
        if _COMPARISON_DIR in run_dir.parts:
            continue
        toml_path = run_dir / _CONFIG_TOML
        if not toml_path.exists():
            continue
        config = _safe_load_config(toml_path)
        if config is None:
            continue

        adv_ca = config.wetting_config.get("advancing_ca", 0) if config.wetting_config else 0
        label = config.simulation_name or _clean_dir_label(run_dir.name)
        entries.append(
            {
                "label": label,
                "sort_key": adv_ca,
                "data": pd.read_csv(csv_path),
            }
        )

    return sorted(entries, key=lambda e: e["sort_key"])


def _clean_dir_label(dir_name: str) -> str:
    parts = dir_name.split("_", 1)
    name = parts[1] if len(parts) == _DIR_SPLIT_PARTS and parts[0].replace("-", "").isdigit() else dir_name
    return name.replace("_", " ").capitalize()


def _plot_comparison_entry(
    ax: matplotlib.axes.Axes,
    df: pd.DataFrame,
    pc: dict,
    x_col: str,
    color: str,
    marker: str,
    label: str,
) -> None:
    """Plot one entry from _COMPARISON_PLOT_CONFIGS for a single run."""
    if "y_pair" in pc:
        y1, y2 = pc["y_pair"]
        l1, l2 = pc["pair_labels"]
        if y1 in df.columns:
            ax.scatter(df[x_col], df[y1], marker=marker, s=15, color=color, alpha=1.0, label=f"{label} ({l1})")
        if y2 in df.columns:
            ax.scatter(df[x_col], df[y2], marker=marker, s=15, color=color, alpha=0.5, label=f"{label} ({l2})")
    else:
        y_col = pc["y"]
        if y_col in df.columns:
            ax.scatter(df[x_col], df[y_col], marker=marker, s=15, color=color, label=label)


def compare_runs(parent_dir: str | Path) -> None:
    """Generate 8 comparison scatter plots across all processed runs.

    Reads ``simulation_data.csv`` files produced by :func:`build_simulation_csv`
    from every run directory under *parent_dir* and writes 8 plots to
    ``<parent_dir>/comparison_analysis/``.

    Args:
        parent_dir: Parent directory that contains the individual run directories.
    """
    import matplotlib.pyplot as plt

    parent_dir = Path(parent_dir)
    entries = _load_comparison_entries(parent_dir)
    if not entries:
        print("No processed simulation data found for comparison.")
        return

    out_dir = parent_dir / _COMPARISON_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = list(TABLEAU_COLORS.values())

    for pc in _COMPARISON_PLOT_CONFIGS:
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, entry in enumerate(entries):
            color = colors[i % len(colors)]
            marker = _COMPARISON_MARKERS[i % len(_COMPARISON_MARKERS)]
            x_col = pc["x"]
            if x_col not in entry["data"].columns:
                continue
            _plot_comparison_entry(ax, entry["data"], pc, x_col, color, marker, entry["label"])

        ax.set_xlabel(pc["xlabel"], fontsize=24)
        ax.set_ylabel(pc["ylabel"], fontsize=24)
        ax.tick_params(axis="both", labelsize=16)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=12, loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = out_dir / pc["filename"]
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


# -- CLI entry point ----------------------------------------------------------


def process_parent_dir(parent_dir: str | Path) -> tuple[int, int]:
    """Discover runs, export per-run CSVs, and generate comparison plots.

    Discovers runs by searching for ``config.toml`` files recursively.
    Skips directories whose path contains ``"init"`` or ``"comparison_analysis"``.
    Writes ``simulation_data.csv`` per run, then generates 8 comparison plots
    when at least one run produced CSV data.

    Args:
        parent_dir: Absolute or relative path to the parent directory.

    Returns:
        A tuple ``(n_runs_found, n_runs_with_csv)``.
    """
    parent = Path(parent_dir)
    skip_dirs = {"init", _COMPARISON_DIR}

    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for toml in sorted(parent.rglob(_CONFIG_TOML)):
        rd = toml.parent
        if rd in seen or any(s in str(rd).lower() for s in skip_dirs):
            continue
        seen.add(rd)
        run_dirs.append(rd)

    if not run_dirs:
        return 0, 0

    print(f"Found {len(run_dirs)} simulation(s) to process.")

    n_ok = 0
    for rd in run_dirs:
        config = _safe_load_config(rd / _CONFIG_TOML)
        if config and build_simulation_csv(rd, config) is not None:
            n_ok += 1

    if n_ok > 0:
        print("\nGenerating comparison plots...")
        compare_runs(parent)
        print(f"Done. Comparison plots in {parent / _COMPARISON_DIR}")

    return len(run_dirs), n_ok


def main(parent_dir: str | None = None) -> None:
    """Script entry point for batch CSV export and run comparison plots."""
    if parent_dir is None:
        parent_dir = input("Enter parent directory (absolute path): ").strip()

    parent = Path(parent_dir)
    if not parent.is_dir():
        print(f"Directory not found: {parent}")
        sys.exit(1)

    n_runs, n_ok = process_parent_dir(parent)

    if n_runs == 0:
        print("No simulation run directories found.")
        sys.exit(1)
    if n_ok == 0:
        print("No runs produced CSV data. Check sim_type and data files.")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
