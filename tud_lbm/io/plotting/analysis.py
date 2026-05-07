"""Analysis plot operators for saved simulation history."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from tud_lbm.io.plotting.base import AnalysisPlot
from tud_lbm.registry import analysis_operator

if TYPE_CHECKING:
    from pathlib import Path
    import matplotlib.axes

_NDIM_2D = 2
_NDIM_3D = 3
_NDIM_4D = 4
_NDIM_5D = 5


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
    color: str = "tab:blue",
    ylog: bool = False,
    required_keys: tuple[str, ...] | None = None,
) -> None:
    ax.clear()
    if len(iters) == 0:
        msg = "No data"
        if required_keys:
            msg += f"\n(Requires: {', '.join(required_keys)})"
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)
        return
    ax.scatter(iters, values, s=16, alpha=0.8, color=color, edgecolors="none")
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if ylog:
        ax.set_yscale("log")


class _BaseAnalysisPlot(AnalysisPlot):
    title: str
    ylabel: str
    color: str = "tab:blue"
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
    color = "tab:blue"
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
    color = "tab:orange"
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
    ylabel = "Contact angle (deg)"
    color = "tab:purple"
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
    ylabel = "Contact angle (deg)"
    color = "tab:red"
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
            msg = "No data"
            msg += f"\n(Requires: {', '.join(self.required_keys)})"
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title("Contact angles vs timestep")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Contact angle (deg)")
            return
        ax.scatter(iters, precomputed["left"], s=16, color="tab:purple", alpha=0.8, edgecolors="none", label="Left")
        ax.scatter(iters, precomputed["right"], s=16, color="tab:red", alpha=0.8, edgecolors="none", label="Right")
        ax.set_title("Contact angles vs timestep")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Contact angle (deg)")
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
            msg = "No data"
            msg += f"\n(Requires: {', '.join(self.required_keys)})"
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title("Contact-line speeds vs timestep")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("d(cll)/dt")
            return
        ax.scatter(iters[: len(left)], left, s=16, color="tab:brown", alpha=0.8, edgecolors="none", label="Left")
        ax.scatter(iters[: len(right)], right, s=16, color="tab:pink", alpha=0.8, edgecolors="none", label="Right")
        ax.set_title("Contact-line speeds vs timestep")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("d(cll)/dt")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
