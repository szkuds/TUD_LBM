"""Ca–θ plotting utilities for leading and trailing contact lines."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from tud_lbm.io.analysis.droplet_metrics import analytical_sigma_lg
from tud_lbm.io.analysis.droplet_metrics import backward_diff
from tud_lbm.io.analysis.droplet_metrics import resolve_r_zero
from tud_lbm.io.analysis.droplet_metrics import resolve_step_x
from tud_lbm.io.analysis.droplet_metrics._snapshot import avg_x_location
from tud_lbm.io.analysis.droplet_metrics._snapshot import contact_angles_from_rho
from tud_lbm.io.analysis.droplet_metrics._snapshot import contact_lines_from_rho
from tud_lbm.io.analysis.droplet_metrics._snapshot import optional_contact_metrics
from tud_lbm.io.analysis.droplet_metrics._snapshot import parse_timestep_from_path
from tud_lbm.io.plotting._analysis_common import _CONTACT_ANGLE_Y_LABEL
from tud_lbm.io.plotting._analysis_common import _extract_rho_2d
from tud_lbm.io.plotting._analysis_common import _parse_timestep
from tud_lbm.io.plotting._analysis_common import _set_empty_state
from tud_lbm.io.plotting.base import AnalysisPlot
from tud_lbm.io.plotting.figure_config import DEFAULT_STYLE
from tud_lbm.registry import analysis_operator

if TYPE_CHECKING:
    from collections.abc import Callable
    import matplotlib.axes
    import matplotlib.figure
    from tud_lbm.config import SimulationConfig

_LABEL_CA = r"$\mathrm{Ca}$ [−]"
_LABEL_THETA = r"$\theta$ (degrees)"

# Colour palette matches the reference code supplied by the researcher
_COLOR_TRAILING_CA = "blue"
_COLOR_LEADING_CA = "red"
_COLOR_TRAILING_THETA = "skyblue"
_COLOR_LEADING_THETA = "coral"


# ---------------------------------------------------------------------------
# Dual-axis helpers (Ca left y, θ right y)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DualAxisStyle:
    """Legend placement and font sizes for the dual-axis Ca/θ plot."""

    legend_fontsize: int = DEFAULT_STYLE.legend_fontsize
    legend_outside: bool = False
    axis_label_fontsize: int = DEFAULT_STYLE.axis_label_fontsize
    tick_label_fontsize: int = DEFAULT_STYLE.tick_label_fontsize


_DEFAULT_DUAL_AXIS_STYLE = DualAxisStyle()


def _draw_dual_axis_on_ax(
    ax1: matplotlib.axes.Axes,
    x_data: np.ndarray,
    ca_trailing: np.ndarray,
    ca_leading: np.ndarray,
    theta_trailing: np.ndarray,
    theta_leading: np.ndarray,
    *,
    x_label: str,
    ca_limits: tuple[float, float] | None = None,
    angle_limits: tuple[float, float] | None = None,
    x_limits: tuple[float, float] | None = None,
    style: DualAxisStyle = _DEFAULT_DUAL_AXIS_STYLE,
) -> matplotlib.axes.Axes:
    """Draw dual-axis Ca/θ scatter onto an existing primary axis.

    Left y-axis (``ax1``): capillary number; right y-axis (twin): contact angle.

    Args:
        ax1:           Primary axis (left y = Ca).
        x_data:        Shared x data (normalised time or position).
        ca_trailing:   Ca values for the trailing contact line.
        ca_leading:    Ca values for the leading contact line.
        theta_trailing: Contact angle (degrees) for the trailing contact line.
        theta_leading:  Contact angle (degrees) for the leading contact line.
        x_label:       Label for the shared x-axis.
        ca_limits:       ``(y_min, y_max)`` for the Ca axis.
        angle_limits:    ``(y_min, y_max)`` for the θ axis.
        x_limits:        ``(x_min, x_max)`` for the shared x-axis.
        style:           Legend placement and font sizes (:class:`DualAxisStyle`).

    Returns:
        The twin axes object (right y = θ).
    """
    # Ca on left axis
    ax1.scatter(
        x_data,
        ca_trailing,
        s=20,
        label="Trailing edge (Ca)",
        facecolors="none",
        edgecolors=_COLOR_TRAILING_CA,
        linewidths=1.5,
    )
    ax1.scatter(
        x_data,
        ca_leading,
        s=20,
        label="Leading edge (Ca)",
        facecolors="none",
        edgecolors=_COLOR_LEADING_CA,
        linewidths=1.5,
    )
    ax1.set_xlabel(x_label, fontsize=style.axis_label_fontsize)
    ax1.set_ylabel("Ca", color="black", fontsize=style.axis_label_fontsize)
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.tick_params(axis="both", labelsize=style.tick_label_fontsize)
    ax1.grid(False)
    if ca_limits is not None:
        ax1.set_ylim(ca_limits)
    if x_limits is not None:
        ax1.set_xlim(x_limits)

    # θ on right twin axis
    ax2: matplotlib.axes.Axes = ax1.twinx()
    ax2.scatter(
        x_data,
        theta_trailing,
        s=15,
        marker="x",
        color=_COLOR_TRAILING_THETA,
        label="Trailing edge (θ)",
    )
    ax2.scatter(
        x_data,
        theta_leading,
        s=15,
        marker="x",
        color=_COLOR_LEADING_THETA,
        label="Leading edge (θ)",
    )
    ax2.set_ylabel(_LABEL_THETA, color="black", fontsize=style.axis_label_fontsize)
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.tick_params(axis="both", labelsize=style.tick_label_fontsize)
    if angle_limits is not None:
        ax2.set_ylim(angle_limits)

    # Combine legends from both axes onto ax1
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if style.legend_outside:
        ax1.legend(
            h1 + h2,
            l1 + l2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncols=2,
            fontsize=style.legend_fontsize,
        )
    else:
        ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=style.legend_fontsize)

    return ax2


def plot_dual_axis_ca_theta(
    x_data: np.ndarray,
    ca_trailing: np.ndarray,
    ca_leading: np.ndarray,
    theta_trailing: np.ndarray,
    theta_leading: np.ndarray,
    *,
    x_label: str = r"$X_{\mathrm{avg}}/R_0$",
    figsize: tuple[float, float] = DEFAULT_STYLE.dual_axis_figsize,
    dpi: int = DEFAULT_STYLE.dpi,
    ca_limits: tuple[float, float] | None = None,
    angle_limits: tuple[float, float] | None = None,
    x_limits: tuple[float, float] | None = None,
) -> matplotlib.figure.Figure:
    """Plot capillary number and contact angle on dual y-axes vs a shared x.

    Creates a figure with Ca on the left y-axis and θ on the right y-axis
    (via ``twinx``). Trailing and leading edges are distinguished by colour.

    Args:
        x_data:        Shared x data; typically normalised time (Δt/t_tot) or
            normalised position (X_avg/R_0).
        ca_trailing:   Capillary number for the trailing contact line.
        ca_leading:    Capillary number for the leading contact line.
        theta_trailing: Contact angle (degrees) for the trailing contact line.
        theta_leading:  Contact angle (degrees) for the leading contact line.
        x_label:       Label for the shared x-axis.
        figsize:       Figure size in inches ``(width, height)``.
        dpi:           Resolution in dots per inch.
        ca_limits:     ``(y_min, y_max)`` for the Ca axis; auto-scaled when ``None``.
        angle_limits:  ``(y_min, y_max)`` for the θ axis; auto-scaled when ``None``.
        x_limits:      ``(x_min, x_max)`` for the x-axis; auto-scaled when ``None``.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    _draw_dual_axis_on_ax(
        ax1,
        np.asarray(x_data),
        np.asarray(ca_trailing),
        np.asarray(ca_leading),
        np.asarray(theta_trailing),
        np.asarray(theta_leading),
        x_label=x_label,
        ca_limits=ca_limits,
        angle_limits=angle_limits,
        x_limits=x_limits,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Single-axis Ca-theta scatter (Ca on x, theta on y) - for reference-data overlay
# ---------------------------------------------------------------------------

_COLOR_TRAILING = "tab:blue"
_COLOR_LEADING = "tab:orange"
_MARKER_TRAILING = "o"
_MARKER_LEADING = "s"
_MARKER_REF_TRAILING = "^"
_MARKER_REF_LEADING = "v"


def _draw_ca_theta_on_ax(
    ax: matplotlib.axes.Axes,
    ca_trailing: np.ndarray,
    theta_trailing: np.ndarray,
    ca_leading: np.ndarray,
    theta_leading: np.ndarray,
    *,
    log_scale: bool = False,
    title: str | None = None,
) -> None:
    """Draw Ca–θ scatter (Ca on x, θ on y) onto an existing axis."""
    ax.scatter(
        ca_trailing,
        theta_trailing,
        marker=_MARKER_TRAILING,
        color=_COLOR_TRAILING,
        label="Trailing edge",
        zorder=3,
    )
    ax.scatter(
        ca_leading,
        theta_leading,
        marker=_MARKER_LEADING,
        color=_COLOR_LEADING,
        label="Leading edge",
        zorder=3,
    )
    ax.set_xlabel(_LABEL_CA, fontsize=DEFAULT_STYLE.axis_label_fontsize)
    ax.set_ylabel(_LABEL_THETA, fontsize=DEFAULT_STYLE.axis_label_fontsize)
    ax.tick_params(axis="both", labelsize=DEFAULT_STYLE.tick_label_fontsize)
    ax.grid(False)
    ax.legend(fontsize=DEFAULT_STYLE.legend_fontsize, loc="best")
    if log_scale:
        ax.set_xscale("log")
    if title is not None:
        ax.set_title(title, fontsize=DEFAULT_STYLE.title_fontsize)


def plot_contact_angle_vs_capillary_number(
    ca_trailing: np.ndarray,
    theta_trailing: np.ndarray,
    ca_leading: np.ndarray,
    theta_leading: np.ndarray,
    *,
    log_scale: bool = False,
    title: str | None = None,
    figsize: tuple[float, float] = DEFAULT_STYLE.ca_theta_figsize,
    dpi: int = DEFAULT_STYLE.dpi,
    reference_csv: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """Plot contact angle vs capillary number (θ on y, Ca on x).

    Useful for overlaying an older reference dataset on a new one. For the
    standard dual-axis time/position view use :func:`plot_dual_axis_ca_theta`.

    Args:
        ca_trailing:   Capillary number array for the trailing edge.
        theta_trailing: Contact angle array (degrees) for the trailing edge.
        ca_leading:    Capillary number array for the leading edge.
        theta_leading: Contact angle array (degrees) for the leading edge.
        log_scale:     Apply logarithmic scale to the Ca axis when ``True``.
        title:         Optional figure title.
        figsize:       Figure size in inches ``(width, height)``.
        dpi:           Figure resolution in dots per inch.
        reference_csv: Path to a CSV file with columns ``Ca_trailing``,
            ``theta_trailing``, ``Ca_leading``, ``theta_leading``. When
            supplied the reference data is overlaid as open markers.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    _draw_ca_theta_on_ax(
        ax,
        np.asarray(ca_trailing),
        np.asarray(theta_trailing),
        np.asarray(ca_leading),
        np.asarray(theta_leading),
        log_scale=log_scale,
        title=title,
    )

    if reference_csv is not None:
        _overlay_reference(ax, Path(reference_csv))

    fig.tight_layout()
    return fig


def _overlay_reference(ax: matplotlib.axes.Axes, csv_path: Path) -> None:
    """Add reference dataset from *csv_path* to *ax* as open markers."""
    try:
        import pandas as pd
    except ImportError as exc:
        msg = "pandas is required to load a reference CSV"
        raise ImportError(msg) from exc

    df = pd.read_csv(csv_path)

    req = {"Ca_trailing", "theta_trailing", "Ca_leading", "theta_leading"}
    missing = req - set(df.columns)
    if missing:
        msg = f"Reference CSV is missing columns: {missing}"
        raise ValueError(msg)

    ax.scatter(
        df["Ca_trailing"].to_numpy(),
        df["theta_trailing"].to_numpy(),
        marker=_MARKER_REF_TRAILING,
        facecolors="none",
        edgecolors=_COLOR_TRAILING,
        label="Trailing edge (reference)",
        zorder=2,
    )
    ax.scatter(
        df["Ca_leading"].to_numpy(),
        df["theta_leading"].to_numpy(),
        marker=_MARKER_REF_LEADING,
        facecolors="none",
        edgecolors=_COLOR_LEADING,
        label="Leading edge (reference)",
        zorder=2,
    )
    ax.legend(fontsize=DEFAULT_STYLE.legend_fontsize, loc="best")


def save_figure(fig: matplotlib.figure.Figure, path: str | Path, *, dpi: int = 300) -> None:
    """Save *fig* to *path*; file format is inferred from the path suffix.

    Args:
        fig:  The figure to save.
        path: Output path (e.g. ``"ca_theta.png"`` or ``"ca_theta.pdf"``).
        dpi:  Resolution in dots per inch (only relevant for raster formats).
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


# ---------------------------------------------------------------------------
# AnalysisPlot operators: dual-axis Ca/θ history from saved snapshots
# ---------------------------------------------------------------------------

_CA_THETA_TITLE = "Contact line Ca and contact angle"
_LABEL_IT_NORM = r"$\Delta\mathrm{t}/\mathrm{t}_{\mathrm{max}}$"
_LABEL_X_AVG_NORM = r"$X_{\mathrm{avg}}/R_0$"


def _resolve_pair(
    left: float | None,
    right: float | None,
    rho_2d: np.ndarray | None,
    rho_mean: float,
    from_rho: Callable[[np.ndarray, float], tuple[float, float]],
) -> tuple[float, float]:
    """Return ``(left, right)``, deriving missing values from ``rho_2d`` via *from_rho*."""
    if left is not None and right is not None:
        return left, right
    if rho_2d is not None:
        return from_rho(rho_2d, rho_mean)
    return 0.0, 0.0


def _read_snapshot_metrics(
    fp: Path,
    rho_mean: float,
    offset_x: float,
    *,
    compute_x_pos: bool,
) -> tuple[int, float, float, float, float, float] | None:
    """Read ``(timestep, ca_l, ca_r, cll_l, cll_r, x_pos)`` from one snapshot file.

    Metrics missing from the file are derived from the density field when
    present, else 0.0. Returns ``None`` when the filename has no timestep.
    """
    it = _parse_timestep(fp.stem)
    if it is None:
        return None
    with np.load(fp) as raw:
        ca_l, ca_r, cll_l, cll_r = optional_contact_metrics(raw)
        needs_rho = ca_l is None or ca_r is None or cll_l is None or cll_r is None or compute_x_pos
        rho_2d = _extract_rho_2d(np.asarray(raw["rho"])) if needs_rho and "rho" in raw else None
        ca_l, ca_r = _resolve_pair(ca_l, ca_r, rho_2d, rho_mean, contact_angles_from_rho)
        cll_l, cll_r = _resolve_pair(cll_l, cll_r, rho_2d, rho_mean, contact_lines_from_rho)
        x_pos_val = avg_x_location(rho_2d, rho_mean, offset_x) if rho_2d is not None else 0.0
    return it, float(ca_l), float(ca_r), float(cll_l), float(cll_r), x_pos_val


def _compute_ca_theta_arrays(
    files: list[Path],
    config: SimulationConfig,
    *,
    compute_x_pos: bool,
) -> dict[str, np.ndarray]:
    """Compute Ca, θ and optional position arrays from snapshot files.

    Returns a dict with keys:
    ``theta_trailing``, ``theta_leading``, ``ca_trailing``, ``ca_leading``,
    ``x_time``, ``x_pos`` (empty when *compute_x_pos* is ``False``).
    Returns all-empty arrays when *config* lacks required multiphase fields.
    """
    _empty: dict[str, np.ndarray] = {
        "theta_trailing": np.array([]),
        "theta_leading": np.array([]),
        "ca_trailing": np.array([]),
        "ca_leading": np.array([]),
        "x_time": np.array([]),
        "x_pos": np.array([]),
        "timesteps": np.array([], dtype=int),
    }

    if config.rho_l is None or config.rho_v is None:
        return _empty

    rho_mean = (float(config.rho_l) + float(config.rho_v)) / 2.0

    sorted_files = sorted(files, key=parse_timestep_from_path)

    timesteps: list[int] = []
    ca_left_list: list[float] = []
    ca_right_list: list[float] = []
    cll_left_list: list[float] = []
    cll_right_list: list[float] = []
    x_pos_list: list[float] = []

    step_x = resolve_step_x(config)
    offset_x = step_x if step_x is not None else float(config.grid_shape[0] // 2)

    for fp in sorted_files:
        metrics = _read_snapshot_metrics(fp, rho_mean, offset_x, compute_x_pos=compute_x_pos)
        if metrics is None:
            continue
        it, ca_l, ca_r, cll_l, cll_r, x_pos_val = metrics
        timesteps.append(it)
        ca_left_list.append(ca_l)
        ca_right_list.append(ca_r)
        cll_left_list.append(cll_l)
        cll_right_list.append(cll_r)
        if compute_x_pos:
            x_pos_list.append(x_pos_val)

    if not ca_left_list:
        return _empty

    sigma = analytical_sigma_lg(config)
    if sigma is None:
        return _empty

    nu = (float(config.tau) - 0.5) / 3.0
    save_iv = max(config.save_interval, 1)

    v_left = backward_diff(np.array(cll_left_list), save_iv)
    v_right = backward_diff(np.array(cll_right_list), save_iv)

    t_arr = np.array(timesteps, dtype=float)
    t_min, t_max = t_arr.min(), t_arr.max()
    x_time = (t_arr - t_min) / (t_max - t_min) if t_max > t_min else np.zeros_like(t_arr)

    if compute_x_pos:
        r_zero = resolve_r_zero(config)
        x_pos = np.array(x_pos_list) / r_zero.value if r_zero.value > 0 else np.zeros(len(x_pos_list))
    else:
        x_pos = np.array([])

    return {
        "theta_trailing": np.array(ca_left_list),
        "theta_leading": np.array(ca_right_list),
        "ca_trailing": v_left * nu / sigma,
        "ca_leading": v_right * nu / sigma,
        "x_time": x_time,
        "x_pos": x_pos,
        "timesteps": np.array(timesteps, dtype=int),
    }


@analysis_operator(name="ca_theta_vs_time")
class CaThetaVsTimePlot(AnalysisPlot):
    """Dual-axis Ca/θ plot with normalised timestep (Δt/t_tot) on the x-axis."""

    name = "ca_theta_vs_time"
    required_keys = ("ca_left", "ca_right", "cll_left", "cll_right")

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute Ca, θ and x_time arrays from snapshot files."""
        if self.config is None:
            return {
                k: np.array([])
                for k in ("theta_trailing", "theta_leading", "ca_trailing", "ca_leading", "x_time", "x_pos")
            }
        return _compute_ca_theta_arrays(files, self.config, compute_x_pos=False)

    def render(self, ax: matplotlib.axes.Axes, precomputed: dict[str, np.ndarray]) -> None:
        """Draw dual-axis Ca/θ scatter with normalised time on x."""
        ax.clear()
        if len(precomputed.get("x_time", np.array([]))) == 0:
            _set_empty_state(ax, title=_CA_THETA_TITLE, ylabel=_CONTACT_ANGLE_Y_LABEL)
            return

        _draw_dual_axis_on_ax(
            ax,
            precomputed["x_time"],
            precomputed["ca_trailing"],
            precomputed["ca_leading"],
            precomputed["theta_trailing"],
            precomputed["theta_leading"],
            x_label=_LABEL_IT_NORM,
            style=DualAxisStyle(legend_fontsize=DEFAULT_STYLE.panel_legend_fontsize, legend_outside=True),
        )


@analysis_operator(name="ca_theta_vs_x")
class CaThetaVsXPlot(AnalysisPlot):
    """Dual-axis Ca/θ plot with normalised position (X_avg/R_0) on the x-axis."""

    name = "ca_theta_vs_x"
    required_keys = ("ca_left", "ca_right", "cll_left", "cll_right")

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute Ca, θ and x_pos arrays from snapshot files."""
        if self.config is None:
            return {
                k: np.array([])
                for k in ("theta_trailing", "theta_leading", "ca_trailing", "ca_leading", "x_time", "x_pos")
            }
        return _compute_ca_theta_arrays(files, self.config, compute_x_pos=True)

    def render(self, ax: matplotlib.axes.Axes, precomputed: dict[str, np.ndarray]) -> None:
        """Draw dual-axis Ca/θ scatter with normalised position on x."""
        ax.clear()
        if len(precomputed.get("x_pos", np.array([]))) == 0:
            _set_empty_state(ax, title=_CA_THETA_TITLE, ylabel=_CONTACT_ANGLE_Y_LABEL)
            return

        _draw_dual_axis_on_ax(
            ax,
            precomputed["x_pos"],
            precomputed["ca_trailing"],
            precomputed["ca_leading"],
            precomputed["theta_trailing"],
            precomputed["theta_leading"],
            x_label=_LABEL_X_AVG_NORM,
            style=DualAxisStyle(legend_fontsize=DEFAULT_STYLE.panel_legend_fontsize, legend_outside=True),
        )
