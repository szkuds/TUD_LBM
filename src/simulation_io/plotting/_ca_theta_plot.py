"""Ca–θ plotting utilities for leading and trailing contact lines."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from src.registry import analysis_operator
from src.simulation_io.analysis.droplet_metrics import series_for_files
from src.simulation_io.plotting._analysis_common import _CONTACT_ANGLE_Y_LABEL
from src.simulation_io.plotting._analysis_common import _set_empty_state
from src.simulation_io.plotting.base import AnalysisPlot
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE
from src.simulation_io.plotting.figure_config import LABEL_IT_NORM
from src.simulation_io.plotting.figure_config import LABEL_X_AVG_NORM

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    from src.config import SimulationConfig

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


#: Panel keys, all empty. Returned when no series can be computed.
_ARRAY_KEYS = ("theta_trailing", "theta_leading", "ca_trailing", "ca_leading", "x_time", "x_pos", "timesteps")


def _empty_arrays() -> dict[str, np.ndarray]:
    """Fresh empty arrays for every panel key."""
    return {key: np.array([]) for key in _ARRAY_KEYS}


def _compute_ca_theta_arrays(files: list[Path], config: SimulationConfig) -> dict[str, np.ndarray]:
    """Adapt the shared droplet series to this module's panel array layout.

    Both x-axis variants (time and position) come from the same series, so the
    snapshots are read once no matter how many Ca/θ panels a figure contains.
    """
    series = series_for_files(files, config)
    if series is None:
        return _empty_arrays()

    return {
        "theta_trailing": series.theta_left,
        "theta_leading": series.theta_right,
        "ca_trailing": series.ca_cll_left,
        "ca_leading": series.ca_cll_right,
        "x_time": series.normalised_iteration,
        "x_pos": series.avg_x_location_norm,
        "timesteps": series.iteration,
    }


@analysis_operator(name="ca_theta_vs_time")
class CaThetaVsTimePlot(AnalysisPlot):
    """Dual-axis Ca/θ plot with normalised timestep (Δt/t_tot) on the x-axis."""

    name = "ca_theta_vs_time"
    required_keys = ("ca_left", "ca_right", "cll_left", "cll_right")

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute Ca, θ and x_time arrays from snapshot files."""
        if self.config is None:
            return _empty_arrays()
        return _compute_ca_theta_arrays(files, self.config)

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
            x_label=LABEL_IT_NORM,
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
            return _empty_arrays()
        return _compute_ca_theta_arrays(files, self.config)

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
            x_label=LABEL_X_AVG_NORM,
            style=DualAxisStyle(legend_fontsize=DEFAULT_STYLE.panel_legend_fontsize, legend_outside=True),
        )
