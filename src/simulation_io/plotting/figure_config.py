"""Centralized figure styling shared by every plot operator in this package."""

from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field


@dataclass(frozen=True)
class FigureStyle:
    """Tunable styling knobs for all figures produced by ``simulation_io.plotting``."""

    dpi: int = 150
    panel_figsize: tuple[float, float] = (5.0, 4.0)
    analysis_figsize: tuple[float, float] = (7.0, 4.5)
    comparison_figsize: tuple[float, float] = (10.0, 6.0)
    dual_axis_figsize: tuple[float, float] = (10.0, 6.0)
    ca_theta_figsize: tuple[float, float] = (10.0, 6.0)

    suptitle_fontsize: int = 12
    title_fontsize: int = 14
    axis_label_fontsize: int = 16
    tick_label_fontsize: int = 12
    legend_fontsize: int = 12
    pair_legend_fontsize: int = 8
    panel_legend_fontsize: int = 6
    empty_state_fontsize: int = 9
    error_text_fontsize: int = 8

    comparison_axis_label_fontsize: int = 24
    comparison_tick_label_fontsize: int = 16
    comparison_legend_fontsize: int = 12

    scatter_marker_size: int = 16
    scatter_alpha: float = 0.8

    colormap_density: str = "viridis"
    colormap_velocity: str = "plasma"
    colormap_force: str = "cividis"
    # Diverging: the bulk pressure straddles zero across a diffuse interface.
    colormap_pressure: str = "coolwarm"

    quiver_color: str = "white"
    quiver_alpha: float = 0.7

    colors: dict[str, str] = field(
        default_factory=lambda: {
            "max_velocity": "tab:blue",
            "density_ratio": "tab:orange",
            "avg_density": "tab:green",
            "total_mass": "tab:olive",
            "contact_angle_left": "tab:purple",
            "contact_angle_right": "tab:red",
            "contact_line_speed_left": "tab:brown",
            "contact_line_speed_right": "tab:pink",
        }
    )


DEFAULT_STYLE = FigureStyle()

#: Axis labels shared across plot modules, so the same quantity is always
#: rendered with the same LaTeX label.
LABEL_CA = r"$\mathrm{Ca}$"
LABEL_RE = r"$\mathrm{Re}$"
LABEL_IT_NORM = r"$\Delta\mathrm{t}/\mathrm{t}_{\mathrm{max}}$"
LABEL_X_AVG_NORM = r"$X_{\mathrm{avg}}/R_0$"
