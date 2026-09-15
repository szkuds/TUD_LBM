"""Centralized figure styling shared by every plot operator in this package."""

from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field


@dataclass(frozen=True)
class FigureStyle:
    """Tunable styling knobs for all figures produced by ``simulation_io.plotting``."""

    dpi: int = 150
    panel_figsize: tuple[float, float] = (5.0, 4.0)
    analysis_figsize: tuple[float, float] = (9.0, 4.5)
    comparison_figsize: tuple[float, float] = (12.0, 8.0)
    dual_axis_figsize: tuple[float, float] = (12.0, 8.0)
    ca_theta_figsize: tuple[float, float] = (12.0, 8.0)

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

    # Interface contour: (colour, linestyle) per marker in
    # analysis.interface_contour. Chosen to read on every field colormap above
    # and on the white standalone panel.
    interface_level_styles: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "config": ("tab:red", "-"),
            "measured": ("tab:cyan", "--"),
        }
    )
    interface_linewidth: float = 1.5
    colormap_interface_time: str = "viridis"

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

#: Axis labels for the per-timestep series plotted from ``simulation_data.csv``,
#: so the same quantity is always rendered with the same LaTeX label.
#:
#: Dimensionless *numbers* are not listed here: each one owns its own label in
#: its module under ``analysis/physical_parameters/numbers/``, reachable via
#: ``dimensionless_label(key)``. Keeping them out is what lets those modules be
#: imported without pulling in this package. ``LABEL_RE`` below is not the
#: buoyancy ``Re = sqrt(Ar)`` but the *measured* droplet Reynolds number
#: ``avg_u_x*2R_0/nu`` from ``droplet_metrics.series`` -- a different quantity
#: that happens to share the symbol.
LABEL_CA = r"$\mathrm{Ca}$"
LABEL_RE = r"$\mathrm{Re}$"
LABEL_IT_NORM = r"$\Delta\mathrm{t}/\mathrm{t}_{\mathrm{max}}$"
LABEL_X_AVG_NORM = r"$X_{\mathrm{CM}}/R_0$"

#: Marker and colour per regime label on the regime map.
#:
#: Keys are the values of
#: :class:`src.simulation_io.analysis.accelerations.Regime`, spelled as plain
#: strings rather than imported: ``regime_classification`` reaches this module
#: through ``acceleration_analysis``, so importing the enum here would close an
#: import cycle. ``Regime`` is a :class:`~enum.StrEnum`, so its members index
#: these dicts directly. ``tests/io/test_regime_map_plot.py`` pins the keys to
#: the enum.
REGIME_MARKERS: dict[str, str] = {
    "Pinning": "o",
    "Dissipative": "s",
    "Capillary": "^",
    "Steady": "D",
    "unknown": "x",
}
REGIME_COLORS: dict[str, str] = {
    "Pinning": "tab:blue",
    "Dissipative": "tab:green",
    "Capillary": "tab:red",
    "Steady": "tab:orange",
    "unknown": "tab:gray",
}
