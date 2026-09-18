"""Centralized figure styling shared by every plot operator in this package."""

from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field

#: Matplotlib ``tab10`` colours, named once each: the palettes below are keyed by
#: meaning, and several of them reach for the same colour, so spelling a colour
#: at every use site is how two tables silently drift to different blues.
BLUE = "tab:blue"
ORANGE = "tab:orange"
GREEN = "tab:green"
RED = "tab:red"
PURPLE = "tab:purple"
BROWN = "tab:brown"
PINK = "tab:pink"
GRAY = "tab:gray"
OLIVE = "tab:olive"
CYAN = "tab:cyan"


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
    #: The single legend shared by every field panel, placed below the panels.
    shared_legend_fontsize: int = 8
    shared_legend_max_columns: int = 3
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
            "config": (RED, "-"),
            "measured": (CYAN, "--"),
        }
    )
    interface_linewidth: float = 1.5
    colormap_interface_time: str = "viridis"

    # Contact-angle overlay: colour per wetting-band threshold, linestyle per
    # contact line (the solver measures each side's bounds separately), colour
    # per modified wall region, and the angle glyphs.
    wetting_band_colors: dict[str, str] = field(
        default_factory=lambda: {
            "upper": ORANGE,
            "lower": PURPLE,
        }
    )
    wetting_band_linestyles: dict[str, str] = field(
        default_factory=lambda: {
            "left": ":",
            "right": "-.",
        }
    )
    wetting_region_colors: dict[str, str] = field(
        default_factory=lambda: {
            "left": GREEN,
            "right": PINK,
        }
    )
    wetting_marker_linewidth: float = 4.0
    contact_angle_color: str = "gold"
    contact_angle_linewidth: float = 1.5
    contact_angle_fontsize: int = 8
    #: Tangent length as a fraction of the shorter domain side.
    contact_angle_length_fraction: float = 0.25

    colors: dict[str, str] = field(
        default_factory=lambda: {
            "max_velocity": BLUE,
            "density_ratio": ORANGE,
            "avg_density": GREEN,
            "total_mass": OLIVE,
            "contact_angle_left": PURPLE,
            "contact_angle_right": RED,
            "contact_line_speed_left": BROWN,
            "contact_line_speed_right": PINK,
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
    "Pinning": BLUE,
    "Dissipative": GREEN,
    "Capillary": RED,
    "Steady": ORANGE,
    "unknown": GRAY,
}
