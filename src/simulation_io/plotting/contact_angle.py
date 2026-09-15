"""The ``contact_angle`` overlay: contact angles and the wetting band on any field panel.

Draws, for a run with a ``"wetting"`` wall:

- the ``rho_upper`` / ``rho_lower`` iso-contours bounding the density band in
  which the wetting BC modifies the ghost row;
- the ghost-row cells it actually modifies, as thick segments on the solid
  surface coloured by left/right contact-line region, with a tick at the split;
- at each contact line, the tangent at the contact angle, an arc from the wall
  and a ``θ=…°`` label.

Overlay-only: it has no panel of its own and is hidden from field selection. The
geometry lives in :mod:`src.simulation_io.analysis.wetting_overlay`, which takes
it from the wetting operators themselves.

Like the interface contour this is a spatial map, so its lines are exempt from
the scatter-only rule for data-series plots.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from matplotlib.collections import LineCollection
from src.registry import plotting_operator
from src.simulation_io.analysis.droplet_metrics import extract_rho_2d
from src.simulation_io.analysis.interface_contour import interface_lines
from src.simulation_io.analysis.wetting_overlay import angle_glyph
from src.simulation_io.analysis.wetting_overlay import band_bounds
from src.simulation_io.analysis.wetting_overlay import band_cell_segments
from src.simulation_io.analysis.wetting_overlay import contact_angles
from src.simulation_io.analysis.wetting_overlay import split_tick
from src.simulation_io.analysis.wetting_overlay import wall_bands
from src.simulation_io.analysis.wetting_overlay import wetting_edge
from src.simulation_io.plotting.base import PlotOperator
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE

if TYPE_CHECKING:
    import matplotlib.axes
    import numpy as np
    from src.config import SimulationConfig
    from src.simulation_io.analysis.wetting_overlay import Side

_SIDES: tuple[Side, Side] = ("left", "right")

#: Split-tick height, in cells normal to the wall.
_SPLIT_TICK_HEIGHT = 2.0


@plotting_operator(name="contact_angle")
class ContactAnglePlotOperator(PlotOperator):
    """Contact angles and the modified wetting band, drawn on top of a field panel."""

    name = "contact_angle"
    opt_in = True
    supports_overlay = True
    overlay_only = True
    overlay_label = "contact angles and wetting band"

    @classmethod
    def overlay_prompt_default(cls, config: SimulationConfig) -> bool | None:
        """Offer the overlay, defaulting to yes, only for a run with a wetting wall."""
        return True if wetting_edge(config) is not None else None

    def is_available(self, data: dict[str, np.ndarray]) -> bool:
        """Needs a density field and a wetting wall."""
        return "rho" in data and wetting_edge(self.config) is not None

    def __call__(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        timestep: int,
    ) -> None:
        """Overlay-only: there is no standalone contact-angle panel."""
        msg = f"Plot operator {self.name!r} is overlay-only; name it in overlay_fields or --overlay."
        raise NotImplementedError(msg)

    def draw_overlay(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        timestep: int,  # noqa: ARG002
    ) -> None:
        """Draw the band contours, modified wall cells and contact-angle glyphs.

        Artists are labelled but no legend is drawn: the figure gathers the
        labels into one shared legend.
        """
        ax.autoscale(enable=False)
        rho_2d = extract_rho_2d(data["rho"])
        shape = (int(rho_2d.shape[0]), int(rho_2d.shape[1]))

        self._draw_band_contours(ax, rho_2d)
        self._draw_wall_cells(ax, rho_2d, shape)
        self._draw_angles(ax, data, rho_2d, shape)

    def _draw_band_contours(self, ax: matplotlib.axes.Axes, rho_2d: np.ndarray) -> None:
        bounds = band_bounds(self.config)
        if bounds is None:
            return
        for key, value in zip(("lower", "upper"), bounds, strict=True):
            color, linestyle = DEFAULT_STYLE.wetting_band_styles[key]
            ax.add_collection(
                LineCollection(
                    interface_lines(rho_2d, value),
                    colors=color,
                    linestyles=linestyle,
                    linewidths=DEFAULT_STYLE.interface_linewidth,
                    label=f"ρ_{key}={value:.4g}",
                )
            )

    def _draw_wall_cells(self, ax: matplotlib.axes.Axes, rho_2d: np.ndarray, shape: tuple[int, int]) -> None:
        for band in wall_bands(rho_2d, self.config):
            for side in _SIDES:
                segments = band_cell_segments(band, side, shape)
                if not segments:
                    continue
                ax.add_collection(
                    LineCollection(
                        segments,
                        colors=DEFAULT_STYLE.wetting_region_colors[side],
                        linewidths=DEFAULT_STYLE.wetting_marker_linewidth,
                        capstyle="butt",
                        clip_on=False,
                        label=f"{band.edge} wetting band ({side})",
                    )
                )
            tick = split_tick(band, shape, _SPLIT_TICK_HEIGHT)
            if tick is not None:
                ax.plot(tick[:, 0], tick[:, 1], color="black", linewidth=1.0, clip_on=False)

    def _draw_angles(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        rho_2d: np.ndarray,
        shape: tuple[int, int],
    ) -> None:
        angles = contact_angles(data, rho_2d, self.config)
        if angles is None:
            return
        length = max(3.0, DEFAULT_STYLE.contact_angle_length_fraction * min(shape))
        color = DEFAULT_STYLE.contact_angle_color
        for side in _SIDES:
            glyph = angle_glyph(angles, side, shape, length)
            for line in (glyph.tangent, glyph.arc):
                ax.plot(line[:, 0], line[:, 1], color=color, linewidth=DEFAULT_STYLE.contact_angle_linewidth)
            ax.annotate(
                f"θ={glyph.theta:.1f}°",
                xy=glyph.label_xy,
                color=color,
                fontsize=DEFAULT_STYLE.contact_angle_fontsize,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.55, "linewidth": 0},
            )
