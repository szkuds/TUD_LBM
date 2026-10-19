"""Velocity magnitude and quiver plot operator."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from src.registry import plotting_operator
from src.simulation_io.plotting.base import PlotOperator
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE
from src.simulation_io.plotting.obstacle import draw_obstacle
from src.simulation_io.plotting.obstacle import solid_mask_2d

if TYPE_CHECKING:
    import matplotlib.axes


@plotting_operator(name="velocity")
class VelocityPlotOperator(PlotOperator):
    """Render velocity magnitude with a downsampled vector overlay."""

    name = "velocity"

    def is_available(self, data: dict[str, np.ndarray]) -> bool:
        """Check if velocity data is available in the dataset.

        Args:
            data: Dictionary containing simulation output data.

        Returns:
            True if velocity field 'u' is present in data, False otherwise.
        """
        return "u" in data

    def __call__(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        timestep: int,
    ) -> None:
        """Render velocity magnitude with a downsampled vector overlay.

        Args:
            ax: Matplotlib axes object to plot on.
            data: Dictionary containing simulation output data with velocity field.
            timestep: Current simulation timestep for display in title.
        """
        u = np.asarray(data["u"])
        ux = u[:, :, 0, 0, 0].T
        uy = u[:, :, 0, 0, 1].T
        mag = np.sqrt(ux**2 + uy**2)

        solid = solid_mask_2d(self.config)
        shown = mag if solid is None else np.ma.masked_where(solid.T, mag)
        im = ax.imshow(shown, origin="lower", aspect="equal", cmap=DEFAULT_STYLE.colormap_velocity)
        if solid is not None:
            draw_obstacle(ax, solid)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|u|")

        ny, nx = mag.shape
        if np.any(mag > 0):
            # At most ~10 arrows across the short side and ~25 along the long one.
            skip = max(1, min(nx, ny) // 10, max(nx, ny) // 25)
            y, x = np.mgrid[0:ny:skip, 0:nx:skip]
            ux_s, uy_s = ux[::skip, ::skip], uy[::skip, ::skip]
            if solid is not None:
                fluid = ~solid.T[::skip, ::skip]
                x, y, ux_s, uy_s = x[fluid], y[fluid], ux_s[fluid], uy_s[fluid]
            ax.quiver(
                x,
                y,
                ux_s,
                uy_s,
                scale_units="xy",
                scale=None,
                angles="xy",
                color=DEFAULT_STYLE.quiver_color,
                alpha=DEFAULT_STYLE.quiver_alpha,
            )
        ax.set_title(f"Velocity magnitude  t={timestep}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
