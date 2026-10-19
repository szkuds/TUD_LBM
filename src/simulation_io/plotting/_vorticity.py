"""Vorticity plot operator."""

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

#: The colour limits are ``±`` this percentile of ``|omega|`` over the fluid, so a
#: single strong vortex at a corner cannot wash out the rest of the street.
_COLOUR_LIMIT_PERCENTILE = 99.0


def compute_vorticity(u: np.ndarray) -> np.ndarray:
    """Out-of-plane vorticity ``omega = du_y/dx - du_x/dy`` of the ``z = 0`` plane.

    Args:
        u: Velocity in the 5D layout, shape ``(nx, ny, nz, 1, d)``.

    Returns:
        ``omega``, shape ``(nx, ny)``; second-order central differences in the
        interior, one-sided at the domain edges.
    """
    ux = np.asarray(u)[:, :, 0, 0, 0]
    uy = np.asarray(u)[:, :, 0, 0, 1]
    return np.gradient(uy, axis=0) - np.gradient(ux, axis=1)


@plotting_operator(name="vorticity")
class VorticityPlotOperator(PlotOperator):
    """Render the out-of-plane vorticity on a diverging, zero-centred colour scale."""

    name = "vorticity"

    #: A diagnostic for flows with structure, not something every run pays a panel for.
    opt_in = True

    def is_available(self, data: dict[str, np.ndarray]) -> bool:
        """Whether the snapshot carries the velocity field ``u``."""
        return "u" in data

    def __call__(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        timestep: int,
    ) -> None:
        """Render the vorticity, with any interior obstacle filled in.

        Args:
            ax: Matplotlib axes object to plot on.
            data: Dictionary containing simulation output data with velocity field.
            timestep: Current simulation timestep for display in title.
        """
        omega = compute_vorticity(data["u"])
        solid = solid_mask_2d(self.config)
        fluid = omega if solid is None else omega[~solid]
        limit = float(np.percentile(np.abs(fluid), _COLOUR_LIMIT_PERCENTILE)) if fluid.size else 0.0
        limit = limit if limit > 0.0 else 1.0

        shown = omega if solid is None else np.ma.masked_where(solid, omega)
        im = ax.imshow(
            shown.T,
            origin="lower",
            aspect="equal",
            cmap=DEFAULT_STYLE.colormap_vorticity,
            vmin=-limit,
            vmax=limit,
        )
        if solid is not None:
            draw_obstacle(ax, solid)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\omega_z$")
        ax.set_title(f"Vorticity  t={timestep}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
