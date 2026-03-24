"""Velocity magnitude and quiver plot operator."""

from __future__ import annotations
import matplotlib.axes
import numpy as np
from registry import plotting_operator
from util.plotting.base import PlotOperator


@plotting_operator(name="velocity")
class VelocityPlotOperator(PlotOperator):
    """Render velocity magnitude with a downsampled vector overlay."""

    name = "velocity"

    def is_available(self, data: dict[str, np.ndarray]) -> bool:
        return "u" in data

    def __call__(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        timestep: int,
    ) -> None:
        u = np.asarray(data["u"])
        ux = u[..., 0, 0].T
        uy = u[..., 0, 1].T
        mag = np.sqrt(ux**2 + uy**2)

        im = ax.imshow(mag, origin="lower", aspect="equal", cmap="plasma")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|u|")

        ny, nx = mag.shape
        if np.any(mag > 0):
            skip = max(1, min(nx, ny) // 10)
            y, x = np.mgrid[0:ny:skip, 0:nx:skip]
            ax.quiver(
                x,
                y,
                ux[::skip, ::skip],
                uy[::skip, ::skip],
                scale_units="xy",
                scale=None,
                angles="xy",
                color="white",
                alpha=0.7,
            )
        ax.set_title(f"Velocity magnitude  t={timestep}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
