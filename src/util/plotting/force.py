"""Force magnitude plot operators."""

from __future__ import annotations
import matplotlib.axes
import numpy as np
from registry import plotting_operator
from util.plotting.base import PlotOperator


class _BaseForceOperator(PlotOperator):
    field_name: str = "force"
    title: str = "Force"

    def is_available(self, data: dict[str, np.ndarray]) -> bool:
        return self.field_name in data

    def __call__(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        timestep: int,
    ) -> None:
        force = np.asarray(data[self.field_name])
        fx = force[..., 0, 0].T
        fy = force[..., 0, 1].T
        mag = np.sqrt(fx**2 + fy**2)
        im = ax.imshow(mag, origin="lower", aspect="equal", cmap="cividis")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|F|")

        ny, nx = mag.shape
        skip = max(1, min(nx, ny) // 10)
        y, x = np.mgrid[0:ny:skip, 0:nx:skip]
        ax.quiver(
            x,
            y,
            fx[::skip, ::skip],
            fy[::skip, ::skip],
            scale_units="xy",
            scale=None,
            angles="xy",
            color="white",
            alpha=0.7,
        )
        ax.set_title(f"{self.title}  t={timestep}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")


@plotting_operator(name="force")
class ForcePlotOperator(_BaseForceOperator):
    """Render the saved total force field."""

    name = "force"
    field_name = "force"
    title = "Total force"


@plotting_operator(name="force_ext")
class ExternalForcePlotOperator(_BaseForceOperator):
    """Render the saved external force field."""

    name = "force_ext"
    field_name = "force_ext"
    title = "External force"
