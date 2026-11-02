"""Density field plot operator."""

from __future__ import annotations
import matplotlib.axes
import matplotlib.colors
import numpy as np
from src.config.multiphase import multiphase_params
from src.protocols import PlotOperator
from src.registry import plotting_operator
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE

# Density ratio threshold for using log scale
DENSITY_RATIO_THRESHOLD = 100


@plotting_operator(name="density")
class DensityPlotOperator(PlotOperator):
    """Render the density field as a 2-D colour map."""

    name = "density"

    def is_available(self, data: dict[str, np.ndarray]) -> bool:
        """Check if density data is available for rendering."""
        return "rho" in data

    def __call__(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        timestep: int,
    ) -> None:
        """Render the density field as a 2D color map."""
        rho = np.asarray(data["rho"])[:, :, 0, 0, 0].T
        use_log_scale = False
        params = multiphase_params(self.config)
        if params is not None:
            use_log_scale = (params.rho_l / params.rho_v) > DENSITY_RATIO_THRESHOLD

        if use_log_scale:
            positive = rho[rho > 0]
            floor = float(np.min(positive) * 1e-10) if positive.size else 1e-30
            plot_data = np.maximum(rho, floor)
            im = ax.imshow(
                plot_data,
                origin="lower",
                aspect="equal",
                cmap=DEFAULT_STYLE.colormap_density,
                norm=matplotlib.colors.LogNorm(),
            )
            ax.set_title(f"Density (log)  t={timestep}")
        else:
            im = ax.imshow(rho, origin="lower", aspect="equal", cmap=DEFAULT_STYLE.colormap_density)
            ax.set_title(f"Density  t={timestep}")

        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Density")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
