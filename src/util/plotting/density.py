"""Density field plot operator."""

from __future__ import annotations

import matplotlib.axes
import matplotlib.colors
import numpy as np

from util.plotting.base import PlotOperator
from registry import plotting_operator


@plotting_operator(name="density")
class DensityPlotOperator(PlotOperator):
    """Render the density field as a 2-D colour map."""

    name = "density"

    def is_available(self, data: dict[str, np.ndarray]) -> bool:
        return "rho" in data

    def __call__(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        timestep: int,
    ) -> None:
        rho = np.asarray(data["rho"])[..., 0, 0].T
        use_log_scale = False
        if self.config.get("simulation_type") == "multiphase":
            rho_l = self.config.get("rho_l")
            rho_v = self.config.get("rho_v")
            if rho_l and rho_v and rho_v != 0:
                use_log_scale = (rho_l / rho_v) > 100

        if use_log_scale:
            positive = rho[rho > 0]
            floor = float(np.min(positive) * 1e-10) if positive.size else 1e-30
            plot_data = np.maximum(rho, floor)
            im = ax.imshow(
                plot_data,
                origin="lower",
                aspect="equal",
                cmap="viridis",
                norm=matplotlib.colors.LogNorm(),
            )
            ax.set_title(f"Density (log)  t={timestep}")
        else:
            im = ax.imshow(rho, origin="lower", aspect="equal", cmap="viridis")
            ax.set_title(f"Density  t={timestep}")

        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Density")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

