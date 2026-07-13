"""Reynolds-number history plot: Re = |u|*L/nu vs. timestep."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from tud_lbm.io.plotting._analysis_common import _BaseAnalysisPlot
from tud_lbm.io.plotting._analysis_common import _extract_u_mag_2d
from tud_lbm.io.plotting._analysis_common import _load_timesteps
from tud_lbm.io.plotting.figure_config import DEFAULT_STYLE
from tud_lbm.io.plotting.simulation_csv import _resolve_r_zero
from tud_lbm.registry import comparison_operator

if TYPE_CHECKING:
    from pathlib import Path


@comparison_operator(name="reynolds_number")
class ReynoldsNumberPlot(_BaseAnalysisPlot):
    """Plot the characteristic droplet Reynolds number over time.

    Re = max(|u|) * L / nu, with L = droplet diameter (2*R0, see
    :func:`tud_lbm.io.plotting.simulation_csv._resolve_r_zero`) and nu the
    lattice kinematic viscosity. Requires a config so R0 and nu can be
    resolved; falls back to an empty series without one.
    """

    name = "reynolds_number"
    title = "Reynolds number vs timestep"
    ylabel = "Re"
    color = DEFAULT_STYLE.colors["reynolds_number"]
    required_keys = ("u",)

    def compute(self, files: list[Path]) -> dict[str, np.ndarray]:
        """Compute Re = max(|u|) * L / nu for each timestep file."""
        iters, snapshots = _load_timesteps(files, ("u",))
        if self.config is None or len(iters) == 0:
            return {"iters": np.asarray([], dtype=int), "values": np.asarray([], dtype=float)}

        nu = (float(self.config.tau) - 0.5) / 3.0
        length = 2.0 * _resolve_r_zero(self.config).value
        vals = np.asarray(
            [float(np.max(_extract_u_mag_2d(snap["u"])) * length / nu) for snap in snapshots],
            dtype=float,
        )
        return {"iters": iters, "values": vals}
