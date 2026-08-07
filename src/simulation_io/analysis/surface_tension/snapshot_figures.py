"""Per-droplet snapshot figures for the Young-Laplace surface-tension sweep.

One figure per calibration droplet, written to
``<run_dir>/surface_tension/snapshots/R_<R>.png``. Each figure shows the
equilibrated field three ways -- density, bulk pressure ``p_0(rho)`` and the
full normal pressure -- with markers on every panel at the exact pixels whose
values enter the Laplace jump. Seeing them makes it obvious whether the
"vapour" corners really sit in the bulk rather than in the diffuse interface,
which is the failure mode the fit alone cannot show.

The panels are the registered plotting operators (``density``, ``pressure``,
``pressure_total``) called directly, so the pressure shown here is by
construction the pressure the calibration measured.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from matplotlib.patches import Circle
from src.simulation_io.analysis.surface_tension.surface_tension import sample_points
from src.simulation_io.plotting.density import DensityPlotOperator
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE
from src.simulation_io.plotting.pressure import BulkPressurePlotOperator
from src.simulation_io.plotting.pressure import TotalPressurePlotOperator

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    import matplotlib.axes
    from src.config import SimulationConfig
    from src.simulation_io.plotting.base import PlotOperator

_PANEL_FIGSIZE = (5.0, 4.5)
_MARKER_SIZE = 9
_MARKER_EDGE_WIDTH = 1.5


def save_snapshot_figures(
    config: SimulationConfig,
    out_dir: Path,
    radii: np.ndarray,
    delta_p: np.ndarray,
    densities: Sequence[np.ndarray],
    *,
    timestep: int,
) -> None:
    """Write one three-panel figure per droplet into *out_dir*.

    Args:
        config: The calibration config the droplets were equilibrated with --
            all-periodic, no forces. The pressure panels build their
            differential operators from it, so its boundary conditions must be
            the ones the sweep ran under, not the parent run's.
        out_dir: Directory for the figures; created if absent.
        radii: Prescribed droplet radii, in lattice units.
        delta_p: Measured Laplace jump per radius, same order as *radii*.
        densities: Equilibrated 2-D ``(nx, ny)`` density field per radius.
        timestep: Iteration count the droplets were equilibrated for; shown in
            the panel titles.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    # Built once for the whole sweep: each operator caches its EOS parameters
    # and differential closures on first use, so the lattice and diff-op
    # construction is paid once rather than per radius.
    operators: list[PlotOperator] = [
        DensityPlotOperator(config),
        BulkPressurePlotOperator(config),
        TotalPressurePlotOperator(config),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    for radius, jump, rho_2d in zip(radii, delta_p, densities, strict=True):
        data = {"rho": np.asarray(rho_2d)[:, :, None, None, None]}
        panels = [op for op in operators if op.is_available(data)]
        if not panels:
            continue
        nx, ny = data["rho"].shape[0], data["rho"].shape[1]

        fig, axes = plt.subplots(1, len(panels), figsize=(_PANEL_FIGSIZE[0] * len(panels), _PANEL_FIGSIZE[1]))
        for ax, op in zip(np.atleast_1d(axes), panels, strict=True):
            op(ax, data, timestep)
            _mark_sample_points(ax, nx, ny, float(radius))

        fig.suptitle(
            f"Calibration droplet R = {float(radius):.2f}   dP = {float(jump):.6g}",
            fontsize=DEFAULT_STYLE.suptitle_fontsize,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"R_{float(radius):.2f}.png", dpi=DEFAULT_STYLE.dpi, bbox_inches="tight")
        plt.close(fig)


def _mark_sample_points(
    ax: matplotlib.axes.Axes,
    nx: int,
    ny: int,
    radius: float,
) -> None:
    """Overlay the pixels entering the Laplace jump on an already-rendered panel.

    The panel operators transpose their field and draw it with
    ``origin="lower"``, so array index ``(i, j)`` lands at data coordinates
    ``x=i, y=j`` and the sample indices can be plotted unchanged.
    """
    inside, outside = sample_points(nx, ny)

    # The prescribed radius, not one re-measured from the field: that is the R
    # the Young-Laplace fit uses.
    ax.add_patch(Circle(inside, radius, fill=False, linestyle="--", edgecolor="black", linewidth=1.0, alpha=0.6))

    ax.plot(
        [inside[0]],
        [inside[1]],
        linestyle="none",
        marker="o",
        markersize=_MARKER_SIZE,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=_MARKER_EDGE_WIDTH,
        label="p_inside",
    )
    ax.plot(
        [point[0] for point in outside],
        [point[1] for point in outside],
        linestyle="none",
        marker="x",
        markersize=_MARKER_SIZE,
        markeredgecolor="black",
        markeredgewidth=_MARKER_EDGE_WIDTH,
        label="p_outside",
    )
    ax.legend(fontsize=DEFAULT_STYLE.panel_legend_fontsize, loc="upper right")
