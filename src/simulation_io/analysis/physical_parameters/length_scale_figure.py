"""Diagnostic figure for the region behind the Bo/Oh length scale.

``L_eff = sqrt(A/pi)`` and ``Bo ∝ L^2 ∝ A``, so an area error passes straight
into every dimensionless number. The area itself is never shown anywhere, which
makes an overestimate invisible: the counted region is thresholded at
``rho_mean`` on the *setup* field, while the droplet the run actually evolves
may be a different size.

This module renders them side by side -- the region as counted, and the region
present in the run's first and last snapshot -- each with its own measured
``rho_mean``, area, ``L_eff`` and ``Bo``. Whether the counted region is too big,
and by how much, is then something you can see rather than infer.

The panels are the registered ``density`` plotting operator called directly, so
the field under the mask is by construction the field the rest of the plotting
layer would show.

Every matplotlib and plotting-layer import is function-local: this module is
re-exported from the package ``__init__``, which ``SimulationIO`` imports at the
start of every run.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import TYPE_CHECKING
from typing import NamedTuple
import numpy as np
from src.config.run_config import ANALYSIS_DIRNAME
from src.config.run_config import DATA_DIRNAME
from src.config.run_config import PLOTS_DIRNAME
from src.config.run_config import SNAPSHOT_GLOB
from src.simulation_io.analysis.droplet_metrics._snapshot import extract_rho_2d
from src.simulation_io.analysis.droplet_metrics._snapshot import parse_timestep_from_path
from src.simulation_io.analysis.physical_parameters.physical_parameters import _get_droplet_area
from src.simulation_io.analysis.physical_parameters.physical_parameters import _load_init_rho
from src.simulation_io.analysis.physical_parameters.physical_parameters import _measure_field
from src.simulation_io.analysis.physical_parameters.physical_parameters import _resolve_buoyancy_delta_rho
from src.simulation_io.analysis.physical_parameters.physical_parameters import _resolve_gravity_inclination
from src.simulation_io.analysis.physical_parameters.physical_parameters import _resolve_gravity_value
from src.simulation_io.analysis.physical_parameters.physical_parameters import _resolve_surface_tension
from src.simulation_io.analysis.physical_parameters.physical_parameters import compute_bond_numbers
from src.simulation_io.analysis.physical_parameters.physical_parameters import inclusion_mask_from_rho

if TYPE_CHECKING:
    from collections.abc import Callable
    import matplotlib.axes
    from src.config.simulation_config import SimulationConfig
    from src.simulation_io.plotting.density import DensityPlotOperator

#: Figure name under ``<out_dir>/plots/analysis/``.
LENGTH_SCALE_FILENAME = "length_scale.png"

_PANEL_FIGSIZE = (5.0, 5.0)
_MASK_ALPHA = 0.28
_MASK_COLOR = "crimson"
_CENTRE_COORDS = 2


class _AnalyticRegion(NamedTuple):
    """The circle-clipped-by-a-wall that ``_get_setup_droplet_area`` integrates."""

    centre_x: float
    centre_y: float
    radius: float
    wall_axis: str  # "x" or "y"
    wall_position: float


class _Panel(NamedTuple):
    """One column of the figure: a field, the region counted on it, and its numbers."""

    title: str
    timestep: int
    rho: np.ndarray | None  # 5-D field for the density operator; None when analytic
    mask: np.ndarray | None  # (nx, ny) boolean region
    analytic: _AnalyticRegion | None
    area: float | None
    rho_mean: float | None
    drho: float | None


def write_length_scale_figure(
    config: SimulationConfig,
    out_dir: str | Path,
    run_dir: str | Path | None = None,
) -> Path | None:
    """Write ``plots/analysis/length_scale.png`` under *out_dir*.

    Args:
        config: The run's configuration. Panel one reproduces exactly what
            :func:`_get_droplet_area` resolves for it.
        out_dir: Directory to write into; ``plots/analysis/`` is created below it.
        run_dir: Directory holding ``data/timestep_*.npz``. Defaults to
            *out_dir*. Without snapshots only the first panel is drawn.

    Returns:
        The figure path, or ``None`` when no panel could be built.
    """
    counted = _counted_panel(config)
    panels = ([] if counted is None else [counted]) + _snapshot_panels(
        Path(run_dir) if run_dir is not None else Path(out_dir)
    )
    if not panels:
        return None

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from src.simulation_io.plotting.density import DensityPlotOperator
    from src.simulation_io.plotting.figure_config import DEFAULT_STYLE

    operator = DensityPlotOperator(config)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(_PANEL_FIGSIZE[0] * len(panels), _PANEL_FIGSIZE[1]),
        squeeze=False,
    )
    caption = _caption_builder(config)
    for ax, panel in zip(axes[0], panels, strict=True):
        _render_panel(ax, panel, operator, config)
        ax.set_title(panel.title, fontsize=10)
        ax.set_xlabel(caption(panel), fontsize=9, linespacing=1.4)

    fig.suptitle(
        "Region counted for the Bo/Oh length scale",
        fontsize=DEFAULT_STYLE.suptitle_fontsize,
    )
    fig.tight_layout()

    dest = Path(out_dir) / PLOTS_DIRNAME / ANALYSIS_DIRNAME / LENGTH_SCALE_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=DEFAULT_STYLE.dpi, bbox_inches="tight")
    plt.close(fig)
    return dest


# --- Panel construction ------------------------------------------------------


def _counted_panel(config: SimulationConfig) -> _Panel | None:
    """The region ``_get_droplet_area`` actually counts, however it resolves it."""
    resolved = _get_droplet_area(config)
    area = None if resolved is None else resolved[0]
    buoyancy = _resolve_buoyancy_delta_rho(config)
    drho = None if buoyancy is None else buoyancy[0]

    field = _load_init_rho(config) if config.init_type == "init_from_file" else None
    if field is not None:
        return _Panel(
            title="as counted  (init NPZ)",
            timestep=0,
            rho=field.rho,
            mask=inclusion_mask_from_rho(field.rho, field.rho_mean),
            analytic=None,
            area=area,
            rho_mean=field.rho_mean,
            drho=drho,
        )

    region = _analytic_region(config)
    if region is None and area is None:
        return None
    return _Panel(
        title="as counted  (init geometry, analytic)",
        timestep=0,
        rho=None,
        mask=None,
        analytic=region,
        area=area,
        rho_mean=None,
        drho=drho,
    )


def _snapshot_panels(run_dir: Path) -> list[_Panel]:
    """Panels for the first and last saved snapshot, or ``[]`` when there are none."""
    data_dir = run_dir / DATA_DIRNAME
    if not data_dir.is_dir():
        return []
    files = sorted(data_dir.glob(SNAPSHOT_GLOB), key=parse_timestep_from_path)
    if not files:
        return []
    # One file means first and last coincide; show it once.
    chosen = [files[0]] if len(files) == 1 else [files[0], files[-1]]
    panels = (_snapshot_panel(path) for path in chosen)
    return [panel for panel in panels if panel is not None]


def _snapshot_panel(path: Path) -> _Panel | None:
    """Measure one snapshot with its own threshold, so drift over the run is visible."""
    try:
        with np.load(path) as data:
            if "rho" not in data:
                return None
            rho_2d = extract_rho_2d(np.asarray(data["rho"]))
    except (KeyError, OSError, TypeError, ValueError):
        return None

    rho = np.asarray(rho_2d)[:, :, None, None, None]
    field = _measure_field(rho)
    if field is None:
        return None
    mask = inclusion_mask_from_rho(rho, field.rho_mean)
    return _Panel(
        title=f"snapshot  t={parse_timestep_from_path(path)}",
        timestep=parse_timestep_from_path(path),
        rho=rho,
        mask=mask,
        analytic=None,
        area=None if mask is None else float(np.count_nonzero(mask)),
        rho_mean=field.rho_mean,
        drho=field.drho,
    )


def _analytic_region(config: SimulationConfig) -> _AnalyticRegion | None:
    """Rebuild the circle and clipping wall that the analytic area integrates.

    Mirrors ``_get_setup_droplet_area`` (radius scaled by ``min(nx, ny)``, the
    nearest of the four walls) and the initialiser's integer centre rounding in
    ``operators/initialise/_multiphase_bubbles.py``.
    """
    init = config.initialisation
    if not init or not isinstance(init, dict):
        return None
    centres = init.get("centres", [])
    radii = init.get("radii", [])
    if not centres or not radii or len(centres[0]) != _CENTRE_COORDS:
        return None

    try:
        nx = float(config.grid_shape[0])
        ny = float(config.grid_shape[1])
        fx, fy = float(centres[0][0]), float(centres[0][1])
        radius = float(radii[0]) * min(nx, ny)
    except (IndexError, TypeError, ValueError):
        return None

    # The four wall distances in the order (axis, position) so the nearest one
    # -- the only segment the area subtracts -- can be drawn.
    walls = (
        (fx * nx, "x", 0.0),
        ((1.0 - fx) * nx, "x", nx - 1.0),
        (fy * ny, "y", 0.0),
        ((1.0 - fy) * ny, "y", ny - 1.0),
    )
    _distance, axis, position = min(walls, key=lambda wall: wall[0])
    return _AnalyticRegion(
        centre_x=float(round(fx * (nx - 1))),
        centre_y=float(round(fy * (ny - 1))),
        radius=radius,
        wall_axis=axis,
        wall_position=position,
    )


# --- Rendering ---------------------------------------------------------------


def _render_panel(
    ax: matplotlib.axes.Axes,
    panel: _Panel,
    operator: DensityPlotOperator,
    config: SimulationConfig,
) -> None:
    """Draw one panel: the density field with its mask, or the analytic circle."""
    if panel.rho is not None:
        operator(ax, {"rho": panel.rho}, panel.timestep)
        if panel.mask is not None:
            _overlay_mask(ax, panel.mask)
        return
    _draw_analytic(ax, panel.analytic, config)


def _overlay_mask(ax: matplotlib.axes.Axes, mask: np.ndarray) -> None:
    """Shade and outline the counted cells over an already-rendered field.

    ``DensityPlotOperator`` transposes its field and draws it with
    ``origin="lower"``, so the mask must be transposed the same way or it lands
    rotated relative to the density beneath it.
    """
    shown = np.asarray(mask, dtype=float).T
    ax.contourf(shown, levels=[0.5, 1.5], colors=[_MASK_COLOR], alpha=_MASK_ALPHA)
    ax.contour(shown, levels=[0.5], colors=[_MASK_COLOR], linewidths=1.2)


def _draw_analytic(
    ax: matplotlib.axes.Axes,
    region: _AnalyticRegion | None,
    config: SimulationConfig,
) -> None:
    """Draw the analytic circle and its clipping wall on an empty domain.

    Deliberately not rasterised into a mask: the analytic branch integrates a
    formula, and a cell count drawn beside it would not be the area reported.
    """
    from matplotlib.patches import Circle

    nx, ny = int(config.grid_shape[0]), int(config.grid_shape[1])
    ax.set_xlim(0, nx - 1)
    ax.set_ylim(0, ny - 1)
    ax.set_aspect("equal")
    ax.set_ylabel("y")

    if region is None:
        ax.text(0.5, 0.5, "no init geometry", transform=ax.transAxes, ha="center", va="center")
        return

    ax.add_patch(
        Circle(
            (region.centre_x, region.centre_y),
            region.radius,
            facecolor=_MASK_COLOR,
            alpha=_MASK_ALPHA,
            edgecolor=_MASK_COLOR,
            linewidth=1.2,
        )
    )
    if region.wall_axis == "y":
        ax.axhline(region.wall_position, color="black", linewidth=1.5, label="clipping wall")
    else:
        ax.axvline(region.wall_position, color="black", linewidth=1.5, label="clipping wall")
    ax.legend(loc="upper right", fontsize=8)


def _caption_builder(config: SimulationConfig) -> Callable[[_Panel], str]:
    """Return a function rendering one panel's numbers, with Bo when resolvable."""
    resolved = _resolve_surface_tension(config)
    gamma = None if resolved is None else resolved[1]
    g_val = _resolve_gravity_value(config)
    angle_deg = _resolve_gravity_inclination(config)

    def caption(panel: _Panel) -> str:
        rows = []
        if panel.rho_mean is not None:
            rows.append(f"rho_mean = {panel.rho_mean:.6g}")
        if panel.area is None:
            rows.append("A = -")
            return "\n".join(rows)

        length = math.sqrt(panel.area / math.pi)
        rows.append(f"A = {panel.area:.6g}")
        rows.append(f"L_eff = {length:.4g}")
        if gamma is not None and g_val is not None and panel.drho is not None:
            bo = compute_bond_numbers(panel.drho, gamma, g_val, length, angle_deg).bo
            rows.append(f"Δρ = {panel.drho:.6g}")
            rows.append(f"Bo = {bo:.6g}")
        return "\n".join(rows)

    return caption
