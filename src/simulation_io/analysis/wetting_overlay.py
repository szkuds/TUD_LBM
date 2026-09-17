"""Geometry for the contact-angle overlay: wetting band cells and angle glyphs.

Two things are drawn for a wetting run, and both come from the wetting
operators themselves rather than from a post-processing copy of them:

Wall band
    The ghost-row cells the wetting BC actually modifies, from
    :func:`~src.operators.wetting._apply_edge.wetting_edge_regions` run on the
    same stencil padding the wetting differential operators use, together with
    the ``rho_lower``/``rho_upper`` thresholds of
    :func:`~src.operators.wetting._wetting_modification.wetting_band_bounds`.
    A second definition of the band here could silently disagree with the
    solver, which is exactly what the overlay exists to check. The wall cells
    always use the config densities, because those are what the solver bakes in.

Band contours
    The same thresholds evaluated per interface marker of
    :mod:`.interface_contour`: ``config`` from the prescribed ``(rho_l, rho_v)``
    — the band the solver actually applies — and ``measured`` from the
    snapshot's bulk-phase medians, the band it *would* apply at the densities
    the run has drifted to. Selected by ``interface_levels``, like the contour.

Contact angles
    The ``ca_*``/``cll_*`` values the simulation saved into the snapshot, or —
    when a snapshot lacks them — the same
    :func:`~src.operators.wetting._contact_angle.compute_contact_angle` and
    :func:`~src.operators.wetting._contact_line.compute_contact_line_location`
    the step runs, evaluated eagerly.

Geometry is built in the wall-aligned canonical frame of
:func:`~src.operators.wetting._canonical_view.to_canonical` (tangential, normal;
wall at normal ``-0.5``) and every vertex is mapped to panel coordinates by
:func:`to_physical`, so the mirrored top/right walls need no sign logic.

Imports no matplotlib, like :mod:`.interface_contour`. The JAX-backed wetting
operators are imported function-locally.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal
import numpy as np
from src.simulation_io.analysis.droplet_metrics._snapshot import to_canonical_2d
from src.simulation_io.analysis.interface_contour import LEVEL_CONFIG
from src.simulation_io.analysis.interface_contour import config_rho_mean
from src.simulation_io.analysis.interface_contour import level_densities

if TYPE_CHECKING:
    from collections.abc import Mapping
    from numpy.typing import ArrayLike
    from src.config import SimulationConfig

Side = Literal["left", "right"]

#: Canonical normal coordinate of the solid surface: half a cell below the wall row.
WALL_NORMAL = -0.5

#: Fluid rows the contact-angle measurement reads (wall row and the two above it).
_MEASURED_ROWS = (0, 1, 2)
_MIN_CROSSINGS = 2
_CONTACT_KEYS = ("ca_left", "ca_right", "cll_left", "cll_right")

#: Label distance from the contact line, in arc radii — clear of the arc.
_LABEL_RADIUS = 1.8


@dataclass(frozen=True)
class WallBand:
    """The ghost-row cells one wetting wall modifies, indexed along that wall."""

    edge: str
    left_cells: np.ndarray
    right_cells: np.ndarray
    split: float | None
    rho_lower: float
    rho_upper: float


@dataclass(frozen=True)
class BandLevel:
    """The wetting-band thresholds at one interface marker's densities."""

    level: str
    rho_lower: float
    rho_upper: float


@dataclass(frozen=True)
class ContactAngles:
    """Dispersed-phase contact angles (degrees) and contact-line positions at one wall."""

    edge: str
    theta_left: float
    theta_right: float
    cll_left: float
    cll_right: float


@dataclass(frozen=True)
class AngleGlyph:
    """Panel-coordinate polylines and label anchor for one contact angle."""

    tangent: np.ndarray
    arc: np.ndarray
    label_xy: tuple[float, float]
    theta: float


def wetting_edge(config: SimulationConfig) -> str | None:
    """The wall contact angles are measured at, or ``None`` for a run without one."""
    from src.operators.wetting._edge_config import first_wetting_edge

    return first_wetting_edge(config.bc_config)


def band_level(level: str, config: SimulationConfig, rho_2d: np.ndarray) -> BandLevel | None:
    """The wetting-band thresholds at interface marker *level*, or ``None`` if unavailable.

    The solver's own :func:`~src.operators.wetting._wetting_modification.wetting_band_bounds`
    applied to the marker's ``(dense, light)`` pair from
    :func:`~src.simulation_io.analysis.interface_contour.level_densities`.
    """
    phases = level_densities(level, config, rho_2d)
    if phases is None:
        return None
    from src.operators.wetting._wetting_modification import wetting_band_bounds

    lower, upper = wetting_band_bounds(*phases)
    return BandLevel(level, float(np.asarray(lower)), float(np.asarray(upper)))


def band_levels(levels: tuple[str, ...], config: SimulationConfig, rho_2d: np.ndarray) -> list[BandLevel]:
    """:func:`band_level` for every marker in *levels* this snapshot can supply."""
    return [band for level in levels if (band := band_level(level, config, rho_2d)) is not None]


def wall_bands(rho_2d: np.ndarray, config: SimulationConfig) -> list[WallBand]:
    """The modified ghost-row cells of every wetting wall in *config*.

    Empty when the config has no wetting wall or lacks ``rho_l``/``rho_v``.
    """
    bounds = band_level(LEVEL_CONFIG, config, rho_2d)
    if bounds is None or config.rho_l is None or config.rho_v is None or not config.bc_config:
        return []
    rho_l, rho_v = float(config.rho_l), float(config.rho_v)

    import jax.numpy as jnp
    from src.operators.differential._pad_utils import _apply_stencil_padding
    from src.operators.differential._pad_utils import determine_pad_modes
    from src.operators.wetting._apply_edge import wetting_edge_regions
    from src.operators.wetting._edge_config import _resolve_wetting_edges

    grid_padded = _apply_stencil_padding(jnp.asarray(rho_2d, dtype=float), tuple(determine_pad_modes(config.bc_config)))
    bands: list[WallBand] = []
    for edge, perp_start_periodic, perp_end_periodic in _resolve_wetting_edges(config.bc_config):
        is_left, is_right, centre = wetting_edge_regions(
            grid_padded, edge, perp_start_periodic, perp_end_periodic, rho_l, rho_v
        )
        left_cells = np.flatnonzero(np.asarray(is_left))
        right_cells = np.flatnonzero(np.asarray(is_right))
        split = float(centre) if left_cells.size or right_cells.size else None
        bands.append(WallBand(edge, left_cells, right_cells, split, bounds.rho_lower, bounds.rho_upper))
    return bands


def _crossings(row: np.ndarray, rho_mean: float) -> int:
    """Number of ``rho_mean`` crossings along one wall-parallel row."""
    return int(np.count_nonzero(np.abs(np.diff((row < rho_mean).astype(int))) == 1))


def contact_angles(
    data: Mapping[str, np.ndarray],
    rho_2d: np.ndarray,
    config: SimulationConfig,
) -> ContactAngles | None:
    """Contact angles at the measurement wall, preferring the snapshot's saved values.

    Returns ``None`` when there is no wetting wall, no ``rho_mean``, or the rows
    the measurement reads do not each cross ``rho_mean`` twice — a detached or
    absent inclusion, for which the solver's static-shape crossing search
    collapses to index 0 and reports a meaningless angle.
    """
    edge = wetting_edge(config)
    rho_mean = config_rho_mean(config)
    if edge is None or rho_mean is None:
        return None
    canonical = to_canonical_2d(rho_2d, edge)
    if canonical.shape[1] <= max(_MEASURED_ROWS):
        return None
    if any(_crossings(canonical[:, row], rho_mean) < _MIN_CROSSINGS for row in _MEASURED_ROWS):
        return None

    if all(key in data for key in _CONTACT_KEYS):
        ca_l, ca_r, cll_l, cll_r = (float(np.asarray(data[key]).squeeze()) for key in _CONTACT_KEYS)
    else:
        ca_l, ca_r, cll_l, cll_r = _recompute(rho_2d, rho_mean, edge)
    return ContactAngles(edge, ca_l, ca_r, cll_l, cll_r)


def _recompute(rho_2d: np.ndarray, rho_mean: float, edge: str) -> tuple[float, float, float, float]:
    """Run the step's own contact-angle and contact-line operators on one field."""
    import jax.numpy as jnp
    from src.operators.wetting._contact_angle import compute_contact_angle
    from src.operators.wetting._contact_line import compute_contact_line_location

    rho = jnp.asarray(rho_2d, dtype=float)[:, :, None, None, None]
    mean = jnp.asarray(rho_mean)
    ca_l, ca_r = compute_contact_angle(rho, mean, edge=edge)
    cll_l, cll_r = compute_contact_line_location(rho, ca_l, ca_r, mean, edge=edge)
    return float(ca_l), float(ca_r), float(cll_l), float(cll_r)


def to_physical(
    tangential: ArrayLike,
    normal: ArrayLike,
    edge: str,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Map canonical ``(tangential, normal)`` coordinates to panel ``(x, y)``.

    The inverse of :func:`~src.operators.wetting._canonical_view.to_canonical`
    on coordinates, in the cell-centre frame of ``imshow(rho.T, origin="lower")``.

    Args:
        tangential: Position along the wall.
        normal: Distance from the wall row into the fluid.
        edge: The wall — ``"bottom"``, ``"top"``, ``"left"`` or ``"right"``.
        shape: Physical ``(nx, ny)``.
    """
    nx, ny = shape
    t = np.asarray(tangential, dtype=float)
    n = np.asarray(normal, dtype=float)
    if edge == "bottom":
        return t, n
    if edge == "top":
        return t, (ny - 1) - n
    if edge == "left":
        return n, t
    if edge == "right":
        return (nx - 1) - n, t
    msg = f"Unknown wall edge {edge!r}."
    raise ValueError(msg)


def band_cell_segments(band: WallBand, side: Side, shape: tuple[int, int]) -> list[np.ndarray]:
    """One ``(2, 2)`` segment per modified cell, lying on the solid surface of *band*'s wall."""
    cells = band.left_cells if side == "left" else band.right_cells
    segments = []
    for cell in cells:
        x, y = to_physical([cell - 0.5, cell + 0.5], [WALL_NORMAL, WALL_NORMAL], band.edge, shape)
        segments.append(np.column_stack((x, y)))
    return segments


def split_tick(band: WallBand, shape: tuple[int, int], height: float) -> np.ndarray | None:
    """A ``(2, 2)`` segment normal to the wall at the band's left/right split."""
    if band.split is None:
        return None
    x, y = to_physical([band.split, band.split], [WALL_NORMAL, WALL_NORMAL + height], band.edge, shape)
    return np.column_stack((x, y))


def angle_glyph(
    angles: ContactAngles,
    side: Side,
    shape: tuple[int, int],
    length: float,
    arc_points: int = 24,
) -> AngleGlyph:
    """Tangent line, wall arc and label anchor for one contact angle.

    Both angles subtend the dispersed phase, which lies between the two contact
    lines, so measured from the wall direction pointing into it the tangent is
    ``(cos θ, sin θ)`` at the left line and ``(-cos θ, sin θ)`` at the right.
    """
    sign = 1.0 if side == "left" else -1.0
    theta_deg = angles.theta_left if side == "left" else angles.theta_right
    cll = angles.cll_left if side == "left" else angles.cll_right
    theta = math.radians(theta_deg)
    radius = 0.5 * length

    tangent_t = [cll, cll + sign * length * math.cos(theta)]
    tangent_n = [WALL_NORMAL, WALL_NORMAL + length * math.sin(theta)]
    sweep = np.linspace(0.0, theta, arc_points)
    arc_t = cll + sign * radius * np.cos(sweep)
    arc_n = WALL_NORMAL + radius * np.sin(sweep)
    label_t = cll + sign * _LABEL_RADIUS * radius * math.cos(theta / 2.0)
    label_n = WALL_NORMAL + _LABEL_RADIUS * radius * math.sin(theta / 2.0)

    tx, ty = to_physical(tangent_t, tangent_n, angles.edge, shape)
    ax_, ay = to_physical(arc_t, arc_n, angles.edge, shape)
    lx, ly = to_physical(label_t, label_n, angles.edge, shape)
    return AngleGlyph(
        tangent=np.column_stack((tx, ty)),
        arc=np.column_stack((ax_, ay)),
        label_xy=(float(lx), float(ly)),
        theta=theta_deg,
    )
