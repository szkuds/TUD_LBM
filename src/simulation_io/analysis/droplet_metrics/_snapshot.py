"""Per-snapshot droplet geometry extracted from saved ``.npz`` fields.

Every function here operates on a single snapshot. Time-series assembly and
config-derived scaling live in :mod:`.series` and :mod:`._scales`.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

_NDIM_2D = 2
_NDIM_3D = 3
_NDIM_4D = 4
_NDIM_5D = 5


def parse_timestep(stem: str) -> int | None:
    """Return the trailing integer of a ``timestep_<n>`` stem, or ``None``."""
    try:
        return int(stem.rsplit("_", maxsplit=1)[-1])
    except ValueError:
        return None


def parse_timestep_from_path(path: Path) -> int:
    """Sort key: the snapshot's timestep, or ``-1`` when unparseable."""
    step = parse_timestep(path.stem)
    return -1 if step is None else step


def extract_rho_2d(rho: np.ndarray) -> np.ndarray:
    """Collapse a density array of any supported rank to its 2-D ``(nx, ny)`` slice."""
    arr = np.asarray(rho)
    if arr.ndim >= _NDIM_5D:
        return arr[:, :, 0, 0, 0]
    if arr.ndim == _NDIM_4D:
        return arr[:, :, 0, 0]
    if arr.ndim == _NDIM_3D:
        return arr[:, :, 0]
    if arr.ndim == _NDIM_2D:
        return arr
    msg = f"Unsupported rho shape: {arr.shape}"
    raise ValueError(msg)


def extract_velocity_components_2d(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the 2-D ``(u_x, u_y)`` slices of a velocity array."""
    arr = np.asarray(u)
    if arr.ndim >= _NDIM_5D:
        return arr[:, :, 0, 0, 0], arr[:, :, 0, 0, 1]
    if arr.ndim == _NDIM_4D:
        return arr[:, :, 0, 0], arr[:, :, 0, 1]
    if arr.ndim == _NDIM_3D:
        return arr[:, :, 0], arr[:, :, 1]
    msg = f"Unsupported u shape: {arr.shape}"
    raise ValueError(msg)


def to_canonical_2d(rho_2d: np.ndarray, wall_edge: str) -> np.ndarray:
    """Numpy twin of :func:`src.operators.wetting._canonical_view.to_canonical`.

    Maps ``rho_2d`` (shape ``(nx, ny)``) into the wall-aligned frame where the
    wetting wall is at column 0 with liquid at increasing column index, so the
    bottom-wall formulas below apply unchanged for all four edges. Must stay in
    lock-step with the JAX version — the live simulation path uses that one and
    this fallback runs only when a snapshot lacks ``ca_*``/``cll_*`` keys.
    """
    arr = rho_2d.T if wall_edge in ("left", "right") else rho_2d
    return arr[:, ::-1] if wall_edge in ("top", "right") else arr


def interpolate_interface(row: np.ndarray, rho_mean: float) -> tuple[float, float]:
    """Sub-cell left/right interface positions along *row* via linear interpolation."""
    mask = (row < rho_mean).astype(int)
    diff = np.diff(mask)
    li = int(np.nonzero(diff == -1)[0][0])
    ri = int(np.nonzero(diff == 1)[0][0]) + 1
    x_left = li + (rho_mean - row[li]) / (row[li + 1] - row[li])
    x_right = ri - (rho_mean - row[ri]) / (row[ri - 1] - row[ri])
    return x_left, x_right


def contact_angles_from_rho(rho_2d: np.ndarray, rho_mean: float, wall_edge: str = "bottom") -> tuple[float, float]:
    """Derive ``(left, right)`` contact angles in degrees from the density field."""
    canon = to_canonical_2d(rho_2d, wall_edge)
    xl0, xr0 = interpolate_interface(canon[:, 1], rho_mean)
    xl1, xr1 = interpolate_interface(canon[:, 2], rho_mean)
    left = float(np.rad2deg(math.pi / 2.0 + np.arctan(xl0 - xl1)))
    right = float(np.rad2deg(math.pi / 2.0 + np.arctan(xr1 - xr0)))
    return left, right


def contact_lines_from_rho(rho_2d: np.ndarray, rho_mean: float, wall_edge: str = "bottom") -> tuple[float, float]:
    """Derive ``(left, right)`` contact-line positions from the density field."""
    return interpolate_interface(to_canonical_2d(rho_2d, wall_edge)[:, 1], rho_mean)


def center_of_mass(rho_2d: np.ndarray, rho_mean: float) -> tuple[float, float]:
    """Density-weighted centre of mass of the liquid region."""
    mask = rho_2d > rho_mean
    xi, yi = np.indices(rho_2d.shape)
    total = np.sum(mask * rho_2d)
    return float(np.sum(xi * mask * rho_2d) / total), float(np.sum(yi * mask * rho_2d) / total)


def avg_x_location(rho_2d: np.ndarray, rho_mean: float, offset_x: float) -> float:
    """Mean x-index of the liquid region, measured relative to *offset_x*."""
    nx = rho_2d.shape[0]
    mask = rho_2d > rho_mean
    x_idx = np.arange(nx, dtype=float) - offset_x
    return float(np.sum(x_idx[:, None] * mask) / np.sum(mask))


def mean_velocity_in_liquid(
    u_x: np.ndarray,
    u_y: np.ndarray,
    rho_2d: np.ndarray,
    rho_mean: float,
) -> tuple[float, float]:
    """Mean ``(u_x, u_y)`` over cells whose density exceeds *rho_mean*."""
    mask = rho_2d > rho_mean
    n_liq = np.sum(mask)
    if n_liq == 0:
        return 0.0, 0.0
    return float(np.sum(u_x * mask) / n_liq), float(np.sum(u_y * mask) / n_liq)


def optional_contact_metrics(
    raw: np.lib.npyio.NpzFile,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Read ``(ca_left, ca_right, cll_left, cll_right)`` when present in *raw*.

    ``ca_*`` are contact ANGLES in degrees, matching the ``.npz`` key names.
    """
    ca_l = float(np.asarray(raw["ca_left"]).squeeze()) if "ca_left" in raw else None
    ca_r = float(np.asarray(raw["ca_right"]).squeeze()) if "ca_right" in raw else None
    cll_l = float(np.asarray(raw["cll_left"]).squeeze()) if "cll_left" in raw else None
    cll_r = float(np.asarray(raw["cll_right"]).squeeze()) if "cll_right" in raw else None
    return ca_l, ca_r, cll_l, cll_r
