"""Circular obstacle mask — pure function.

Builds a static boolean solid-cell mask for a circular cylinder, evaluated
once at setup time using plain numpy (no JAX tracing required).
"""

from __future__ import annotations
from typing import Any
import numpy as np
from tud_lbm.registry import obstacle_operator


@obstacle_operator(name="circle")
def build_circle_mask(
    params: dict[str, Any],
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    """Build a boolean solid-cell mask for a circular cylinder.

    Args:
        params: Obstacle config dict with keys ``center_x``, ``center_y``,
            ``radius`` (all in lattice units).
        grid_shape: Spatial dimensions ``(nx, ny, nz)``.

    Returns:
        Boolean array, shape ``(nx, ny, nz, 1, 1)``. ``True`` marks solid
        cells (inside or on the cylinder boundary).

    Raises:
        ValueError: If ``nz > 1`` — this obstacle shape is 2D-only.
    """
    nx, ny, nz = grid_shape[:3]
    if nz > 1:
        msg = f"circle obstacle only supports 2D grids (nz=1), got nz={nz}"
        raise ValueError(msg)

    cx = float(params["center_x"])
    cy = float(params["center_y"])
    radius = float(params["radius"])

    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2

    return inside.reshape(nx, ny, 1, 1, 1)
