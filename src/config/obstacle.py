"""``[obstacle]``: the interior obstacle's geometry, its validation and its solid-cell mask.

The mask depends on the section and ``grid_shape`` alone and never changes during
a run, so it belongs to the configuration, not to the obstacle operator package.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
import jax.numpy as jnp
from src.config._checks import registered_names
from src.config.defaults import DEFAULT_OBSTACLE_SHAPE

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig


def complete_obstacle(section: dict[str, Any] | None) -> dict[str, Any] | None:
    """*section* with its shape filled in; ``None`` without an obstacle."""
    return None if section is None else {"shape": DEFAULT_OBSTACLE_SHAPE, **section}


def validate_obstacle(config: SimulationConfig) -> None:
    """Validate the obstacle geometry against the grid and the boundary topology."""
    section = config.obstacle
    if section is None:
        return

    valid_shapes = registered_names("src.operators.obstacle", "obstacle")
    if section["shape"] not in valid_shapes:
        msg = f"obstacle shape must be one of {sorted(valid_shapes)}, got '{section['shape']}'"
        raise ValueError(msg)

    nx, ny, nz = config.grid_shape[:3]
    if nz > 1:
        msg = "[obstacle] only supports 2D grids (nz=1)"
        raise ValueError(msg)

    cx = section.get("center_x")
    cy = section.get("center_y")
    radius = section.get("radius")
    if radius is None or radius <= 0:
        msg = f"obstacle radius must be positive, got {radius}"
        raise ValueError(msg)
    if cx is None or not (radius <= cx <= nx - 1 - radius):
        msg = f"obstacle x-extent [{cx - radius}, {cx + radius}] must fit within grid x in [0, {nx - 1}]"
        raise ValueError(msg)
    if cy is None or not (radius + 1 <= cy <= ny - 1 - radius - 1):
        msg = (
            f"obstacle must keep >=1 cell clearance from top/bottom walls, got center_y={cy}, radius={radius}, ny={ny}"
        )
        raise ValueError(msg)

    edges = config.boundary_conditions
    assert edges is not None  # noqa: S101 - complete_boundary_conditions fills it in
    if edges["left"] != "periodic" and cx - radius <= 1:
        msg = f"obstacle must keep >1 cell clearance from a non-periodic left edge, got cx={cx}, radius={radius}"
        raise ValueError(msg)
    if edges["right"] != "periodic" and cx + radius >= nx - 2:
        msg = f"obstacle must keep >1 cell clearance from a non-periodic right edge, got cx={cx}, radius={radius}"
        raise ValueError(msg)


def obstacle_mask(config: SimulationConfig) -> jnp.ndarray | None:
    """Boolean solid-cell mask, shape ``(nx, ny, nz, 1, 1)``; ``None`` without an obstacle."""
    if config.obstacle is None:
        return None
    from src.operators.obstacle import build_obstacle_mask_fn

    nx, ny, nz = config.grid_shape[:3]
    mask = build_obstacle_mask_fn(config.obstacle["shape"])(config.obstacle, (nx, ny, nz))
    return jnp.asarray(mask, dtype=bool)
