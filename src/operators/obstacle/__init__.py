"""Interior-obstacle operators — composite builder.

Public API: build_obstacle_mask(), build_obstacle_fn()

Implementation modules (_circle.py) are internal.

Example:
    from src.operators.obstacle import build_obstacle_mask, build_obstacle_fn

    mask = build_obstacle_mask({"shape": "circle", "center_x": 80, "center_y": 50, "radius": 10}, (400, 100, 1))
    obstacle_fn = build_obstacle_fn(mask, lattice)
    f_stream = obstacle_fn(f_stream, f_collision)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
import jax.numpy as jnp
import numpy as np
from src.operators._loader import auto_load_operators
from src.operators.obstacle import _circle as _circ  # noqa: F401
from src.registry import get_operators

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice
    from src.operators.protocols import ObstacleOperator

auto_load_operators("src.operators.obstacle")


def build_obstacle_mask(
    obstacle_config: dict[str, Any] | None,
    grid_shape: tuple[int, int, int],
) -> jnp.ndarray | None:
    """Build a static solid-cell mask from an obstacle config.

    Args:
        obstacle_config: Mapping with an optional ``shape`` key (default
            ``"circle"``) plus shape-specific geometry params. ``None``
            means no obstacle.
        grid_shape: Spatial dimensions ``(nx, ny, nz)``.

    Returns:
        Boolean jax array, shape ``(nx, ny, nz, 1, 1)``, or ``None`` if
        *obstacle_config* is ``None``.
    """
    if obstacle_config is None:
        return None

    shape = obstacle_config.get("shape", "circle")
    ops = get_operators("obstacle")
    try:
        build_fn = ops[shape].target
    except KeyError as exc:
        valid_shapes = ", ".join(sorted(ops.keys()))
        msg = f"Unknown obstacle shape '{shape}'. Valid shapes: {valid_shapes}"
        raise ValueError(msg) from exc

    mask = build_fn(obstacle_config, grid_shape)
    return jnp.asarray(mask, dtype=bool)


def build_obstacle_fn(
    mask: jnp.ndarray | None,
    lattice: Lattice,
) -> ObstacleOperator | None:
    """Build an obstacle bounce-back closure from a precomputed mask.

    Args:
        mask: Boolean solid-cell mask, shape ``(nx, ny, nz, 1, 1)``, or
            ``None`` if there is no obstacle.
        lattice: :class:`~src.lattice.lattice.Lattice`.

    Returns:
        A callable ``obstacle_fn(f_stream, f_col) -> f_stream`` satisfying
        :class:`~src.operators.protocols.ObstacleOperator`, or ``None``
        if *mask* is ``None``.
    """
    if mask is None:
        return None

    opp = [int(x) for x in np.array(lattice.opp_indices)]

    def obstacle_fn(f_stream: jnp.ndarray, f_col: jnp.ndarray) -> jnp.ndarray:
        """Reverse populations at masked solid cells (full bounce-back)."""
        return jnp.where(mask, f_col[..., opp, :], f_stream)

    return obstacle_fn


__all__ = ["build_obstacle_fn", "build_obstacle_mask"]
