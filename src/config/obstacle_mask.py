"""Obstacle mask — the static solid cells of a configuration.

Built only by :attr:`SimulationConfig.obstacle_mask
<src.config.simulation_config.SimulationConfig.obstacle_mask>`: the mask
depends on ``obstacle_config`` and ``grid_shape`` alone and never changes
during a run, so it belongs to the configuration, not to the obstacle
operator package.
"""

from __future__ import annotations
from typing import Any
import jax.numpy as jnp
from src.registry import get_operators


def build_obstacle_mask(
    obstacle_config: dict[str, Any] | None,
    grid_shape: tuple[int, int, int],
) -> jnp.ndarray | None:
    """Build the solid-cell mask from a validated obstacle config.

    Args:
        obstacle_config: Mapping with a registered ``shape`` plus that shape's
            geometry parameters, as :class:`SimulationConfig` validates it.
            ``None`` means no obstacle.
        grid_shape: Spatial dimensions ``(nx, ny, nz)``.

    Returns:
        Boolean array, shape ``(nx, ny, nz, 1, 1)``, or ``None`` without an obstacle.
    """
    if obstacle_config is None:
        return None
    import src.operators.obstacle  # noqa: F401 - registers the obstacle shapes

    mask = get_operators("obstacle")[obstacle_config["shape"]].target(obstacle_config, grid_shape)
    return jnp.asarray(mask, dtype=bool)
