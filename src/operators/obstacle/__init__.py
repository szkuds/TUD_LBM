"""Interior-obstacle operators — implementations of ObstacleOperator protocol.

Public API: build_obstacle_fn()

Implementation modules (_circle.py, _obstacle_aggregator.py) are internal. The
solid-cell mask is built by the configuration, ``SimulationConfig.obstacle_mask``
— see :mod:`src.config.obstacle_mask`.

Example:
    from src.operators.obstacle import build_obstacle_fn

    obstacle_fn = build_obstacle_fn(config.obstacle_mask, lattice)
    f_stream = obstacle_fn(f_stream, f_collision)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators._loader import auto_load_operators
from src.operators.obstacle._obstacle_aggregator import ObstacleBounceBack

if TYPE_CHECKING:
    import jax.numpy as jnp
    from src.lattice.lattice import Lattice
    from src.operators.protocols import ObstacleOperator

# Auto-discover and import private operator modules for registry registration
auto_load_operators("src.operators.obstacle")


def build_obstacle_fn(mask: jnp.ndarray | None, lattice: Lattice) -> ObstacleOperator | None:
    """Return the obstacle bounce-back operator for *mask*.

    Args:
        mask: :attr:`SimulationConfig.obstacle_mask
            <src.config.simulation_config.SimulationConfig.obstacle_mask>`;
            ``None`` means the run has no obstacle.
        lattice: :class:`~src.lattice.lattice.Lattice`.

    Returns:
        ``obstacle_fn(f_stream, f_col) -> f_stream``, or ``None`` without an obstacle.
    """
    if mask is None:
        return None
    return ObstacleBounceBack(mask=mask, lattice=lattice)


__all__ = ["build_obstacle_fn"]
