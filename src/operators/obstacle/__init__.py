"""Interior-obstacle shapes — implementations of the ObstacleMaskBuilder signature.

Public API: build_obstacle_mask_fn()

Each ``_*.py`` module registers one shape under the ``obstacle`` kind. The
configuration builds the run's solid-cell mask with it
(``SimulationConfig.obstacle_mask``, see :mod:`src.config.obstacle_mask`), and
the bounce-back on that mask is the first link of the boundary chain
(:func:`src.operators.boundary.build_bc`).

Example:
    from src.operators.obstacle import build_obstacle_mask_fn

    mask = build_obstacle_mask_fn("circle")(obstacle_config, grid_shape)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators._loader import auto_load_operators
from src.registry import get_operators

if TYPE_CHECKING:
    from src.protocols import ObstacleMaskBuilder

# Auto-discover and import private operator modules for registry registration
auto_load_operators("src.operators.obstacle")


def build_obstacle_mask_fn(shape: str) -> ObstacleMaskBuilder:
    """Return the mask builder registered for the obstacle *shape*.

    Args:
        shape: The validated ``obstacle_config["shape"]``.

    Returns:
        ``build(params, grid_shape) -> mask``, a boolean array of shape ``(nx, ny, nz, 1, 1)``.
    """
    return get_operators("obstacle")[shape].target


__all__ = ["build_obstacle_mask_fn"]
