"""Interior-obstacle drawing shared by the field panels.

A public module on purpose: it registers no operator, so the plotting
auto-discovery scan skips it.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from matplotlib.colors import ListedColormap

if TYPE_CHECKING:
    import matplotlib.axes
    from src.config import SimulationConfig

#: Flat fill of the solid cells; reads on every field colormap in ``figure_config``.
OBSTACLE_COLOUR = "#4d4d4d"


def solid_mask_2d(config: SimulationConfig) -> np.ndarray | None:
    """The ``(nx, ny)`` solid-cell mask of the ``z = 0`` plane, or ``None`` without an obstacle."""
    mask = config.obstacle_mask
    if mask is None:
        return None
    return np.asarray(mask)[:, :, 0, 0, 0]


def draw_obstacle(ax: matplotlib.axes.Axes, solid: np.ndarray) -> None:
    """Fill the solid cells of an ``imshow(field.T, origin="lower")`` panel with a flat colour.

    Drawn from the mask itself, so any registered obstacle shape renders without
    shape-specific code, at the same cell resolution as the field beneath it.
    """
    ax.imshow(
        np.ma.masked_where(~solid.T, solid.T),
        origin="lower",
        aspect="equal",
        cmap=ListedColormap([OBSTACLE_COLOUR]),
        interpolation="nearest",
    )
