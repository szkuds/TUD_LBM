"""Outlet boundary condition — pure function.

Zero-gradient (convective/Neumann) outflow: copies the second-to-last
column onto the last column, post-streaming. Simplest stable outlet for
a worked channel-flow example.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.registry import boundary_condition

if TYPE_CHECKING:
    import jax.numpy as jnp
    from src.lattice.lattice import Lattice


@boundary_condition(name="outlet", pad_edge_mode="edge")
def apply_outlet(
    f_streamed: jnp.ndarray,
    _f_collision: jnp.ndarray,
    _lattice: Lattice,
    edge: str,
) -> jnp.ndarray:
    """Apply a zero-gradient outlet on the ``"right"`` edge.

    Args:
        f_streamed: Post-streaming populations, shape ``(nx, ny, nz, q, 1)``.
        _f_collision: Post-collision populations (unused).
        _lattice: Lattice (unused).
        edge: Edge name; only ``"right"`` is handled, other edges are no-ops.

    Returns:
        Populations with the last column set to the second-to-last column.
    """
    if edge != "right":
        return f_streamed

    return f_streamed.at[-1, :, 0, :, 0].set(f_streamed[-2, :, 0, :, 0])
