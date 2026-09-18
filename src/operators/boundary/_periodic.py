"""Periodic boundary condition — pure function (no-op).

Streaming via ``jnp.roll`` already wraps periodically, so periodic
boundaries require no additional transformation.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.registry import boundary_condition

if TYPE_CHECKING:
    import jax.numpy as jnp
    from src.lattice.lattice import Lattice


@boundary_condition(name="periodic", pad_edge_mode="wrap")
def apply_periodic(
    f_streamed: jnp.ndarray,
    _f_collision: jnp.ndarray,
    _lattice: Lattice,
    _edge: str,
) -> jnp.ndarray:
    """No-op: streaming already handles periodicity.

    Args:
        f_streamed: Post-streaming populations.
        _f_collision: Post-collision populations (unused).
        _lattice: Lattice (unused).
        _edge: Edge name (unused).

    Returns:
        ``f_streamed`` unchanged.
    """
    return f_streamed
