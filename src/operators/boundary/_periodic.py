"""Periodic boundary condition — pure function (no-op).

Streaming via ``jnp.roll`` already wraps periodically, so periodic
boundaries require no additional transformation.
"""

from __future__ import annotations
import jax.numpy as jnp
from registry import boundary_condition
from setup.lattice import Lattice


@boundary_condition(name="periodic", pad_edge_mode="wrap")
def apply_periodic(
    f_streamed: jnp.ndarray,
    f_collision: jnp.ndarray,
    lattice: Lattice,
    edge: str,
) -> jnp.ndarray:
    """No-op: streaming already handles periodicity.

    Args:
        f_streamed: Post-streaming populations.
        f_collision: Post-collision populations (unused).
        lattice: Lattice (unused).
        edge: Edge name (unused).

    Returns:
        ``f_streamed`` unchanged.
    """
    return f_streamed
