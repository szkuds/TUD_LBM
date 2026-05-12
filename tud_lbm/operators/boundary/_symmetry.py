"""Mirror-symmetry boundary condition — pure function.

Extracted from
:class:`simulation_operators.boundary_condition.SymmetryBoundaryCondition`.

For each symmetry edge, incoming distributions are replaced by
mirrored distributions from the post-collision state.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
import numpy as np
from tud_lbm.registry import boundary_condition

if TYPE_CHECKING:
    from tud_lbm.lattice.lattice import Lattice

# Diagonal component indices for streaming roll correction
DIAGONAL_1_IDX = 1
DIAGONAL_2_IDX = 2


def _apply_bottom_symmetry(
    f_streamed: jnp.ndarray,
    f_collision: jnp.ndarray,
    top_indices: list[int],
    bot_indices: list[int],
) -> jnp.ndarray:
    """Apply mirror-symmetry at bottom edge (y=0)."""
    for k in range(len(top_indices)):
        dst = top_indices[k]
        src = bot_indices[k]
        f_streamed = f_streamed.at[:, 0, 0, dst, 0].set(f_collision[:, 0, 0, src, 0])
    return f_streamed


def _apply_top_symmetry(
    f_streamed: jnp.ndarray,
    f_collision: jnp.ndarray,
    top_indices: list[int],
    bot_indices: list[int],
) -> jnp.ndarray:
    """Apply mirror-symmetry at top edge (y=ny-1) with diagonal roll correction."""
    for k in range(len(bot_indices)):
        dst = bot_indices[k]
        src = top_indices[k]
        src_vals = f_collision[:, -1, 0, src, 0]
        # Diagonal components need a roll correction for streaming shift
        if k == DIAGONAL_1_IDX:
            src_vals = jnp.roll(src_vals, 1, axis=0)
        elif k == DIAGONAL_2_IDX:
            src_vals = jnp.roll(src_vals, -1, axis=0)
        f_streamed = f_streamed.at[:, -1, 0, dst, 0].set(src_vals)
    return f_streamed


def _apply_left_symmetry(
    f_streamed: jnp.ndarray,
    f_collision: jnp.ndarray,
    right_indices: list[int],
    left_indices: list[int],
) -> jnp.ndarray:
    """Apply mirror-symmetry at left edge (x=0)."""
    for k in range(len(right_indices)):
        dst = right_indices[k]
        src = left_indices[k]
        f_streamed = f_streamed.at[0, :, 0, dst, 0].set(f_collision[0, :, 0, src, 0])
    return f_streamed


def _apply_right_symmetry(
    f_streamed: jnp.ndarray,
    f_collision: jnp.ndarray,
    right_indices: list[int],
    left_indices: list[int],
) -> jnp.ndarray:
    """Apply mirror-symmetry at right edge (x=nx-1)."""
    for k in range(len(left_indices)):
        dst = left_indices[k]
        src = right_indices[k]
        f_streamed = f_streamed.at[-1, :, 0, dst, 0].set(f_collision[-1, :, 0, src, 0])
    return f_streamed


@boundary_condition(name="symmetry", pad_edge_mode="edge")
def apply_symmetry(
    f_streamed: jnp.ndarray,
    f_collision: jnp.ndarray,
    lattice: Lattice,
    edge: str,
) -> jnp.ndarray:
    """Apply mirror-symmetry BC on one edge.

    Args:
        f_streamed: Post-streaming populations, shape ``(nx, ny, q, 1)``.
        f_collision: Post-collision populations, same shape.
        lattice: :class:`~setup.lattice.Lattice`.
        edge: ``"top"``, ``"bottom"``, ``"left"``, or ``"right"``.

    Returns:
        Updated populations with symmetry applied on *edge*.
    """
    # Convert to plain Python ints for JAX compatibility under tracing
    top = [int(x) for x in np.array(lattice.top_indices)]
    bot = [int(x) for x in np.array(lattice.bottom_indices)]
    right = [int(x) for x in np.array(lattice.right_indices)]
    left = [int(x) for x in np.array(lattice.left_indices)]

    if edge == "bottom":
        return _apply_bottom_symmetry(f_streamed, f_collision, top, bot)
    if edge == "top":
        return _apply_top_symmetry(f_streamed, f_collision, top, bot)
    if edge == "left":
        return _apply_left_symmetry(f_streamed, f_collision, right, left)
    if edge == "right":
        return _apply_right_symmetry(f_streamed, f_collision, right, left)
    msg = f"Unknown edge '{edge}'"
    raise ValueError(msg)
