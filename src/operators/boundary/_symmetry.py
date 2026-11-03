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
from src.operators.boundary._faces import face_axis
from src.operators.boundary._faces import face_cells
from src.operators.boundary._faces import incoming_directions
from src.registry import boundary_condition

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice

# Diagonal component indices for streaming roll correction
DIAGONAL_1_IDX = 1
DIAGONAL_2_IDX = 2
SPATIAL_3D = 3


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


def _apply_specular(
    f_streamed: jnp.ndarray,
    f_collision: jnp.ndarray,
    lattice: Lattice,
    edge: str,
) -> jnp.ndarray:
    """Specular reflection at a face of a 3D lattice.

    Each incoming direction takes the direction with its normal component
    flipped. That population started the step one tangential velocity away, so
    it is rolled along the tangential axes; the wall normal keeps its momentum
    flux and the tangential momentum is conserved.
    """
    axis = face_axis(edge)
    c = np.array(lattice.c)[0, 0, 0]
    directions = [tuple(int(v) for v in ci) for ci in c]
    cells = face_cells(edge)
    for dst in incoming_directions(lattice, edge):
        mirrored = list(directions[dst])
        mirrored[axis] = -mirrored[axis]
        src = directions.index(tuple(mirrored))
        tangential = [(ax, directions[src][ax]) for ax in range(lattice.d) if ax != axis]
        rolled = jnp.roll(
            f_collision[..., src, 0],
            shift=tuple(shift for _, shift in tangential),
            axis=tuple(ax for ax, _ in tangential),
        )
        f_streamed = f_streamed.at[(*cells, dst, 0)].set(rolled[cells])
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
        edge: A domain face; ``"front"`` and ``"back"`` exist on a 3D lattice.

    Returns:
        Updated populations with symmetry applied on *edge*.
    """
    if lattice.d == SPATIAL_3D:
        return _apply_specular(f_streamed, f_collision, lattice, edge)

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
