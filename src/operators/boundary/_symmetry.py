"""Mirror-symmetry boundary condition — pure function.

Extracted from
:class:`simulation_operators.boundary_condition.SymmetryBoundaryCondition`.

For each symmetry edge, incoming distributions are replaced by
mirrored distributions from the post-collision state.
"""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
from registry import boundary_condition
from setup.lattice import Lattice


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
        # Mirror vertical component at y = 0
        for k in range(len(top)):
            dst = top[k]
            src = bot[k]
            f_streamed = f_streamed.at[:, 0, dst, 0].set(f_collision[:, 0, src, 0])
    elif edge == "top":
        # Mirror vertical component at y = ny-1
        for k in range(len(bot)):
            dst = bot[k]
            src = top[k]
            src_vals = f_collision[:, -1, src, 0]
            # Diagonal components need a roll correction for
            # the streaming shift — match the legacy behaviour.
            if k == 1:
                src_vals = jnp.roll(src_vals, 1, axis=0)
            elif k == 2:
                src_vals = jnp.roll(src_vals, -1, axis=0)
            f_streamed = f_streamed.at[:, -1, dst, 0].set(src_vals)
    elif edge == "left":
        # Mirror horizontal component at x = 0
        for k in range(len(right)):
            dst = right[k]
            src = left[k]
            f_streamed = f_streamed.at[0, :, dst, 0].set(f_collision[0, :, src, 0])
    elif edge == "right":
        # Mirror horizontal component at x = nx-1
        for k in range(len(left)):
            dst = left[k]
            src = right[k]
            f_streamed = f_streamed.at[-1, :, dst, 0].set(f_collision[-1, :, src, 0])
    else:
        raise ValueError(f"Unknown edge '{edge}'")

    return f_streamed
