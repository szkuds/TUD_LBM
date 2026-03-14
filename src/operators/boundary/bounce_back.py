"""Half-way bounce-back boundary condition — pure function.

Extracted from
:class:`simulation_operators.boundary_condition.BounceBackBoundaryCondition`.

For each bounce-back edge, incoming populations are replaced by
opposite-direction populations from the post-collision field.

The ``for`` loops over direction indices are unrolled at trace time
because the loop bounds (and the index values themselves) are
compile-time constants derived from the :class:`Lattice`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from setup.lattice import Lattice
from registry import boundary_condition


@boundary_condition(name="bounce-back")
def apply_bounce_back(
    f_streamed: jnp.ndarray,
    f_collision: jnp.ndarray,
    lattice: Lattice,
    edge: str,
) -> jnp.ndarray:
    """Apply half-way bounce-back on one edge.

    Args:
        f_streamed: Post-streaming populations, shape ``(nx, ny, q, 1)``.
        f_collision: Post-collision populations (before streaming), same shape.
        lattice: :class:`~setup.lattice.Lattice`.
        edge: ``"top"``, ``"bottom"``, ``"left"``, or ``"right"``.

    Returns:
        Updated populations with bounce-back applied on *edge*.
    """
    # Pre-extract indices as plain Python ints so they are static
    # under JAX tracing (no ConcretizationTypeError).
    opp = [int(x) for x in np.array(lattice.opp_indices)]

    if edge == "bottom":
        # Incoming at y=0: directions with positive cy → top_indices
        incoming = [int(x) for x in np.array(lattice.top_indices)]
        for i in incoming:
            f_streamed = f_streamed.at[:, 0, i, 0].set(f_collision[:, 0, opp[i], 0])
    elif edge == "top":
        # Incoming at y=-1: directions with negative cy → bottom_indices
        incoming = [int(x) for x in np.array(lattice.bottom_indices)]
        for i in incoming:
            f_streamed = f_streamed.at[:, -1, i, 0].set(f_collision[:, -1, opp[i], 0])
    elif edge == "left":
        # Incoming at x=0: directions with positive cx → right_indices
        incoming = [int(x) for x in np.array(lattice.right_indices)]
        for i in incoming:
            f_streamed = f_streamed.at[0, :, i, 0].set(f_collision[0, :, opp[i], 0])
    elif edge == "right":
        # Incoming at x=-1: directions with negative cx → left_indices
        incoming = [int(x) for x in np.array(lattice.left_indices)]
        for i in incoming:
            f_streamed = f_streamed.at[-1, :, i, 0].set(f_collision[-1, :, opp[i], 0])
    else:
        raise ValueError(f"Unknown edge '{edge}'")

    return f_streamed
