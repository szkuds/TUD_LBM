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
from typing import TYPE_CHECKING
import numpy as np
from src.lattice.lattice import incoming_directions
from src.registry import boundary_condition

if TYPE_CHECKING:
    import jax.numpy as jnp
    from src.config.boundary import Edge
    from src.lattice.lattice import Lattice


@boundary_condition(name="bounce-back", pad_edge_mode="edge")
def apply_bounce_back(
    f_streamed: jnp.ndarray,
    f_collision: jnp.ndarray,
    lattice: Lattice,
    edge: Edge,
) -> jnp.ndarray:
    """Apply half-way bounce-back on one edge.

    Args:
        f_streamed: Post-streaming populations, shape ``(nx, ny, nz, q, d)``.
        f_collision: Post-collision populations (before streaming), same shape.
        lattice: :class:`~setup.lattice.Lattice`.
        edge: The domain face.

    Returns:
        Updated populations with bounce-back applied on *edge*.
    """
    # Plain Python ints, so they are static under JAX tracing.
    opp = [int(x) for x in np.array(lattice.opp_indices)]
    cells = edge.cells
    for i in incoming_directions(lattice, edge):
        f_streamed = f_streamed.at[(*cells, i, 0)].set(f_collision[(*cells, opp[i], 0)])
    return f_streamed


# A wetting wall is a bounce-back wall for the populations; the wetting itself is
# applied in the density stencils, not here. An alias is a second registration,
# not a remap inside the builder.
boundary_condition(name="wetting", pad_edge_mode="edge")(apply_bounce_back)
