"""Mirror-symmetry boundary condition — pure function.

Specular reflection: each population entering through the face is the one that
left with its normal velocity component flipped. That population started the
step one tangential velocity away, so it is rolled along the face. The normal
momentum flux through the face is zero and the tangential momentum is kept,
which is what a free-slip plane of symmetry means.

The roll wraps along the face. On a face whose tangential axis is periodic
that is exact; where it meets a wall, the corner cells take the populations
wrapped from the far end, and the BC of the perpendicular edge, applied later,
overwrites the ones it owns.

Before 2026-11-03 the top and bottom faces of a 2D lattice took each incoming
diagonal from the mirror image in *both* axes, so they reversed the tangential
velocity of the diagonals instead of keeping it: a uniform channel flow between
two such faces fell from ``u = 0.05`` to ``0.006`` in 500 steps. The left and
right faces mirrored correctly but did not roll the diagonals along the face.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
import numpy as np
from src.lattice.lattice import incoming_directions
from src.registry import boundary_condition

if TYPE_CHECKING:
    from src.config.boundary import Edge
    from src.lattice.lattice import Lattice


@boundary_condition(name="symmetry", pad_edge_mode="edge")
def apply_symmetry(
    f_streamed: jnp.ndarray,
    f_collision: jnp.ndarray,
    lattice: Lattice,
    edge: Edge,
) -> jnp.ndarray:
    """Reflect the populations leaving through *edge* specularly back into the domain.

    Args:
        f_streamed: Post-streaming populations, shape ``(nx, ny, nz, q, 1)``.
        f_collision: Post-collision populations, same shape.
        lattice: :class:`~src.lattice.lattice.Lattice`.
        edge: A domain face; ``"front"`` and ``"back"`` exist on a 3D lattice.

    Returns:
        Updated populations with symmetry applied on *edge*.
    """
    axis = edge.normal_axis
    directions = [tuple(int(v) for v in ci) for ci in np.array(lattice.c)[0, 0, 0]]
    cells = edge.cells
    for dst in incoming_directions(lattice, edge):
        mirrored = list(directions[dst])
        mirrored[axis] = -mirrored[axis]
        src = directions.index(tuple(mirrored))
        tangential = [(ax, directions[src][ax]) for ax in range(lattice.d) if ax != axis and directions[src][ax] != 0]
        reflected = f_collision[..., src, 0]
        if tangential:
            reflected = jnp.roll(
                reflected,
                shift=tuple(shift for _, shift in tangential),
                axis=tuple(ax for ax, _ in tangential),
            )
        f_streamed = f_streamed.at[(*cells, dst, 0)].set(reflected[cells])
    return f_streamed
