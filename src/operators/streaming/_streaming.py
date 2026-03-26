"""Streaming (propagation) operator — pure function.

Extracted from :class:`simulation_operators.stream.Streaming`.
Propagates populations along their respective lattice velocity directions
using ``jnp.roll``.  The Python ``for`` loop over ``q`` directions is
unrolled at JAX trace time (``q`` is a compile-time constant).

For edges with a bounce-back wall (including wetting, which implements
bounce-back internally), the wrapped ghost layer is zeroed out after
each roll so that wrap-around populations do not contaminate the domain
interior before the boundary-condition operator runs.
"""

from __future__ import annotations
from typing import Any
import jax.numpy as jnp
import numpy as np
from registry import stream_operator
from setup.lattice import Lattice

# BC types that implement solid-wall (bounce-back) behaviour.
_WALL_BC_TYPES = frozenset({"bounce-back", "wetting"})


def _has_wall_bc(bc_config: dict[str, Any] | None, edge: str) -> bool:
    """Return ``True`` if *edge* has a wall-type BC (bounce-back or wetting)."""
    if bc_config is None:
        return False
    return bc_config.get(edge, "periodic") in _WALL_BC_TYPES


@stream_operator(name="standard")
def stream(
    f: jnp.ndarray,
    lattice: Lattice,
    bc_config: dict[str, Any] | None = None,
) -> jnp.ndarray:
    """Propagate populations along lattice velocity directions.

    After each ``jnp.roll``, the boundary row where the wrap-around
    lands is zero-filled when that edge carries a bounce-back or
    wetting boundary condition.  This prevents spurious wrapped
    populations from persisting before the BC operator runs.

    Args:
        f: Population distributions, shape ``(nx, ny, q, 1)``.
        lattice: :class:`~setup.lattice.Lattice` with velocity vectors ``c``.
        bc_config: Boundary-condition config dict, e.g.
            ``{"top": "bounce-back", "bottom": "bounce-back",
              "left": "periodic", "right": "periodic"}``.
            ``None`` (default) means fully periodic — no zero-fill.

    Returns:
        Post-streaming populations, same shape.
    """
    axes: tuple[int, ...] = tuple(range(f.ndim - 2))  # grid axes (0, 1)

    # Pre-extract velocity vectors as plain Python ints so they are
    # compile-time constants under JAX tracing.
    c_np = np.array(lattice.c)  # (d, q) as numpy

    # Pre-compute per-edge wall flags (resolved once at trace time).
    wall_left = _has_wall_bc(bc_config, "left")
    wall_right = _has_wall_bc(bc_config, "right")
    wall_bottom = _has_wall_bc(bc_config, "bottom")
    wall_top = _has_wall_bc(bc_config, "top")

    for i in range(lattice.q):
        shift = tuple(int(c_np[d, i]) for d in range(c_np.shape[0]))
        f = f.at[..., i, :].set(jnp.roll(f[..., i, :], shift=shift, axis=axes))

        # Zero-fill the boundary row where jnp.roll deposited a
        # wrapped population, but only when that edge is a wall.
        #
        #   roll(+1) along axis → wrap lands at index  0 of that axis
        #   roll(-1) along axis → wrap lands at index -1 of that axis
        sx, sy = shift  # for 2-D lattices (d == 2)
        if sx > 0 and wall_left:
            f = f.at[0, :, i, :].set(0.0)
        elif sx < 0 and wall_right:
            f = f.at[-1, :, i, :].set(0.0)
        if sy > 0 and wall_bottom:
            f = f.at[:, 0, i, :].set(0.0)
        elif sy < 0 and wall_top:
            f = f.at[:, -1, i, :].set(0.0)
    return f
