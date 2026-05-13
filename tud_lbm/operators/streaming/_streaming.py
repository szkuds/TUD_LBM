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
from typing import TYPE_CHECKING
import jax.numpy as jnp
import numpy as np
from tud_lbm.registry import stream_operator

if TYPE_CHECKING:
    from tud_lbm.lattice.lattice import Lattice


@stream_operator(name="standard")
def stream(
    f: jnp.ndarray,
    lattice: Lattice,
) -> jnp.ndarray:
    """Propagate populations along lattice velocity directions.

    After each ``jnp.roll``, the boundary row where the wrap-around
    lands is zero-filled when that edge carries a bounce-back or
    wetting boundary condition.  This prevents spurious wrapped
    populations from persisting before the BC operator runs.

    Args:
        f: Population distributions, shape ``(nx, ny, nz, q, 1)``.
        lattice: :class:`~setup.lattice.Lattice` with velocity vectors ``c``.
        bc_config: Boundary-condition config dict, e.g.
            ``{"top": "bounce-back", "bottom": "bounce-back", "left": "periodic", "right": "periodic"}``.
            ``None`` (default) means fully periodic — no zero-fill.

    Returns:
        Post-streaming populations, same shape.
    """
    axes: tuple[int, ...] = tuple(range(lattice.d))  # grid axes: 0=x, 1=y, 2=z

    # Pre-extract velocity vectors as plain Python ints so they are
    # compile-time constants under JAX tracing.
    # lattice.c has shape (1, 1, 1, q, d); extracting [i, :] and flattening
    # gives us the d-component velocity vector for direction i.
    c_np = np.array(lattice.c)  # (1, 1, 1, q, d)

    for i in range(lattice.q):
        shift = tuple(c_np[..., i, :].flatten())
        f = f.at[..., i, :].set(jnp.roll(f[..., i, :], shift=shift, axis=axes))

    return f
