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
from typing import Any
import jax.numpy as jnp
import numpy as np
from tud_lbm.registry import stream_operator

if TYPE_CHECKING:
    from tud_lbm.lattice.lattice import Lattice

# BC types that implement solid-wall (bounce-back) behaviour.
_WALL_BC_TYPES = frozenset({"bounce-back", "wetting"})
_DIM_X = 1
_DIM_Y = 2
_DIM_Z = 3


def _has_wall_bc(bc_config: dict[str, Any] | None, edge: str) -> bool:
    """Return ``True`` if *edge* has a wall-type BC (bounce-back or wetting)."""
    if bc_config is None:
        return False
    return bc_config.get(edge, "periodic") in _WALL_BC_TYPES


def _zero_fill_x_walls(
    f: jnp.ndarray,
    shift_x: int,
    wall_left: bool,
    wall_right: bool,
    i: int,
) -> jnp.ndarray:
    """Zero-fill x-axis walls after rolling."""
    if shift_x > 0 and wall_left:
        f = f.at[0, :, i, :].set(0.0)
    elif shift_x < 0 and wall_right:
        f = f.at[-1, :, i, :].set(0.0)
    return f


def _zero_fill_y_walls(
    f: jnp.ndarray,
    shift_y: int,
    wall_bottom: bool,
    wall_top: bool,
    i: int,
) -> jnp.ndarray:
    """Zero-fill y-axis walls after rolling."""
    if shift_y > 0 and wall_bottom:
        f = f.at[:, 0, i, :].set(0.0)
    elif shift_y < 0 and wall_top:
        f = f.at[:, -1, i, :].set(0.0)
    return f


def _zero_fill_z_walls(
    f: jnp.ndarray,
    shift_z: int,
    wall_front: bool,
    wall_back: bool,
    i: int,
) -> jnp.ndarray:
    """Zero-fill z-axis walls after rolling."""
    if shift_z > 0 and wall_front:
        f = f.at[:, :, 0, i, :].set(0.0)
    elif shift_z < 0 and wall_back:
        f = f.at[:, :, -1, i, :].set(0.0)
    return f


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

    # Pre-compute per-edge wall flags (resolved once at trace time).
    wall_left = _has_wall_bc(bc_config, "left")
    wall_right = _has_wall_bc(bc_config, "right")
    wall_bottom = _has_wall_bc(bc_config, "bottom")
    wall_top = _has_wall_bc(bc_config, "top")
    wall_front = _has_wall_bc(bc_config, "front")
    wall_back = _has_wall_bc(bc_config, "back")

    for i in range(lattice.q):
        shift = tuple(c_np[..., i, :].flatten())
        f = f.at[..., i, :].set(jnp.roll(f[..., i, :], shift=shift, axis=axes))

        # Zero-fill the boundary row where jnp.roll deposited a
        # wrapped population, but only when that edge is a wall.
        #
        #   roll(+1) along axis → wrap lands at index  0 of that axis
        #   roll(-1) along axis → wrap lands at index -1 of that axis

        if lattice.d >= _DIM_X:
            f = _zero_fill_x_walls(f, shift[0], wall_left, wall_right, i)

        if lattice.d >= _DIM_Y:
            f = _zero_fill_y_walls(f, shift[1], wall_bottom, wall_top, i)

        if lattice.d >= _DIM_Z:
            f = _zero_fill_z_walls(f, shift[2], wall_front, wall_back, i)

    return f
