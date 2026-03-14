"""Streaming (propagation) operator — pure function.

Extracted from :class:`simulation_operators.stream.Streaming`.
Propagates populations along their respective lattice velocity directions
using ``jnp.roll``.  The Python ``for`` loop over ``q`` directions is
unrolled at JAX trace time (``q`` is a compile-time constant).
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import numpy as np

from setup.lattice import Lattice
from registry import stream_operator


@stream_operator(name="standard")
def stream(f: jnp.ndarray, lattice: Lattice) -> jnp.ndarray:
    """Propagate populations along lattice velocity directions.

    Args:
        f: Population distributions, shape ``(nx, ny, q, 1)``.
        lattice: :class:`~setup.lattice.Lattice` with velocity vectors ``c``.

    Returns:
        Post-streaming populations, same shape.
    """
    axes: Tuple[int, ...] = tuple(range(f.ndim - 2))  # grid axes (0, 1)

    # Pre-extract velocity vectors as plain Python ints so they are
    # compile-time constants under JAX tracing.
    c_np = np.array(lattice.c)  # (d, q) as numpy

    for i in range(lattice.q):
        shift = tuple(int(c_np[d, i]) for d in range(c_np.shape[0]))
        f = f.at[..., i, :].set(jnp.roll(f[..., i, :], shift=shift, axis=axes))
    return f
