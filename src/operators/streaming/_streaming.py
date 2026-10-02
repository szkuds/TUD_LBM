"""Streaming (propagation) operator — pure function.

Propagates populations along their respective lattice velocity directions
using ``jnp.roll``. The Python ``for`` loop over ``q`` directions is
unrolled at JAX trace time (``q`` is a compile-time constant).

On non-periodic axes the wrap-around layer produced by ``jnp.roll`` is
zeroed after each roll, so populations that leave one wall do not re-enter
at the opposite wall before the boundary-condition operator runs. Without
this, wrap-around populations at bounce-back / wetting walls leak mass.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
import numpy as np
from src.registry import stream_operator

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice


@stream_operator(name="standard")
def stream(
    f: jnp.ndarray,
    lattice: Lattice,
    *,
    periodic_axes: tuple[bool, bool] = (True, True),
) -> jnp.ndarray:
    """Propagate populations; zero-fill the wrap-around on non-periodic axes.

    Args:
        f: Population distributions, shape ``(nx, ny, nz, q, 1)``.
        lattice: :class:`~lattice.lattice.Lattice` with velocity vectors ``c``.
        periodic_axes: Per-axis periodicity ``(x, y)`` —
            :attr:`SimulationConfig.periodic_axes
            <src.config.simulation_config.SimulationConfig.periodic_axes>`.
            The default, fully periodic, zero-fills nothing.

    Returns:
        Post-streaming populations, same shape.
    """
    axes: tuple[int, ...] = tuple(range(lattice.d))  # 0=x, 1=y, (2=z)
    c_np = np.array(lattice.c)  # (1, 1, 1, q, d)

    for i in range(lattice.q):
        fi = f[..., i, :]
        shift = tuple(int(s) for s in c_np[..., i, :].flatten())
        fi = jnp.roll(fi, shift=shift, axis=axes)
        # Kill the wrapped layer on each non-periodic axis.
        for ax, s in zip(axes, shift, strict=False):
            if s != 0 and not periodic_axes[ax]:
                idx: list = [slice(None)] * fi.ndim
                idx[ax] = slice(None, s) if s > 0 else slice(s, None)
                fi = fi.at[tuple(idx)].set(0.0)
        f = f.at[..., i, :].set(fi)

    return f
