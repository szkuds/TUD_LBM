"""Streaming (propagation) operator — pure function.

0Propagates populations along their respective lattice velocity directions
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
from tud_lbm.registry import stream_operator

if TYPE_CHECKING:
    from tud_lbm.lattice.lattice import Lattice


def _periodic_axes(bc_config: dict | None) -> tuple[bool, ...]:
    """Per-axis periodicity: (x, y).

    An axis is periodic only if both its edges are periodic. ``None`` → fully periodic.
    """
    if bc_config is None:
        return (True, True)
    x_periodic = bc_config.get("left", "periodic") == "periodic" and bc_config.get("right", "periodic") == "periodic"
    y_periodic = bc_config.get("bottom", "periodic") == "periodic" and bc_config.get("top", "periodic") == "periodic"
    return (x_periodic, y_periodic)


@stream_operator(name="standard")
def stream(
    f: jnp.ndarray,
    lattice: Lattice,
    bc_config: dict | None = None,
) -> jnp.ndarray:
    """Propagate populations; zero-fill the wrap-around on non-periodic axes.

    Args:
        f: Population distributions, shape ``(nx, ny, nz, q, 1)``.
        lattice: :class:`~lattice.lattice.Lattice` with velocity vectors ``c``.
        bc_config: Boundary-condition config dict, e.g.
            ``{"top": "bounce-back", "bottom": "bounce-back", "left": "periodic", "right": "periodic"}``.
            ``None`` (default) means fully periodic — no zero-fill.

    Returns:
        Post-streaming populations, same shape.
    """
    axes: tuple[int, ...] = tuple(range(lattice.d))  # 0=x, 1=y, (2=z)
    periodic = _periodic_axes(bc_config)
    c_np = np.array(lattice.c)  # (1, 1, 1, q, d)

    for i in range(lattice.q):
        fi = f[..., i, :]
        shift = tuple(int(s) for s in c_np[..., i, :].flatten())
        fi = jnp.roll(fi, shift=shift, axis=axes)
        # Kill the wrapped layer on each non-periodic axis.
        for ax, s in zip(axes, shift, strict=False):
            if s != 0 and not periodic[ax]:
                idx: list = [slice(None)] * fi.ndim
                idx[ax] = slice(None, s) if s > 0 else slice(s, None)
                fi = fi.at[tuple(idx)].set(0.0)
        f = f.at[..., i, :].set(fi)

    return f
