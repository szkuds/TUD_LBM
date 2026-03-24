"""JAX-friendly Lattice definition and factory.

``Lattice`` is a :class:`typing.NamedTuple` — automatically a valid JAX
pytree.  All array fields are stored as :mod:`jax.numpy` arrays so they
can flow through ``jax.jit`` and ``jax.lax.scan`` without conversion.

Usage::

    from setup.lattice import build_lattice

    lattice = build_lattice("D2Q9")
    assert lattice.d == 2
    assert lattice.q == 9
    assert lattice.w.shape == (9,)
"""

from __future__ import annotations
from typing import NamedTuple
import jax.numpy as jnp
import numpy as np
from registry import get_operators
from registry import lattice_operator


class Lattice(NamedTuple):
    """Immutable lattice velocity model — valid JAX pytree.

    Attributes:
        name: Human-readable identifier, e.g. ``"D2Q9"``.
        d: Number of spatial dimensions.
        q: Number of discrete velocities.
        c: Velocity vectors, shape ``(d, q)`` — ``jax.Array``.
        w: Quadrature weights, shape ``(q,)`` — ``jax.Array``.
        opp_indices: Opposite-direction index for each velocity,
            shape ``(q,)`` — ``jax.Array`` of ints.
        main_indices: Indices of the cardinal (non-diagonal) directions,
            shape varies — ``jax.Array`` of ints.
        right_indices: Indices with positive x-component — ``jax.Array``.
        left_indices: Indices with negative x-component — ``jax.Array``.
        top_indices: Indices with positive y-component — ``jax.Array``.
        bottom_indices: Indices with negative y-component — ``jax.Array``.
    """

    name: str
    d: int
    q: int
    c: jnp.ndarray
    w: jnp.ndarray
    opp_indices: jnp.ndarray
    main_indices: jnp.ndarray
    right_indices: jnp.ndarray
    left_indices: jnp.ndarray
    top_indices: jnp.ndarray
    bottom_indices: jnp.ndarray


# ── D2Q9 constants (computed once at import time) ────────────────────

_D2Q9_CX = [0, 1, 0, -1, 0, 1, -1, -1, 1]
_D2Q9_CY = [0, 0, 1, 0, -1, 1, 1, -1, -1]
_D2Q9_W = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]


@lattice_operator(name="D2Q9", dim=2, q=9)
def _build_d2q9() -> Lattice:
    """Construct a D2Q9 :class:`Lattice`."""
    c_np = np.array(list(zip(_D2Q9_CX, _D2Q9_CY, strict=False))).T  # shape (2, 9)
    w_np = np.array(_D2Q9_W)
    c_t = c_np.T  # shape (9, 2) — row per velocity

    opp = np.array([c_t.tolist().index((-c_t[i]).tolist()) for i in range(9)])
    main = np.nonzero((np.abs(c_t[:, 0]) + np.abs(c_t[:, 1])) == 1)[0]
    right = np.nonzero(c_t[:, 0] == 1)[0]
    left = np.nonzero(c_t[:, 0] == -1)[0]
    top = np.nonzero(c_t[:, 1] == 1)[0]
    bottom = np.nonzero(c_t[:, 1] == -1)[0]

    return Lattice(
        name="D2Q9",
        d=2,
        q=9,
        c=jnp.array(c_np),
        w=jnp.array(w_np),
        opp_indices=jnp.array(opp),
        main_indices=jnp.array(main),
        right_indices=jnp.array(right),
        left_indices=jnp.array(left),
        top_indices=jnp.array(top),
        bottom_indices=jnp.array(bottom),
    )


# ── Public factory ───────────────────────────────────────────────────


def build_lattice(name: str) -> Lattice:
    """Create a :class:`Lattice` for the given velocity model.

    Uses the global operator registry to resolve lattice builders.

    Args:
        name: Lattice identifier, e.g. ``"D2Q9"``.

    Returns:
        An immutable :class:`Lattice` with JAX arrays.

    Raises:
        ValueError: If *name* is not a supported lattice type.
    """
    # Normalise to upper-case for robustness
    key = name.upper()
    lattices = get_operators("lattice")
    if key not in lattices:
        supported = ", ".join(sorted(lattices))
        raise ValueError(f"Unsupported lattice type '{name}'. Supported: {supported}")
    target = lattices[key].target
    return target()
