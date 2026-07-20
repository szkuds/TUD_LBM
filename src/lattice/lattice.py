"""JAX-friendly Lattice definition and factory.

``Lattice`` is a :class:`typing.NamedTuple` — automatically a valid JAX
pytree.  All array fields are stored as :mod:`jax.numpy` arrays so they
can flow through ``jax.jit`` and ``jax.lax.scan`` without conversion.

All lattice arrays are broadcast-compatible with the 5D population layout
(nx, ny, nz, q, d):
- Velocity vectors ``c`` have shape ``(1, 1, 1, q, d)``
- Quadrature weights ``w`` have shape ``(1, 1, 1, q, 1)``

Usage::

    from src.lattice import build_lattice

    lattice = build_lattice("D2Q9")
    assert lattice.d == 2
    assert lattice.q == 9
    assert lattice.c.shape == (1, 1, 1, 9, 2)
    assert lattice.w.shape == (1, 1, 1, 9, 1)
"""

from __future__ import annotations
from typing import NamedTuple
import jax.numpy as jnp
import numpy as np
from src.registry import get_operators
from src.registry import lattice_operator


class Lattice(NamedTuple):
    """Immutable lattice velocity model — valid JAX pytree.

    All arrays are broadcast-compatible with the 5D population layout
    (nx, ny, nz, q, d). Directional indices are 1D and encode no spatial
    assumptions (direction -> q index mapping only).

    Attributes:
        name: Human-readable identifier, e.g. ``"D2Q9"``.
        d: Number of spatial dimensions.
        q: Number of discrete velocities.
        c: Velocity vectors, shape ``(1, 1, 1, q, d)`` — ``jax.Array``.
        w: Quadrature weights, shape ``(1, 1, 1, q, 1)`` — ``jax.Array``.
        opp_indices: Opposite-direction index for each velocity,
            shape ``(q,)`` — ``jax.Array`` of ints.
        main_indices: Indices of the cardinal (non-diagonal) directions,
            shape ``(m,)`` — ``jax.Array`` of ints.
        right_indices: Indices with positive x-component, shape ``(nr,)`` — ``jax.Array``.
        left_indices: Indices with negative x-component, shape ``(nl,)`` — ``jax.Array``.
        top_indices: Indices with positive y-component, shape ``(nt,)`` — ``jax.Array``.
        bottom_indices: Indices with negative y-component, shape ``(nb,)`` — ``jax.Array``.
        front_indices: Indices with positive z-component (3D only), shape ``(nf,)`` — ``jax.Array``.
        back_indices: Indices with negative z-component (3D only), shape ``(nbk,)`` — ``jax.Array``.
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
    front_indices: jnp.ndarray | None = None
    back_indices: jnp.ndarray | None = None


# ── D2Q9 constants (computed once at import time) ────────────────────

_D2Q9_CX = [0, 1, 0, -1, 0, 1, -1, -1, 1]
_D2Q9_CY = [0, 0, 1, 0, -1, 1, 1, -1, -1]
_D2Q9_W = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]


@lattice_operator(name="D2Q9", dim=2, q=9)
def _build_d2q9() -> Lattice:
    """Construct a D2Q9 :class:`Lattice`.

    Produces velocity vectors with shape (1, 1, 1, 9, 2) and weights
    with shape (1, 1, 1, 9, 1) for broadcast compatibility with
    the 5D population layout (nx, ny, nz, q, d).
    """
    c_np = jnp.array(list(zip(_D2Q9_CX, _D2Q9_CY, strict=False)))[None, None, None, :, :]  # shape (1, 1, 1, 9, 2)
    w_np = jnp.array(_D2Q9_W)[None, None, None, :, None]  # shape (1, 1, 1, 9, 1)

    opp = jnp.array([c_np[0, 0, 0].tolist().index((-ci).tolist()) for ci in c_np[0, 0, 0]])

    main = jnp.nonzero(np.sum(np.abs(c_np), axis=-1) == 1)[-1]

    right = jnp.nonzero(c_np[0, 0, 0][:, 0] == 1)[0]
    left = jnp.nonzero(c_np[0, 0, 0][:, 0] == -1)[0]
    top = jnp.nonzero(c_np[0, 0, 0][:, 1] == 1)[0]
    bottom = jnp.nonzero(c_np[0, 0, 0][:, 1] == -1)[0]

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
        front_indices=None,
        back_indices=None,
    )


# ── D3Q19 constants (computed once at import time) ────────────────────
_D3Q19_CX = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0]
_D3Q19_CY = [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1]
_D3Q19_CZ = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1]

_D3Q19_W = [
    1 / 3,  # Center (1)
    1 / 18,
    1 / 18,
    1 / 18,
    1 / 18,
    1 / 18,
    1 / 18,  # Faces (6)
    1 / 36,
    1 / 36,
    1 / 36,
    1 / 36,
    1 / 36,
    1 / 36,  # Edges (12)
    1 / 36,
    1 / 36,
    1 / 36,
    1 / 36,
    1 / 36,
    1 / 36,
]


@lattice_operator(name="D3Q19", dim=3, q=19)
def _build_d3q19() -> Lattice:
    """Construct a D3Q19 :class:`Lattice`.

    Produces velocity vectors with shape (1, 1, 1, 19, 3) and weights
    with shape (1, 1, 1, 19, 1) for broadcast compatibility with
    the 5D population layout (nx, ny, nz, q, d).
    """
    c_np = jnp.array(list(zip(_D3Q19_CX, _D3Q19_CY, _D3Q19_CZ, strict=False)))[
        None, None, None, :, :
    ]  # shape (1, 1, 1, 19, 3)
    w_np = jnp.array(_D3Q19_W)[None, None, None, :, None]  # shape (1, 1, 1, 19, 1)

    opp = jnp.array([c_np[0, 0, 0].tolist().index((-ci).tolist()) for ci in c_np[0, 0, 0]])

    main = jnp.nonzero(np.sum(np.abs(c_np), axis=-1) == 1)[-1]

    right = jnp.nonzero(c_np[0, 0, 0][:, 0] == 1)[0]
    left = jnp.nonzero(c_np[0, 0, 0][:, 0] == -1)[0]
    top = jnp.nonzero(c_np[0, 0, 0][:, 1] == 1)[0]
    bottom = jnp.nonzero(c_np[0, 0, 0][:, 1] == -1)[0]
    front = jnp.nonzero(c_np[0, 0, 0][:, 2] == 1)[0]
    back = jnp.nonzero(c_np[0, 0, 0][:, 2] == -1)[0]

    return Lattice(
        name="D3Q19",
        d=3,
        q=19,
        c=jnp.array(c_np),
        w=jnp.array(w_np),
        opp_indices=jnp.array(opp),
        main_indices=jnp.array(main),
        right_indices=jnp.array(right),
        left_indices=jnp.array(left),
        top_indices=jnp.array(top),
        bottom_indices=jnp.array(bottom),
        front_indices=jnp.array(front),
        back_indices=jnp.array(back),
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
        msg = f"Unsupported lattice type '{name}'. Supported: {supported}"
        raise ValueError(msg)
    target = lattices[key].target
    return target()
