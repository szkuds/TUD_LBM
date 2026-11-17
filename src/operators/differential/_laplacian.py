r"""LBM-stencil Laplacian operator — pure function.

Resolved through :func:`~src.operators.differential.build_differential`.

The Laplacian formula follows the standard LBM isotropic stencil:

.. math::

    \\nabla^2 f = 6 \\sum_i w_i \\bigl[f(\\mathbf{x} + \\mathbf{c}_i) - f(\\mathbf{x})\\bigr]

The factor 6 holds for any lattice with ``sum_i w_i c_i c_i = I / 3``
(``c_s^2 = 1/3``), so one kernel serves D2Q9 and D3Q19.
"""

from __future__ import annotations
import functools
import operator
from typing import TYPE_CHECKING
import numpy as np
from src.operators.differential._pad_utils import _apply_stencil_padding
from src.operators.differential._pad_utils import neighbour
from src.operators.differential._pad_utils import spatial

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Any
    import jax.numpy as jnp
    from src.protocols import ExtDifferentialOperator
    from src.protocols import ExtGhostFill


def build_laplacian(
    w: jnp.ndarray, c: jnp.ndarray, pad_mode: Sequence[str], ghost_fill: ExtGhostFill | None = None
) -> ExtDifferentialOperator:
    """Return the Laplacian closure over the lattice, *pad_mode* and an optional ghost fill.

    Args:
        w: Lattice weights, shape ``(1, 1, 1, q, 1)``.
        c: Lattice velocity vectors, shape ``(1, 1, 1, q, d)``.
        pad_mode: Padding modes ``(top, bottom, right, left)``, then ``(front, back)`` in 3D.
        ghost_fill: Rewrites the padded field's ghost layer from ``ext`` before
            the stencil, or ``None`` for the plain stencil.

    Returns:
        ``lap(grid, ext) → (nx, ny, nz, 1, 1)``. Without a fill, *ext* is ignored.
    """
    _pad_mode = tuple(pad_mode)
    dim = c.shape[-1]

    def laplacian(grid: jnp.ndarray, ext: Mapping[str, Any]) -> jnp.ndarray:
        padded = _apply_stencil_padding(spatial(grid, dim), _pad_mode)
        if ghost_fill is not None:
            padded = ghost_fill(padded, ext)
        return lap_core(padded, w, c).reshape(*grid.shape[:3], 1, 1)

    return laplacian


def lap_core(padded: jnp.ndarray, w: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """Laplacian kernel on a field padded by one ghost layer on every edge.

    Args:
        padded: Shape ``(nx + 2, ny + 2)``, or ``(nx + 2, ny + 2, nz + 2)``.
        w: Lattice weights, shape ``(1, 1, 1, q, 1)``.
        c: Lattice velocity vectors, shape ``(1, 1, 1, q, d)``, ``d == padded.ndim``.

    Returns:
        Laplacian field, shape ``(nx, ny[, nz])``.
    """
    directions = np.asarray(c)[0, 0, 0]
    centre = neighbour(padded, np.zeros(directions.shape[1], dtype=int))
    terms = [
        float(weight) * (neighbour(padded, direction) - centre)
        for weight, direction in zip(np.asarray(w)[0, 0, 0, :, 0], directions, strict=True)
        if direction.any()
    ]
    return 6.0 * functools.reduce(operator.add, terms)
