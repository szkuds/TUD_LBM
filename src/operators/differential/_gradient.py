r"""LBM-stencil gradient operator — pure function.

Resolved through :func:`~src.operators.differential.build_differential`.

The gradient formula follows the standard LBM moment approach:

.. math::

    \\partial_\\alpha f = 3 \\sum_i w_i c_{i\\alpha} f(\\mathbf{x} + \\mathbf{c}_i)

where the off-centre neighbours are obtained by slicing the padded array. The
factor 3 holds for any lattice with ``sum_i w_i c_i c_i = I / 3``, so one kernel
serves D2Q9 and D3Q19: the sum runs over the rows of ``lattice.c``.
"""

from __future__ import annotations
import functools
import operator
from typing import TYPE_CHECKING
import jax.numpy as jnp
import numpy as np
from src.operators.differential._pad_utils import _apply_stencil_padding
from src.operators.differential._pad_utils import neighbour
from src.operators.differential._pad_utils import spatial

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Any
    from src.protocols import ExtDifferentialOperator
    from src.protocols import ExtGhostFill


def build_gradient(
    w: jnp.ndarray, c: jnp.ndarray, pad_mode: Sequence[str], ghost_fill: ExtGhostFill | None = None
) -> ExtDifferentialOperator:
    """Return the gradient closure over the lattice, *pad_mode* and an optional ghost fill.

    Args:
        w: Lattice weights, shape ``(1, 1, 1, q, 1)``.
        c: Lattice velocity vectors, shape ``(1, 1, 1, q, d)``.
        pad_mode: Padding modes ``(top, bottom, right, left)``, then ``(front, back)`` in 3D.
        ghost_fill: Rewrites the padded field's ghost layer from ``ext`` before
            the stencil, or ``None`` for the plain stencil.

    Returns:
        ``grad(grid, ext) → (nx, ny, nz, 1, d)``. Without a fill, *ext* is ignored.
    """
    _pad_mode = tuple(pad_mode)
    dim = c.shape[-1]

    def gradient(grid: jnp.ndarray, ext: Mapping[str, Any]) -> jnp.ndarray:
        padded = _apply_stencil_padding(spatial(grid, dim), _pad_mode)
        if ghost_fill is not None:
            padded = ghost_fill(padded, ext)
        # Written component by component into zeros, not stacked: under jit, XLA
        # fuses a stack differently and moves the 2D gradient by one ulp.
        out = jnp.zeros((*grid.shape[:3], 1, dim), dtype=padded.dtype)
        for axis, component in enumerate(grad_components(padded, w, c)):
            out = out.at[..., 0, axis].set(component.reshape(grid.shape[:3]))
        return out

    return gradient


def grad_components(padded: jnp.ndarray, w: jnp.ndarray, c: jnp.ndarray) -> list[jnp.ndarray]:
    """Gradient kernel on a field padded by one ghost layer on every edge.

    Each component sums only the directions with a non-zero velocity along it,
    in lattice order.

    Args:
        padded: Shape ``(nx + 2, ny + 2)``, or ``(nx + 2, ny + 2, nz + 2)``.
        w: Lattice weights, shape ``(1, 1, 1, q, 1)``.
        c: Lattice velocity vectors, shape ``(1, 1, 1, q, d)``, ``d == padded.ndim``.

    Returns:
        The ``d`` gradient components, each of shape ``(nx, ny[, nz])``.
    """
    weights = np.asarray(w)[0, 0, 0, :, 0]
    directions = np.asarray(c)[0, 0, 0]
    components = []
    for axis in range(directions.shape[1]):
        terms = [
            float(weight * direction[axis]) * neighbour(padded, direction)
            for weight, direction in zip(weights, directions, strict=True)
            if direction[axis] != 0
        ]
        components.append(3.0 * functools.reduce(operator.add, terms))
    return components
