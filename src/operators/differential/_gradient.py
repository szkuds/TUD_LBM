r"""LBM-stencil gradient operator — pure function.

Resolved through :func:`~src.operators.differential.build_differential`.

The gradient formula follows the standard LBM moment approach:

.. math::

    \\partial_\\alpha f = 3 \\sum_i w_i c_{i\\alpha} f(\\mathbf{x} + \\mathbf{c}_i)

where the off-centre neighbours are obtained by slicing the padded array.

Design
~~~~~~
*pad_mode* is a tuple of four ``jnp.pad`` mode strings:
``(top, bottom, right, left)``, i.e. ``(y_end, y_start, x_end, x_start)``.  Because it
is a plain Python tuple of strings it must be treated as a *static* argument
when JIT-compiling — use ``jax.jit(fn, static_argnames=("pad_mode",))`` or
close over it to get a jittable closure.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.operators.differential._pad_utils import _apply_stencil_padding
from src.operators.differential._pad_utils import to_2d

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Any
    from src.addons.hooks import ExtGhostFill
    from src.operators.protocols import ExtDifferentialOperator


def compute_gradient(
    grid: jnp.ndarray,
    w: jnp.ndarray,
    c: jnp.ndarray,
    pad_mode: Sequence[str],
) -> jnp.ndarray:
    """LBM-stencil gradient of a scalar field.

    ``pad_mode`` must be a compile-time constant (Python list/tuple of
    strings).  To JIT-compile calls to this function, use::

        jax.jit(compute_gradient, static_argnames=("pad_mode",))

    or close over *pad_mode* in a wrapper.

    Args:
        grid: Scalar field, shape ``(nx, ny, nz, 1, 1)`` or ``(nx, ny)``.
        w: Lattice weights, shape ``(1, 1, 1, q, 1)``.
        c: Lattice velocity vectors, shape ``(1, 1, 1, q, 2)``.
        pad_mode: Four padding modes ``(top, bottom, right, left)``.

    Returns:
        Gradient field, shape ``(nx, ny, nz, 1, 2)``.
    """
    grid_padded = _apply_stencil_padding(to_2d(grid), tuple(pad_mode))
    return grad_core_2d(grid_padded, w, c)


def build_gradient(
    w: jnp.ndarray, c: jnp.ndarray, pad_mode: Sequence[str], ghost_fill: ExtGhostFill | None = None
) -> ExtDifferentialOperator:
    """Return the gradient closure over the lattice, *pad_mode* and an optional ghost fill.

    Args:
        w: Lattice weights, shape ``(1, 1, 1, q, 1)``.
        c: Lattice velocity vectors, shape ``(1, 1, 1, q, 2)``.
        pad_mode: Four padding modes ``(top, bottom, right, left)``.
        ghost_fill: Rewrites the padded field's ghost layer from ``ext`` before
            the stencil, or ``None`` for the plain stencil.

    Returns:
        ``grad(grid, ext) → (nx, ny, nz, 1, 2)``. Without a fill, *ext* is ignored.
    """
    _pad_mode = tuple(pad_mode)

    def gradient(grid: jnp.ndarray, ext: Mapping[str, Any]) -> jnp.ndarray:
        padded = _apply_stencil_padding(to_2d(grid), _pad_mode)
        if ghost_fill is not None:
            padded = ghost_fill(padded, ext)
        return grad_core_2d(padded, w, c)

    return gradient


def grad_core_2d(
    padded: jnp.ndarray,
    w: jnp.ndarray,
    c: jnp.ndarray,
) -> jnp.ndarray:
    """Gradient kernel on an already-padded ``(nx+2, ny+2)`` array.

    Args:
        padded: Shape ``(nx + 2, ny + 2)``.
        w: Lattice weights, shape ``(q,)``.
        c: Lattice velocity vectors, shape ``(2, q)``.

    Returns:
        Gradient field, shape ``(nx, ny, 1, 1, 2)``.
    """
    # Neighbour slices (D2Q9 directions 1-8; direction 0 cancels out)
    ip1_j0 = padded[2:, 1:-1]  # (i+1, j)
    im1_j0 = padded[:-2, 1:-1]  # (i-1, j)
    i0_jp1 = padded[1:-1, 2:]  # (i, j+1)
    i0_jm1 = padded[1:-1, :-2]  # (i, j-1)
    ip1_jp1 = padded[2:, 2:]  # (i+1, j+1)
    im1_jp1 = padded[:-2, 2:]  # (i-1, j+1)
    im1_jm1 = padded[:-2, :-2]  # (i-1, j-1)
    ip1_jm1 = padded[2:, :-2]  # (i+1, j-1)

    w_flat = w[0, 0, 0, :, 0]
    c_flat = c[0, 0, 0, :, :]

    # x-component: sum over directions with non-zero cx
    gx = 3.0 * (
        w_flat[1] * c_flat[1, 0] * ip1_j0
        + w_flat[3] * c_flat[3, 0] * im1_j0
        + w_flat[5] * c_flat[5, 0] * ip1_jp1
        + w_flat[6] * c_flat[6, 0] * im1_jp1
        + w_flat[7] * c_flat[7, 0] * im1_jm1
        + w_flat[8] * c_flat[8, 0] * ip1_jm1
    )

    # y-component: sum over directions with non-zero cy
    gy = 3.0 * (
        w_flat[2] * c_flat[2, 1] * i0_jp1
        + w_flat[4] * c_flat[4, 1] * i0_jm1
        + w_flat[5] * c_flat[5, 1] * ip1_jp1
        + w_flat[6] * c_flat[6, 1] * im1_jp1
        + w_flat[7] * c_flat[7, 1] * im1_jm1
        + w_flat[8] * c_flat[8, 1] * ip1_jm1
    )

    nx = padded.shape[0] - 2
    ny = padded.shape[1] - 2
    nz = 1  # Pseudo-3D: stencil operates on nz=1, output preserves it
    out = jnp.zeros((nx, ny, nz, 1, 2))
    out = out.at[:, :, 0, 0, 0].set(gx)
    return out.at[:, :, 0, 0, 1].set(gy)
