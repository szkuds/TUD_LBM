r"""LBM-stencil Laplacian operator — pure function.

Resolved through :func:`~src.operators.differential.build_differential`.

The Laplacian formula follows the standard LBM isotropic stencil:

.. math::

    \\nabla^2 f = 6 \\sum_i w_i \\bigl[f(\\mathbf{x} + \\mathbf{c}_i) - f(\\mathbf{x})\\bigr]

where the factor 6 restores the correct Laplacian coefficient for the D2Q9
lattice (``c_s^2 = 1/3``).
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
import numpy as np
from src.operators.differential._pad_utils import _apply_stencil_padding
from src.operators.differential._pad_utils import neighbour
from src.operators.differential._pad_utils import to_2d
from src.operators.differential._pad_utils import to_3d

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Any
    from src.protocols import ExtDifferentialOperator
    from src.protocols import ExtGhostFill

SPATIAL_3D = 3


def compute_laplacian(
    grid: jnp.ndarray,
    w: jnp.ndarray,
    pad_mode: Sequence[str],
) -> jnp.ndarray:
    """LBM-stencil Laplacian of a scalar field.

    ``pad_mode`` must be a compile-time constant (Python list/tuple of
    strings).  To JIT-compile calls to this function, use::

        jax.jit(compute_laplacian, static_argnames=("pad_mode",))

    or close over *pad_mode* in a wrapper.

    Args:
        grid: Scalar field, shape ``(nx, ny, nz, 1, 1)`` or ``(nx, ny)``.
        w: Lattice weights, shape ``(q,)``.
        pad_mode: Four padding modes ``(top, bottom, right, left)``.

    Returns:
        Laplacian field, shape ``(nx, ny, nz, 1, 1)``.
    """
    grid_padded = _apply_stencil_padding(to_2d(grid), tuple(pad_mode))
    return lap_core_2d(grid_padded, w)


def build_laplacian(
    w: jnp.ndarray, c: jnp.ndarray, pad_mode: Sequence[str], ghost_fill: ExtGhostFill | None = None
) -> ExtDifferentialOperator:
    """Return the Laplacian closure over the lattice weights, *pad_mode* and an optional ghost fill.

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
    if c.shape[-1] == SPATIAL_3D:

        def laplacian_3d(grid: jnp.ndarray, ext: Mapping[str, Any]) -> jnp.ndarray:
            padded = _apply_stencil_padding(to_3d(grid), _pad_mode)
            if ghost_fill is not None:
                padded = ghost_fill(padded, ext)
            return lap_core_3d(padded, w, c)

        return laplacian_3d

    def laplacian(grid: jnp.ndarray, ext: Mapping[str, Any]) -> jnp.ndarray:
        padded = _apply_stencil_padding(to_2d(grid), _pad_mode)
        if ghost_fill is not None:
            padded = ghost_fill(padded, ext)
        return lap_core_2d(padded, w)

    return laplacian


def lap_core_3d(padded: jnp.ndarray, w: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """Laplacian kernel on an already-padded ``(nx+2, ny+2, nz+2)`` array.

    ``6 * sum_i w_i [f(x + c_i) - f(x)]``: the factor holds for any lattice with
    ``sum_i w_i c_i c_i = I / 3``, D3Q19 among them.

    Args:
        padded: Shape ``(nx + 2, ny + 2, nz + 2)``.
        w: Lattice weights, shape ``(1, 1, 1, q, 1)``.
        c: Lattice velocity vectors, shape ``(1, 1, 1, q, 3)``.

    Returns:
        Laplacian field, shape ``(nx, ny, nz, 1, 1)``.
    """
    centre = padded[1:-1, 1:-1, 1:-1]
    total = jnp.zeros_like(centre)
    for weight, direction in zip(np.asarray(w)[0, 0, 0, 1:, 0], np.asarray(c)[0, 0, 0, 1:], strict=True):
        total = total + float(weight) * (neighbour(padded, tuple(int(step) for step in direction)) - centre)
    return (6.0 * total)[..., None, None]


def lap_core_2d(
    padded: jnp.ndarray,
    w: jnp.ndarray,
) -> jnp.ndarray:
    """Laplacian kernel on an already-padded ``(nx+2, ny+2)`` array.

    Args:
        padded: Shape ``(nx + 2, ny + 2)``.
        w: Lattice weights, shape ``(1, 1, 1, q, 1)``.

    Returns:
        Laplacian field, shape ``(nx, ny, 1, 1, 1)``.
    """
    i0 = padded[1:-1, 1:-1]  # centre values

    w_flat = w[0, 0, 0, :, 0]

    lap = (
        6.0
        * (
            w_flat[1] * (padded[2:, 1:-1] - i0)  # (i+1, j)
            + w_flat[2] * (padded[1:-1, 2:] - i0)  # (i, j+1)
            + w_flat[3] * (padded[:-2, 1:-1] - i0)  # (i-1, j)
            + w_flat[4] * (padded[1:-1, :-2] - i0)  # (i, j-1)
            + w_flat[5] * (padded[2:, 2:] - i0)  # (i+1, j+1)
            + w_flat[6] * (padded[:-2, 2:] - i0)  # (i-1, j+1)
            + w_flat[7] * (padded[:-2, :-2] - i0)  # (i-1, j-1)
            + w_flat[8] * (padded[2:, :-2] - i0)  # (i+1, j-1)
        )
    )

    nx = padded.shape[0] - 2
    ny = padded.shape[1] - 2
    nz = 1  # Pseudo-3D: stencil operates on nz=1, output preserves it
    out = jnp.zeros((nx, ny, nz, 1, 1))
    return out.at[:, :, 0, 0, 0].set(lap)
