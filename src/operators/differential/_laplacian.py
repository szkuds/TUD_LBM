"""LBM-stencil Laplacian operator — pure function.

Registered as ``("differential", "laplacian")`` via ``@register_operator``.

The Laplacian formula follows the standard LBM isotropic stencil:

.. math::

    \\nabla^2 f = 6 \\sum_i w_i \\bigl[f(\\mathbf{x} + \\mathbf{c}_i) - f(\\mathbf{x})\\bigr]

where the factor 6 restores the correct Laplacian coefficient for the D2Q9
lattice (``c_s^2 = 1/3``).
"""

from __future__ import annotations
import jax.numpy as jnp
from operators.differential._pad_utils import apply_stencil_padding
from operators.differential._pad_utils import to_2d
from registry import register_operator


@register_operator("differential", name="laplacian")
def compute_laplacian(
    grid: jnp.ndarray,
    w: jnp.ndarray,
    pad_mode: list | tuple,
) -> jnp.ndarray:
    """LBM-stencil Laplacian of a scalar field.

    ``pad_mode`` must be a compile-time constant (Python list/tuple of
    strings).  To JIT-compile calls to this function, use::

        jax.jit(compute_laplacian, static_argnames=("pad_mode",))

    or close over *pad_mode* in a wrapper.

    Args:
        grid: Scalar field, shape ``(nx, ny, 1, 1)`` or ``(nx, ny)``.
        w: Lattice weights, shape ``(q,)``.
        pad_mode: Four padding modes ``(right_y, left_y, bottom_x, top_x)``.

    Returns:
        Laplacian field, shape ``(nx, ny, 1, 1)``.
    """
    gp = apply_stencil_padding(to_2d(grid), tuple(pad_mode))
    return lap_core(gp, w)


def lap_core(
    padded: jnp.ndarray,
    w: jnp.ndarray,
) -> jnp.ndarray:
    """Laplacian kernel on an already-padded ``(nx+2, ny+2)`` array.

    Public so the wetting addon can reuse it after modifying ghost cells.

    Args:
        padded: Shape ``(nx + 2, ny + 2)``.
        w: Lattice weights, shape ``(q,)``.

    Returns:
        Laplacian field, shape ``(nx, ny, 1, 1)``.
    """
    i0 = padded[1:-1, 1:-1]  # centre values

    lap = (
        6.0
        * (
            w[1] * (padded[2:, 1:-1] - i0)  # (i+1, j)
            + w[2] * (padded[1:-1, 2:] - i0)  # (i, j+1)
            + w[3] * (padded[:-2, 1:-1] - i0)  # (i-1, j)
            + w[4] * (padded[1:-1, :-2] - i0)  # (i, j-1)
            + w[5] * (padded[2:, 2:] - i0)  # (i+1, j+1)
            + w[6] * (padded[:-2, 2:] - i0)  # (i-1, j+1)
            + w[7] * (padded[:-2, :-2] - i0)  # (i-1, j-1)
            + w[8] * (padded[2:, :-2] - i0)  # (i+1, j-1)
        )
    )

    nx = padded.shape[0] - 2
    ny = padded.shape[1] - 2
    out = jnp.zeros((nx, ny, 1, 1))
    return out.at[:, :, 0, 0].set(lap)
