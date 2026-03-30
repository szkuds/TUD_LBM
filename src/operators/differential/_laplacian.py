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
from registry import register_operator
from operators.differential._pad_utils import apply_stencil_padding, to_2d


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
    grid_2d = to_2d(grid)
    gp = apply_stencil_padding(grid_2d, tuple(pad_mode))

    i0 = gp[1:-1, 1:-1]  # centre values

    lap = (
        6.0
        * (
            w[1] * (gp[2:, 1:-1] - i0)  # (i+1, j)
            + w[2] * (gp[1:-1, 2:] - i0)  # (i, j+1)
            + w[3] * (gp[:-2, 1:-1] - i0)  # (i-1, j)
            + w[4] * (gp[1:-1, :-2] - i0)  # (i, j-1)
            + w[5] * (gp[2:, 2:] - i0)  # (i+1, j+1)
            + w[6] * (gp[:-2, 2:] - i0)  # (i-1, j+1)
            + w[7] * (gp[:-2, :-2] - i0)  # (i-1, j-1)
            + w[8] * (gp[2:, :-2] - i0)  # (i+1, j-1)
        )
    )

    nx, ny = grid_2d.shape
    out = jnp.zeros((nx, ny, 1, 1))
    return out.at[:, :, 0, 0].set(lap)
