"""LBM-stencil gradient operator — pure function.

Registered as ``("differential", "gradient")`` via ``@register_operator``.
Auto-discovered by ``auto_load_operators('operators.differential')``.

The gradient formula follows the standard LBM moment approach:

.. math::

    \\partial_\\alpha f = 3 \\sum_i w_i c_{i\\alpha} f(\\mathbf{x} + \\mathbf{c}_i)

where the off-centre neighbours are obtained by slicing the padded array.

Design
~~~~~~
*pad_mode* is a tuple of four ``jnp.pad`` mode strings:
``(right_y, left_y, bottom_x, top_x)`` applied in that order.  Because it
is a plain Python tuple of strings it must be treated as a *static* argument
when JIT-compiling — use ``jax.jit(fn, static_argnames=("pad_mode",))`` or
close over it to get a jittable closure.
"""

from __future__ import annotations
import jax.numpy as jnp
from operators.differential._pad_utils import apply_stencil_padding
from operators.differential._pad_utils import to_2d
from registry import register_operator


@register_operator("differential", name="gradient")
def compute_gradient(
    grid: jnp.ndarray,
    w: jnp.ndarray,
    c: jnp.ndarray,
    pad_mode: list | tuple,
) -> jnp.ndarray:
    """LBM-stencil gradient of a scalar field.

    ``pad_mode`` must be a compile-time constant (Python list/tuple of
    strings).  To JIT-compile calls to this function, use::

        jax.jit(compute_gradient, static_argnames=("pad_mode",))

    or close over *pad_mode* in a wrapper.

    Args:
        grid: Scalar field, shape ``(nx, ny, 1, 1)`` or ``(nx, ny)``.
        w: Lattice weights, shape ``(q,)``.
        c: Lattice velocity vectors, shape ``(2, q)``.
        pad_mode: Four padding modes ``(right_y, left_y, bottom_x, top_x)``.

    Returns:
        Gradient field, shape ``(nx, ny, 1, 2)``.
    """
    gp = apply_stencil_padding(to_2d(grid), tuple(pad_mode))
    return grad_core(gp, w, c)


def grad_core(
    padded: jnp.ndarray,
    w: jnp.ndarray,
    c: jnp.ndarray,
) -> jnp.ndarray:
    """Gradient kernel on an already-padded ``(nx+2, ny+2)`` array.

    Public so the wetting addon can reuse it after modifying ghost cells.

    Args:
        padded: Shape ``(nx + 2, ny + 2)``.
        w: Lattice weights, shape ``(q,)``.
        c: Lattice velocity vectors, shape ``(2, q)``.

    Returns:
        Gradient field, shape ``(nx, ny, 1, 2)``.
    """
    # Neighbour slices (D2Q9 directions 1–8; direction 0 cancels out)
    ip1_j0 = padded[2:, 1:-1]  # (i+1, j)
    im1_j0 = padded[:-2, 1:-1]  # (i-1, j)
    i0_jp1 = padded[1:-1, 2:]  # (i, j+1)
    i0_jm1 = padded[1:-1, :-2]  # (i, j-1)
    ip1_jp1 = padded[2:, 2:]  # (i+1, j+1)
    im1_jp1 = padded[:-2, 2:]  # (i-1, j+1)
    im1_jm1 = padded[:-2, :-2]  # (i-1, j-1)
    ip1_jm1 = padded[2:, :-2]  # (i+1, j-1)

    # x-component: sum over directions with non-zero cx
    gx = 3.0 * (
        w[1] * c[0, 1] * ip1_j0
        + w[3] * c[0, 3] * im1_j0
        + w[5] * c[0, 5] * ip1_jp1
        + w[6] * c[0, 6] * im1_jp1
        + w[7] * c[0, 7] * im1_jm1
        + w[8] * c[0, 8] * ip1_jm1
    )

    # y-component: sum over directions with non-zero cy
    gy = 3.0 * (
        w[2] * c[1, 2] * i0_jp1
        + w[4] * c[1, 4] * i0_jm1
        + w[5] * c[1, 5] * ip1_jp1
        + w[6] * c[1, 6] * im1_jp1
        + w[7] * c[1, 7] * im1_jm1
        + w[8] * c[1, 8] * ip1_jm1
    )

    nx = padded.shape[0] - 2
    ny = padded.shape[1] - 2
    out = jnp.zeros((nx, ny, 1, 2))
    out = out.at[:, :, 0, 0].set(gx)
    return out.at[:, :, 0, 1].set(gy)
