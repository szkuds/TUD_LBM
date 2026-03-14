"""LBM-stencil gradient operator — pure functions.

Provides :func:`compute_gradient`, an LBM-weighted gradient of a scalar
field, and :func:`make_wetting_gradient`, a setup-time factory that bakes
wetting ghost-cell corrections into a jitted closure.

The gradient formula follows the standard LBM moment approach:

.. math::

    \\partial_\\alpha f = 3 \\sum_i w_i c_{i\\alpha} f(\\mathbf{x} + \\mathbf{c}_i)

where the off-centre neighbours are obtained by slicing the padded array.

Design
~~~~~~
*pad_mode* is a list of four ``jnp.pad`` mode strings:
``[right_y, left_y, bottom_x, top_x]`` applied in that order.  Because it
is a plain Python list of strings it must be treated as a *static* argument
when JIT-compiling — use ``jax.jit(fn, static_argnames=("pad_mode",))`` or
close over it (as :func:`make_wetting_gradient` does) to get a jittable
closure.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from registry import register_operator


@register_operator("differential", name="gradient")
def compute_gradient(
    grid: jnp.ndarray,
    w: jnp.ndarray,
    c: jnp.ndarray,
    pad_mode: list,
) -> jnp.ndarray:
    """LBM-stencil gradient of a scalar field.

    ``pad_mode`` must be a compile-time constant (Python list of strings).
    To JIT-compile calls to this function, use::

        jax.jit(compute_gradient, static_argnames=("pad_mode",))

    or close over *pad_mode* in a wrapper (see :func:`make_wetting_gradient`).

    Args:
        grid: Scalar field, shape ``(nx, ny, 1, 1)`` or ``(nx, ny)``.
        w: Lattice weights, shape ``(q,)``.
        c: Lattice velocity vectors, shape ``(2, q)``.
        pad_mode: Four padding modes ``[right_y, left_y, bottom_x, top_x]``.

    Returns:
        Gradient field, shape ``(nx, ny, 1, 2)``.
    """
    grid_2d = grid[:, :, 0, 0] if grid.ndim == 4 else grid

    # Apply four pads to obtain one ghost cell on every side
    gp = jnp.pad(grid_2d, ((0, 0), (0, 1)), mode=pad_mode[0])
    gp = jnp.pad(gp, ((0, 0), (1, 0)), mode=pad_mode[1])
    gp = jnp.pad(gp, ((0, 1), (0, 0)), mode=pad_mode[2])
    gp = jnp.pad(gp, ((1, 0), (0, 0)), mode=pad_mode[3])

    # Neighbour slices (D2Q9 directions 1–8; direction 0 cancels out)
    ip1_j0 = gp[2:, 1:-1]  # (i+1, j)
    im1_j0 = gp[:-2, 1:-1]  # (i-1, j)
    i0_jp1 = gp[1:-1, 2:]  # (i, j+1)
    i0_jm1 = gp[1:-1, :-2]  # (i, j-1)
    ip1_jp1 = gp[2:, 2:]  # (i+1, j+1)
    im1_jp1 = gp[:-2, 2:]  # (i-1, j+1)
    im1_jm1 = gp[:-2, :-2]  # (i-1, j-1)
    ip1_jm1 = gp[2:, :-2]  # (i+1, j-1)

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

    nx, ny = grid_2d.shape
    out = jnp.zeros((nx, ny, 1, 2))
    out = out.at[:, :, 0, 0].set(gx)
    out = out.at[:, :, 0, 1].set(gy)
    return out


def make_wetting_gradient(
    w: jnp.ndarray,
    c: jnp.ndarray,
    pad_mode: list,
    wetting_params: dict,
    chemical_step: int | None = None,
):
    """Return a jitted gradient closure with wetting ghost-cell correction.

    Call **once at setup time** and store the returned callable.  The
    ghost-cell correction is applied in Python (before JAX tracing) by
    resolving the wetting parameters and writing them into the padded array.
    The returned closure closes over *pad_mode* so JAX traces it as a
    compile-time constant — no ``static_argnums`` needed.

    Args:
        w: Lattice weights, shape ``(q,)``.
        c: Lattice velocity vectors, shape ``(2, q)``.
        pad_mode: Four padding modes ``[right_y, left_y, bottom_x, top_x]``.
        wetting_params: Dict with keys ``rho_l``, ``rho_v``, ``width``,
            and per-side wetting scalars (``phi_l``, ``phi_r``, ``d_rho_l``,
            ``d_rho_r``) **or** array-valued ``phi``/``drho`` with a
            ``chemical_step`` index.
        chemical_step: Optional step index for chemical-step simulations.

    Returns:
        ``_grad(grid) → jnp.ndarray`` of shape ``(nx, ny, 1, 2)``.
        The returned function is already jitted.
    """
    from operators.wetting.wetting_util import (
        apply_wetting_to_all_edges,
        resolve_wetting_fields,
    )

    phi_l, phi_r, d_rho_l, d_rho_r = resolve_wetting_fields(
        wetting_params, chemical_step
    )
    rho_l = wetting_params["rho_l"]
    rho_v = wetting_params["rho_v"]
    width = wetting_params["width"]

    # Build a static-pad-mode version of compute_gradient once
    _grad_jit = jax.jit(compute_gradient, static_argnames=("pad_mode",))

    @jax.jit
    def _grad(grid: jnp.ndarray) -> jnp.ndarray:
        grid_2d = grid[:, :, 0, 0] if grid.ndim == 4 else grid

        # Build padded field with one ghost cell on every side
        gp = jnp.pad(grid_2d, ((0, 0), (0, 1)), mode=pad_mode[0])
        gp = jnp.pad(gp, ((0, 0), (1, 0)), mode=pad_mode[1])
        gp = jnp.pad(gp, ((0, 1), (0, 0)), mode=pad_mode[2])
        gp = jnp.pad(gp, ((1, 0), (0, 0)), mode=pad_mode[3])

        # Overwrite bottom ghost row with wetting boundary values
        gp = apply_wetting_to_all_edges(
            gp, rho_l, rho_v, phi_l, phi_r, d_rho_l, d_rho_r, width
        )

        # Delegate to the plain gradient on the corrected interior
        return _grad_jit(gp[1:-1, 1:-1][:, :, None, None], w, c, tuple(pad_mode))

    return _grad
