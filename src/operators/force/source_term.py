"""Forcing source term — pure function.

Implements the well-balanced forcing scheme for LBM.

Uses pre-built :class:`~operators.differential.operators.DifferentialOperators`
for the density gradient (with correct per-edge padding).
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
import numpy as np
from registry import force_model
from setup.lattice import Lattice

if TYPE_CHECKING:
    from operators.differential.operators import DifferentialOperators


@force_model(name="source_term_wb")
def source(
    rho: jnp.ndarray,
    u: jnp.ndarray,
    force: jnp.ndarray,
    lattice: Lattice,
    *,
    diff_ops: DifferentialOperators,
) -> jnp.ndarray:
    """Compute the well-balanced forcing source term.

    Args:
        rho: Density field, shape ``(nx, ny, 1, 1)``.
        u: Velocity field, shape ``(nx, ny, 1, 2)``.
        force: Force field, shape ``(nx, ny, 1, 2)``.
        lattice: :class:`~setup.lattice.Lattice`.
        diff_ops: Pre-built
            :class:`~operators.differential.operators.DifferentialOperators`.
            ``diff_ops.grad_standard`` is used for the density gradient.

    Returns:
        Source term, shape ``(nx, ny, q, 1)``.
    """
    q = lattice.q
    d = lattice.d

    # Pre-extract lattice data as numpy for JIT safety
    w_np = np.array(lattice.w)
    c_np = np.array(lattice.c)  # (d, q) as numpy — avoids tracing issues
    cx_np = c_np[0]
    cy_np = c_np[1]

    # Extract 2D slices
    ux = u[:, :, 0, 0]
    uy = u[:, :, 0, 1]
    fx = force[:, :, 0, 0]
    fy = force[:, :, 0, 1]
    rho_2d = rho[:, :, 0, 0]

    # Density gradient via LBM-stencil operator
    grad_rho_4d = diff_ops.grad_standard(rho)  # (nx, ny, 1, 2)
    grad_rho_x = grad_rho_4d[:, :, 0, 0]
    grad_rho_y = grad_rho_4d[:, :, 0, 1]

    # Corrected force
    fx_cor = fx + grad_rho_x / 3.0
    fy_cor = fy + grad_rho_y / 3.0

    nx_grid, ny_grid = rho_2d.shape
    source_3d = jnp.zeros((nx_grid, ny_grid, q))

    for i in range(q):
        cxi = float(cx_np[i])
        cyi = float(cy_np[i])
        wi = float(w_np[i])

        cu = cxi * ux + cyi * uy
        cf = cxi * fx + cyi * fy
        cf_cor = cxi * fx_cor + cyi * fy_cor
        uf_cor = ux * fx_cor + uy * fy_cor
        u_grad_rho = ux * grad_rho_x + uy * grad_rho_y

        source_3d = source_3d.at[:, :, i].set(
            wi * (3.0 * cf + 9.0 * cf_cor * cu - 3.0 * uf_cor + 0.5 * (3.0 * (cxi * cxi + cyi * cyi) - d) * u_grad_rho),
        )

    return jnp.expand_dims(source_3d, axis=-1)
