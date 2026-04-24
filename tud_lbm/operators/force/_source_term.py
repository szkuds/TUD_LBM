"""Forcing source term — pure function.

Implements the well-balanced forcing scheme for LBM.

Uses the density gradient operator for computing gravity corrections.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
import numpy as np
from tud_lbm.registry import force_model

if TYPE_CHECKING:
    from tud_lbm.lattice.lattice import Lattice
    from tud_lbm.operators.protocols import DifferentialOperator


@force_model(name="source_term_wb")
def source(
    rho: jnp.ndarray,
    u: jnp.ndarray,
    force: jnp.ndarray,
    lattice: Lattice,
    *,
    gradient: DifferentialOperator,
) -> jnp.ndarray:
    """Compute the well-balanced forcing source term.

    Args:
        rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
        u: Velocity field, shape ``(nx, ny, nz, 1, 2)``.
        force: Force field, shape ``(nx, ny, nz, 1, 2)``.
        lattice: :class:`~setup.lattice.Lattice`.
        gradient: Standard LBM-stencil gradient callable
            (grid) → gradient. Used for density gradient.

    Returns:
        Source term, shape ``(nx, ny, nz, q, 1)``.
    """
    # Support pseudo-3D (nz=1) — stencil logic remains 2D
    # but array shapes include nz dimension for compatibility

    q = lattice.q
    d = lattice.d

    # Pre-extract lattice data as numpy for JIT safety
    w_np = np.array(lattice.w)[0, 0, 0, :, 0]  # (q,) as numpy — avoids tracing issues
    c_np = np.array(lattice.c)[0, 0, 0].T  # (d, q) as numpy — avoids tracing issues
    cx_np = c_np[0]
    cy_np = c_np[1]

    # Extract 2D slices
    ux = u[:, :, 0, 0, 0]
    uy = u[:, :, 0, 0, 1]
    fx = force[:, :, 0, 0, 0]
    fy = force[:, :, 0, 0, 1]
    rho_2d = rho[:, :, 0, 0, 0]

    # Density gradient via LBM-stencil operator
    grad_rho_5d = gradient(rho)  # (nx, ny, nz, 1, 2)
    grad_rho_x = grad_rho_5d[:, :, 0, 0, 0]
    grad_rho_y = grad_rho_5d[:, :, 0, 0, 1]

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

    return jnp.expand_dims(source_3d, axis=(-3, -1))
