"""Forcing source term — pure function.

Implements the well-balanced forcing scheme for LBM.
The density gradient is computed with periodic central differences.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from setup.lattice import Lattice
from registry import force_model


def _gradient_rho_periodic(rho_2d: jnp.ndarray):
    """Central-difference gradient on a periodic 2D field."""
    drho_dx = (jnp.roll(rho_2d, -1, axis=0) - jnp.roll(rho_2d, 1, axis=0)) / 2.0
    drho_dy = (jnp.roll(rho_2d, -1, axis=1) - jnp.roll(rho_2d, 1, axis=1)) / 2.0
    return drho_dx, drho_dy


@force_model(name="source_term")
def source(
    rho: jnp.ndarray,
    u: jnp.ndarray,
    force: jnp.ndarray,
    lattice: Lattice,
) -> jnp.ndarray:
    """Compute the well-balanced forcing source term.

    Args:
        rho: Density field, shape ``(nx, ny, 1, 1)``.
        u: Velocity field, shape ``(nx, ny, 1, 2)``.
        force: Force field, shape ``(nx, ny, 1, 2)``.
        lattice: :class:`~setup.lattice.Lattice`.

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

    # Density gradient (periodic central differences)
    grad_rho_x, grad_rho_y = _gradient_rho_periodic(rho_2d)

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
            wi
            * (
                3.0 * cf
                + 9.0 * cf_cor * cu
                - 3.0 * uf_cor
                + 0.5 * (3.0 * (cxi * cxi + cyi * cyi) - d) * u_grad_rho
            )
        )

    return jnp.expand_dims(source_3d, axis=-1)
