"""Equilibrium distribution computation — pure function.

Extracted from :class:`simulation_operators.equilibrium.EquilibriumWB`.
Implements the well-balanced equilibrium used throughout TUD-LBM:

.. math::

    f_i^{\\text{eq}} = w_i \\rho \\left[
        1 + \\frac{\\mathbf{c}_i \\cdot \\mathbf{u}}{c_s^2}
        + \\frac{(\\mathbf{c}_i \\cdot \\mathbf{u})^2}{2 c_s^4}
        - \\frac{\\mathbf{u} \\cdot \\mathbf{u}}{2 c_s^2}
    \\right]

with :math:`c_s^2 = 1/3`.

The *rest direction* (``i = 0``) is computed via mass conservation:
``feq_0 = rho − Σ_{i>0} feq_i``, which matches the legacy
``EquilibriumWB`` class exactly.
"""

from __future__ import annotations

import jax.numpy as jnp

from setup.lattice import Lattice
from registry import equilibrium_operator


@equilibrium_operator(name="wb")
def compute_equilibrium(
    rho: jnp.ndarray,
    u: jnp.ndarray,
    lattice: Lattice,
) -> jnp.ndarray:
    """Compute the well-balanced equilibrium distribution.

    Args:
        rho: Density field, shape ``(nx, ny, 1, 1)``.
        u: Velocity field, shape ``(nx, ny, 1, 2)``.
        lattice: :class:`~setup.lattice.Lattice` with weights ``w``
            and velocity vectors ``c``.

    Returns:
        Equilibrium populations ``feq``, shape ``(nx, ny, q, 1)``.
    """
    w = lattice.w  # (q,)
    cx = lattice.c[0]  # (q,)
    cy = lattice.c[1]  # (q,)
    q = lattice.q

    # Extract 2D fields ─ shape (nx, ny)
    ux = u[:, :, 0, 0]
    uy = u[:, :, 0, 1]
    rho_2d = rho[:, :, 0, 0]

    u2 = ux * ux + uy * uy

    nx, ny = rho_2d.shape
    feq = jnp.zeros((nx, ny, q, 1))

    # Directions 1 … q-1 (standard formula)
    for i in range(1, q):
        cu = cx[i] * ux + cy[i] * uy
        feq = feq.at[:, :, i, 0].set(
            w[i] * rho_2d * (3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
        )

    # Rest direction via mass conservation
    f_sum = jnp.sum(feq[:, :, 1:, 0], axis=2)
    feq = feq.at[:, :, 0, 0].set(rho_2d - f_sum)

    return feq
