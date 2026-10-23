r"""Equilibrium distribution computation — pure function.

Implements the standard second-order equilibrium, used by single-phase runs:

.. math::

    f_i^{\\text{eq}} = w_i \\rho \\left[
        1 + \\frac{\\mathbf{c}_i \\cdot \\mathbf{u}}{c_s^2}
        + \\frac{(\\mathbf{c}_i \\cdot \\mathbf{u})^2}{2 c_s^4}
        - \\frac{\\mathbf{u} \\cdot \\mathbf{u}}{2 c_s^2}
    \\right]

with :math:`c_s^2 = 1/3`.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.registry import equilibrium_operator

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice


@equilibrium_operator(name="standard_equilibrium")
def compute_equilibrium(
    rho: jnp.ndarray,
    u: jnp.ndarray,
    lattice: Lattice,
) -> jnp.ndarray:
    """Compute the standard equilibrium distribution.

    Args:
        rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
        u: Velocity field, shape ``(nx, ny, nz, 1, 2)``.
        lattice: :class:`~setup.lattice.Lattice` with weights ``w``
            and velocity vectors ``c``.

    Returns:
        Equilibrium populations ``feq``, shape ``(nx, ny, nz, q, 1)``.
    """
    u2 = jnp.sum(u**2, axis=-1, keepdims=True)  # (nx, ny, nz, 1, 1)

    cu = jnp.sum(u * lattice.c, axis=-1, keepdims=True)  # (nx, ny, nz, q, 1)

    return lattice.w * rho * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2)  # (nx, ny, nz, q, 1)
