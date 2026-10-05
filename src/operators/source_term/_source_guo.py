r"""Guo forcing source term — pure function.

The source term that pairs with the ``"standard_equilibrium"`` equilibrium of a
single-phase run:

.. math::

    S_i = w_i \left[
        \frac{\mathbf{c}_i \cdot \mathbf{F}}{c_s^2}
        + \frac{(\mathbf{c}_i \cdot \mathbf{F})(\mathbf{c}_i \cdot \mathbf{u})}{c_s^4}
        - \frac{\mathbf{u} \cdot \mathbf{F}}{c_s^2}
    \right]

with :math:`c_s^2 = 1/3`. The collision operator applies the ``1 - omega/2``
prefactor.

This is the ``"wb"`` source without its density-gradient corrections. Those
compensate for the isotropic pressure the ``"wb"`` equilibrium leaves out; the
standard equilibrium carries that pressure itself, so they do not belong here.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.registry import source_term_operator

if TYPE_CHECKING:
    from src.lattice.lattice import Lattice
    from src.operators.protocols import DifferentialOperator


@source_term_operator(name="guo")
def compute_source(
    rho: jnp.ndarray,  # noqa: ARG001  # required by SourceTermOperator protocol
    u: jnp.ndarray,
    force: jnp.ndarray,
    lattice: Lattice,
    *,
    gradient: DifferentialOperator,  # noqa: ARG001  # required by SourceTermOperator protocol
) -> jnp.ndarray:
    """Compute the Guo forcing source term.

    Args:
        rho: Density field (unused), shape ``(nx, ny, nz, 1, 1)``.
        u: Velocity field, shape ``(nx, ny, nz, 1, d)``.
        force: Force field, shape ``(nx, ny, nz, 1, d)``.
        lattice: :class:`~setup.lattice.Lattice`.
        gradient: Density gradient operator (unused).

    Returns:
        Source term, shape ``(nx, ny, nz, q, 1)``.
    """
    cu = jnp.sum(lattice.c * u, axis=-1, keepdims=True)  # (nx, ny, nz, q, 1)
    cf = jnp.sum(lattice.c * force, axis=-1, keepdims=True)  # (nx, ny, nz, q, 1)
    uf = jnp.sum(u * force, axis=-1, keepdims=True)  # (nx, ny, nz, 1, 1)

    return lattice.w * (3.0 * cf + 9.0 * cf * cu - 3.0 * uf)  # (nx, ny, nz, q, 1)
