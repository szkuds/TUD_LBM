r"""Equilibrium distribution computation — pure function.

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
``feq_0 = rho - Σ_{i>0} feq_i``, which matches the legacy
``EquilibriumWB`` class exactly.

An optional ``viscous_stress`` ``A*S`` adds ``cs2*rho*A*S`` to the second
moment (Zhang, Guo & Wang 2022, Eq. 25), decoupling the viscosity from the
shear relaxation rate; see :mod:`._viscous_stress`. Its rest-population share
comes from mass conservation like the rest of ``f_0`` — note that the PDF's
explicit ``f_0`` carries ``+ w_0 rho A div u``, which violates ``sum feq = rho``;
the correct sign is minus, and the mass balance gives it. An optional
``reference_pressure`` ``p_g`` adds ``w_i p_g / cs2`` (Eq. 25), with its
``-(1 - w_0) p_g / cs2`` rest share again from mass conservation.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.operators.equilibrium._viscous_stress import viscous_stress_term
from src.registry import equilibrium_operator

if TYPE_CHECKING:
    from src.lattice import Lattice

_CS2 = 1.0 / 3.0


@equilibrium_operator(name="wb")
def compute_equilibrium(
    rho: jnp.ndarray,
    u: jnp.ndarray,
    lattice: Lattice,
    viscous_stress: jnp.ndarray | None = None,
    reference_pressure: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute the well-balanced equilibrium distribution.

    Args:
        rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
        u: Velocity field, shape ``(nx, ny, nz, 1, 2)``.
        lattice: :class:`~setup.lattice.Lattice` with weights ``w``
            and velocity vectors ``c``.
        viscous_stress: ``A*S``, shape ``(nx, ny, nz, 1, d, d)``, or ``None``
            for the plain well-balanced equilibrium.
        reference_pressure: ``p_g``, shape ``(nx, ny, nz, 1, 1)``, adding
            ``p_g I`` to the second moment (the hydrostatic weight of the
            reference density), or ``None``.

    Returns:
        Equilibrium populations ``feq``, shape ``(nx, ny, nz, q, 1)``.
    """
    cu = jnp.sum(lattice.c * u, axis=-1, keepdims=True)  # (nx, ny, nz, q, 1)
    u2 = jnp.sum(u**2, axis=-1, keepdims=True)  # (nx, ny, nz, 1, 1)

    feq_rest = lattice.w[..., 1:, :] * rho * (3.0 * cu[..., 1:, :] + 4.5 * cu[..., 1:, :] ** 2 - 1.5 * u2)
    if viscous_stress is not None:
        feq_rest = feq_rest + viscous_stress_term(rho, viscous_stress, lattice)[..., 1:, :]
    if reference_pressure is not None:
        feq_rest = feq_rest + lattice.w[..., 1:, :] * reference_pressure / _CS2
    f0 = rho - jnp.sum(feq_rest, axis=-2, keepdims=True)  # (nx, ny, nz, 1, 1)

    return jnp.concatenate([f0, feq_rest], axis=-2)  # (nx, ny, nz, q, 1)
