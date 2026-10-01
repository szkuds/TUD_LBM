r"""Improved well-balanced equilibrium of Zhang, Guo & Wang (Phys. Fluids 34, 012110, 2022).

Registered as ``"wb_improved"``. Its model inputs (``viscosity``, ``reference_pressure``,
``gradient``) are keyword-only parameters the ``EquilibriumOperator`` protocol lacks,
so ``build_setup`` binds them with ``functools.partial``.

The equilibrium is the ``wb`` one (:mod:`._equilibrium_well_balanced`, called, never
edited) plus two second-moment terms of Eq. 25 on the moving populations:

* ``w_i rho (A S):(c_i c_i - cs2 I) / (2 cs2)`` — adds ``cs2 rho A S`` to the second
  moment, with ``S = grad u + grad u^T``. The shear moments relax at ``1/lambda_v``
  while the kinematic viscosity is

  .. math::

      \nu = c_s^2 \left(\lambda_v - \tfrac{1}{2} - A\right),

  so ``lambda_v`` becomes a stability knob and ``A(rho)`` sets the viscosity per node,
  from the linear interpolation of ``nu`` between the phases (Eqs. 7 and 29).
* ``w_i p_g / cs2`` — adds ``p_g I``, the hydrostatic weight of the reference density
  of a ``gravity_referenced_force``.

The rest population loses the sum of both, so ``sum feq = rho`` still holds. The
PDF's explicit ``f_0`` carries ``+ w_0 rho A div u``, which would violate that; mass
conservation gives the minus sign, and it matches the paper's Appendix A3 moments.
The strain uses the isotropic central stencil (Eq. 30) through the run's gradient
operator, never the non-equilibrium-moment estimate (Eqs. 31-33), which the paper
reports unstable for large ``A``.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.operators.equilibrium._equilibrium_well_balanced import compute_equilibrium as compute_equilibrium_wb
from src.registry import equilibrium_operator

if TYPE_CHECKING:
    from src.config.viscosity_params import ViscosityParams
    from src.lattice.lattice import Lattice
    from src.operators.protocols import DifferentialOperator

_CS2 = 1.0 / 3.0


def _strain_rate(u: jnp.ndarray, gradient: DifferentialOperator) -> jnp.ndarray:
    """``S = grad u + (grad u)^T``, shape ``(nx, ny, nz, 1, d, d)``, through *gradient* per component."""
    # grad_u[..., a, b] = d u_a / d x_b
    grad_u = jnp.stack([gradient(u[..., a : a + 1]) for a in range(u.shape[-1])], axis=-2)
    return grad_u + jnp.swapaxes(grad_u, -1, -2)


def _viscous_stress(
    rho: jnp.ndarray,
    u: jnp.ndarray,
    gradient: DifferentialOperator,
    params: ViscosityParams,
) -> jnp.ndarray:
    """``A(rho) * S``: ``nu`` linear in ``rho`` between the phases (Eq. 7), ``A = lambda_v - 1/2 - nu/cs2``."""
    fraction = (rho - params.rho_v) / (params.rho_l - params.rho_v)
    nu = params.nu_v + fraction * (params.nu_l - params.nu_v)
    a = params.lambda_v - 0.5 - nu / _CS2  # (nx, ny, nz, 1, 1)
    return a[..., None] * _strain_rate(u, gradient)


def _viscous_stress_term(rho: jnp.ndarray, stress: jnp.ndarray, lattice: Lattice) -> jnp.ndarray:
    """``w_i rho (A S):(c_i c_i - cs2 I) / (2 cs2)``: second moment ``cs2 rho A S``, first moment 0."""
    c = lattice.c  # (1, 1, 1, q, d)
    stress_cc = jnp.sum(c[..., :, None] * c[..., None, :] * stress, axis=(-2, -1))[..., None]
    trace = jnp.trace(stress, axis1=-2, axis2=-1)[..., None]  # (nx, ny, nz, 1, 1)
    return lattice.w * rho * (stress_cc - _CS2 * trace) / (2.0 * _CS2)


@equilibrium_operator(name="wb_improved")
def compute_equilibrium(
    rho: jnp.ndarray,
    u: jnp.ndarray,
    lattice: Lattice,
    *,
    viscosity: ViscosityParams | None,
    reference_pressure: jnp.ndarray | None,
    gradient: DifferentialOperator,
) -> jnp.ndarray:
    """Compute the improved well-balanced equilibrium.

    Args:
        rho: Density field, shape ``(nx, ny, nz, 1, 1)``.
        u: Velocity field, shape ``(nx, ny, nz, 1, 2)``.
        lattice: :class:`~src.lattice.lattice.Lattice`.
        viscosity: Viscous-stress parameters, or ``None`` for no ``A S`` term.
        reference_pressure: ``p_g``, shape ``(nx, ny, nz, 1, 1)``, or ``None``.
        gradient: The run's standard gradient operator, for the strain.

    Returns:
        Equilibrium populations ``feq``, shape ``(nx, ny, nz, q, 1)``.
    """
    feq = compute_equilibrium_wb(rho, u, lattice)
    extra = jnp.zeros_like(feq)
    if viscosity is not None:
        extra = extra + _viscous_stress_term(rho, _viscous_stress(rho, u, gradient, viscosity), lattice)
    if reference_pressure is not None:
        extra = extra + lattice.w * reference_pressure / _CS2
    moving = extra[..., 1:, :]
    return feq + jnp.concatenate([-jnp.sum(moving, axis=-2, keepdims=True), moving], axis=-2)
