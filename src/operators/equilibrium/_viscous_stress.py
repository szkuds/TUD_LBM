r"""Viscous-stress term of the equilibrium — registers nothing.

Zhang, Guo & Wang (Phys. Fluids 34, 012110, 2022) add ``cs2*A*dt*S`` to the
second moment of the equilibrium (Eq. 25), with ``S = grad u + grad u^T``. The
shear moments then relax at ``1/lambda_v`` while the kinematic viscosity is

.. math::

    \nu = c_s^2 \left(\lambda_v - \tfrac{1}{2} - A\right),

so ``lambda_v`` becomes a stability knob and ``A(x)`` sets the viscosity per node.
``viscous_stress`` returns the product ``A*S`` that the ``wb`` equilibrium takes
as its ``viscous_stress`` keyword; ``viscous_stress_term`` is that equilibrium's
contribution. The strain uses the isotropic central stencil (Eq. 30), never the
non-equilibrium-moment estimate (Eqs. 31-33), which the paper reports unstable
for large ``A``.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp

if TYPE_CHECKING:
    from src.config.viscosity_params import ViscosityParams
    from src.lattice.lattice import Lattice
    from src.operators.protocols import DifferentialOperator

_CS2 = 1.0 / 3.0


def strain_rate(u: jnp.ndarray, gradient: DifferentialOperator) -> jnp.ndarray:
    """``S = grad u + (grad u)^T``, shape ``(nx, ny, nz, 1, d, d)``.

    ``gradient`` is applied to each velocity component, so the strain inherits
    its stencil and its boundary pad modes.
    """
    # grad_u[..., a, b] = d u_a / d x_b
    grad_u = jnp.stack([gradient(u[..., a : a + 1]) for a in range(u.shape[-1])], axis=-2)
    return grad_u + jnp.swapaxes(grad_u, -1, -2)


def viscous_stress(
    rho: jnp.ndarray,
    u: jnp.ndarray,
    gradient: DifferentialOperator,
    params: ViscosityParams,
) -> jnp.ndarray:
    """``A(rho) * S``, shape ``(nx, ny, nz, 1, d, d)``.

    ``nu(rho)`` interpolates linearly between ``nu_v`` at ``rho_v`` and ``nu_l``
    at ``rho_l`` (Eq. 7), and ``A = lambda_v - 1/2 - nu/cs2`` (Eq. 29).
    """
    fraction = (rho - params.rho_v) / (params.rho_l - params.rho_v)
    nu = params.nu_v + fraction * (params.nu_l - params.nu_v)
    a = params.lambda_v - 0.5 - nu / _CS2  # (nx, ny, nz, 1, 1)
    return a[..., None] * strain_rate(u, gradient)


def viscous_stress_term(rho: jnp.ndarray, stress: jnp.ndarray, lattice: Lattice) -> jnp.ndarray:
    r"""Equilibrium contribution ``w_i rho (A S):(c_i c_i - cs2 I) / (2 cs2)``, shape ``(nx, ny, nz, q, 1)``.

    Its first moment vanishes and its second is ``cs2 * rho * A * S``; the rest
    population's share follows from mass conservation in the caller.
    """
    c = lattice.c  # (1, 1, 1, q, d)
    stress_cc = jnp.sum(c[..., :, None] * c[..., None, :] * stress, axis=(-2, -1))[..., None]
    trace = jnp.trace(stress, axis1=-2, axis2=-1)[..., None]  # (nx, ny, nz, 1, 1)
    return lattice.w * rho * (stress_cc - _CS2 * trace) / (2.0 * _CS2)
