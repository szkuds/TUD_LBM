"""Electric (leaky-dielectric) force — pure functions.

Implements the leaky-dielectric model for electric-field-induced
forces in multiphase flows.  The electric potential is solved via
a secondary lattice Boltzmann sub-step (distribution ``hi``).

Provides:

* **Setup-time** (non-jitted): :func:`build_electric_params`, :func:`init_hi`.
* **Step-time** (jittable): :func:`compute_electric_force`, :func:`update_hi`.
"""

from __future__ import annotations
from collections.abc import Callable
from typing import NamedTuple
import jax.numpy as jnp
from registry import force_model
from setup.lattice import Lattice


class ElectricParams(NamedTuple):
    """Static parameters for the leaky-dielectric electric model.

    All fields are Python scalars or pre-computed JAX arrays.  This
    NamedTuple is a valid JAX pytree and can be stored inside
    :class:`~setup.simulation_setup.SimulationSetup`.

    Attributes:
        permittivity_liquid: Relative permittivity of the liquid phase.
        permittivity_vapour: Relative permittivity of the vapour phase.
        conductivity_liquid: Electrical conductivity of the liquid phase.
        conductivity_vapour: Electrical conductivity of the vapour phase.
        applied_voltage: Applied voltage across the domain (top - bottom).
        voltage_top: Potential at the top boundary.
        voltage_bottom: Potential at the bottom boundary.
    """

    permittivity_liquid: float
    permittivity_vapour: float
    conductivity_liquid: float
    conductivity_vapour: float
    applied_voltage: float = 0.0
    voltage_top: float = 0.0
    voltage_bottom: float = 0.0


# ── Setup-time helpers (non-jitted) ─────────────────────────────────


@force_model(name="electric")
def build_electric_params(
    *,
    permittivity_liquid: float,
    permittivity_vapour: float,
    conductivity_liquid: float,
    conductivity_vapour: float,
    applied_voltage: float = 0.0,
    voltage_top: float = 0.0,
    voltage_bottom: float = 0.0,
) -> ElectricParams:
    """Construct :class:`ElectricParams` from user-supplied values.

    Args:
        permittivity_liquid: Liquid-phase permittivity.
        permittivity_vapour: Vapour-phase permittivity.
        conductivity_liquid: Liquid-phase conductivity.
        conductivity_vapour: Vapour-phase conductivity.
        applied_voltage: Total voltage across domain.
        voltage_top: Potential at top boundary.
        voltage_bottom: Potential at bottom boundary.

    Returns:
        An :class:`ElectricParams` NamedTuple.
    """
    return ElectricParams(
        permittivity_liquid=permittivity_liquid,
        permittivity_vapour=permittivity_vapour,
        conductivity_liquid=conductivity_liquid,
        conductivity_vapour=conductivity_vapour,
        applied_voltage=applied_voltage,
        voltage_top=voltage_top,
        voltage_bottom=voltage_bottom,
    )


def init_hi(
    nx: int,
    ny: int,
    lattice: Lattice,
    *,
    voltage_top: float = 0.0,
    voltage_bottom: float = 0.0,
) -> jnp.ndarray:
    """Initialise the electric-potential distribution ``hi``.

    Sets up a linear voltage profile from bottom to top, then
    computes the corresponding equilibrium distribution.

    Args:
        nx: Grid size in x.
        ny: Grid size in y.
        lattice: :class:`~setup.lattice.Lattice`.
        voltage_top: Potential at the top boundary.
        voltage_bottom: Potential at the bottom boundary.

    Returns:
        Electric potential distribution ``hi``, shape ``(nx, ny, q, 1)``.
    """
    w = lattice.w  # (q,)

    # Linear potential profile from bottom to top
    y_vals = jnp.linspace(voltage_bottom, voltage_top, ny)
    potential_2d = jnp.broadcast_to(y_vals[None, :], (nx, ny))  # (nx, ny)
    potential = potential_2d[:, :, None, None]  # (nx, ny, 1, 1)

    return w[None, None, :, None] * potential  # (nx, ny, q, 1)


# ── Step-time helpers (jittable) ─────────────────────────────────────


def _rho_to_phi(
    rho: jnp.ndarray,
    phi_liquid: float,
    phi_vapour: float,
) -> jnp.ndarray:
    """Map density to a material property (permittivity or conductivity).

    Uses a linear interpolation based on normalised density.

    Args:
        rho: Density field (any shape).
        phi_liquid: Property value at liquid density.
        phi_vapour: Property value at vapour density.

    Returns:
        Interpolated material property, same shape as *rho*.
    """
    rho_max = jnp.max(rho)
    rho_min = jnp.min(rho)
    denom = jnp.maximum(rho_max - rho_min, 1e-16)
    frac = (rho - rho_min) / denom
    return frac * phi_liquid + (1.0 - frac) * phi_vapour


def _equilibrium_h(
    potential: jnp.ndarray,
    w: jnp.ndarray,
) -> jnp.ndarray:
    """Equilibrium distribution for the electric potential.

    ``hi_eq_i = w_i * U``

    Args:
        potential: Macroscopic potential, shape ``(nx, ny, 1, 1)``.
        w: Lattice weights, shape ``(q,)``.

    Returns:
        Equilibrium ``hi``, shape ``(nx, ny, q, 1)``.
    """
    return w[None, None, :, None] * potential


def _gradient_2d(field_2d: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Central-difference gradient on a 2D field (periodic BCs)."""
    df_dx = (jnp.roll(field_2d, -1, axis=0) - jnp.roll(field_2d, 1, axis=0)) / 2.0
    df_dy = (jnp.roll(field_2d, -1, axis=1) - jnp.roll(field_2d, 1, axis=1)) / 2.0
    return df_dx, df_dy


def compute_electric_force(
    rho: jnp.ndarray,
    hi: jnp.ndarray,
    electric_params: ElectricParams,
    lattice: Lattice,
) -> jnp.ndarray:
    """Compute the leaky-dielectric electric body force.

    .. math::

        F = \\rho_e \\mathbf{E}
          - \\tfrac{1}{2} |\\mathbf{E}|^2 \\nabla\\epsilon

    where the free-charge density is
    ``rho_e = -div(epsilon * E)``.

    Args:
        rho: Density field, shape ``(nx, ny, 1, 1)``.
        hi: Electric potential distribution, shape ``(nx, ny, q, 1)``.
        electric_params: :class:`ElectricParams`.
        lattice: :class:`~setup.lattice.Lattice`.

    Returns:
        Electric force field, shape ``(nx, ny, 1, 2)``.
    """
    rho_2d = rho[:, :, 0, 0]

    # Permittivity field
    epsilon_2d = _rho_to_phi(
        rho_2d,
        electric_params.permittivity_liquid,
        electric_params.permittivity_vapour,
    )

    # Macroscopic potential U = sum_i hi
    potential = jnp.sum(hi, axis=2, keepdims=True)  # (nx, ny, 1, 1)
    u_2d = potential[:, :, 0, 0]

    # Electric field E = -grad(U)
    du_dx, du_dy = _gradient_2d(u_2d)
    ex = -du_dx
    ey = -du_dy

    # Gradient of permittivity
    deps_dx, deps_dy = _gradient_2d(epsilon_2d)

    # Divergence of (epsilon * E)
    eps_ex = epsilon_2d * ex
    eps_ey = epsilon_2d * ey
    d_eps_ex_dx = (jnp.roll(eps_ex, -1, axis=0) - jnp.roll(eps_ex, 1, axis=0)) / 2.0
    d_eps_ey_dy = (jnp.roll(eps_ey, -1, axis=1) - jnp.roll(eps_ey, 1, axis=1)) / 2.0
    rho_e = -(d_eps_ex_dx + d_eps_ey_dy)

    # NOTE: Coulombic + dielectric force contributions
    e2 = ex * ex + ey * ey
    fx = rho_e * ex - 0.5 * e2 * deps_dx
    fy = rho_e * ey - 0.5 * e2 * deps_dy

    return jnp.stack(
        [fx[:, :, None, None], fy[:, :, None, None]],
        axis=-1,
    )[
        :,
        :,
        :,
        0,
        :,
    ]  # (nx, ny, 1, 2)


def update_hi(
    hi: jnp.ndarray,
    rho: jnp.ndarray,
    electric_params: ElectricParams,
    lattice: Lattice,
    stream_fn: Callable,
) -> jnp.ndarray:
    """Advance the electric-potential distribution by one LBM sub-step.

    Uses a BGK-like collision with a local relaxation time derived from
    the conductivity field:

    .. math::

        \\tau_e = 3 \\sigma + 0.5

    Boundary conditions: Dirichlet at top/bottom (fixed voltage),
    periodic left/right.

    Args:
        hi: Current potential distribution, shape ``(nx, ny, q, 1)``.
        rho: Density field, shape ``(nx, ny, 1, 1)``.
        electric_params: :class:`ElectricParams`.
        lattice: :class:`~setup.lattice.Lattice`.
        stream_fn: Streaming function ``stream(f, lattice) -> f``.

    Returns:
        Updated potential distribution ``hi``, shape ``(nx, ny, q, 1)``.
    """
    rho_2d = rho[:, :, 0, 0]
    w = lattice.w  # (q,)

    # Conductivity field
    conductivity = _rho_to_phi(
        rho_2d,
        electric_params.conductivity_liquid,
        electric_params.conductivity_vapour,
    )

    # Macroscopic potential
    potential = jnp.sum(hi, axis=2, keepdims=True)  # (nx, ny, 1, 1)

    # Equilibrium for hi
    hi_eq = _equilibrium_h(potential, w)

    # Local relaxation time
    tau_e = 3.0 * conductivity[:, :, None, None] + 0.5
    omega_e = 1.0 / tau_e

    # Collision
    hi_col = (1.0 - omega_e) * hi + omega_e * hi_eq

    # Dirichlet BCs at top and bottom
    pot_top = jnp.full((hi.shape[0], 1, 1, 1), electric_params.voltage_top)
    hi_col = hi_col.at[:, -1:, :, :].set(_equilibrium_h(pot_top, w))
    pot_bot = jnp.full((hi.shape[0], 1, 1, 1), electric_params.voltage_bottom)
    hi_col = hi_col.at[:, :1, :, :].set(_equilibrium_h(pot_bot, w))

    # Streaming
    return stream_fn(hi_col, lattice)
