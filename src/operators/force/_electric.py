"""Electric (leaky-dielectric) force module.

Implements the leaky-dielectric model for electric-field-induced
forces in multiphase flows.  The electric potential is solved via
a secondary lattice Boltzmann sub-step (distribution ``hi``).

The registry-backed :class:`ElectricForceModule` exposes setup-time
``build`` and step-time ``compute`` methods directly.

Usage::

    # Via registry (preferred)
    from operators.force import build_force_fn

    module = build_force_fn("electric_force")
    params = module.build(config_dict, (64, 64), config, lattice)
    force = module.compute(state, params)

    # Direct (internal / testing)
    from operators.force._electric import ElectricForceModule

    params = ElectricForceModule.build(config_dict, (64, 64), config, lattice)
    force = ElectricForceModule.compute(state, params)
"""

from __future__ import annotations
from collections.abc import Callable
from typing import NamedTuple
import jax.numpy as jnp
from registry import force_model
from setup.lattice import Lattice

# ══════════════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════════════


class ElectricParams(NamedTuple):
    """Static parameters for the leaky-dielectric electric model.

    All fields are Python scalars, pre-computed JAX arrays, or callables.
    This NamedTuple is a valid JAX pytree and can be stored inside
    :class:`~setup.simulation_setup.SimulationSetup`.

    Attributes:
        permittivity_liquid: Relative permittivity of the liquid phase.
        permittivity_vapour: Relative permittivity of the vapour phase.
        conductivity_liquid: Electrical conductivity of the liquid phase.
        conductivity_vapour: Electrical conductivity of the vapour phase.
        applied_voltage: Applied voltage across the domain (top - bottom).
        voltage_top: Potential at the top boundary.
        voltage_bottom: Potential at the bottom boundary.
        gradient_standard: Closed-over gradient callable (grid) → (nx, ny, 1, 2),
                          injected at build time from build_diff_ops.
    """

    permittivity_liquid: float
    permittivity_vapour: float
    conductivity_liquid: float
    conductivity_vapour: float
    applied_voltage: float = 0.0
    voltage_top: float = 0.0
    voltage_bottom: float = 0.0
    gradient_standard: Callable | None = None


# ══════════════════════════════════════════════════════════════════════
# Implementation — step-time helpers (jittable)
# ══════════════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════════════
# Registry-backed force module
# ══════════════════════════════════════════════════════════════════════


@force_model(name="electric_force")
class ElectricForceModule:
    """Electric force conforming to :class:`ForceOperator` protocol.

    Stateful — carries auxiliary electric potential distribution ``h``
    that evolves each time step via a secondary LBM sub-step.
    """

    @staticmethod
    def build(
        params: dict,
        grid_shape: tuple[int, ...],
        config,
        lattice: Lattice,
    ) -> ElectricParams:
        """Build electric parameters (setup-time, non-jitted).

        Args:
            params: Config dict from ``[electric_force]`` TOML section.
                Required keys: ``permittivity_liquid``, ``permittivity_vapour``,
                ``conductivity_liquid``, ``conductivity_vapour``.
                Optional keys: ``applied_voltage``, ``voltage_top``,
                ``voltage_bottom``.
            grid_shape: Spatial dimensions (unused, but required by protocol).
            config: Full simulation configuration (for pad-mode resolution).
            lattice: Simulation lattice (weights and velocities for diff ops).

        Returns:
            :class:`ElectricParams` NamedTuple with a closed-over gradient callable.
        """
        from operators.differential import build_diff_ops

        gradient_standard, _, _ = build_diff_ops(config, mp_params=None, lattice=lattice)
        return ElectricParams(**params, gradient_standard=gradient_standard)

    @staticmethod
    def compute(
        state,
        precomputed: ElectricParams,
        **kwargs,
    ) -> jnp.ndarray:
        """Compute electric force (step-time, jittable).

        Reads ``state.f`` for density and ``state.h`` for the current
        electric potential distribution.  Uses the pre-built gradient
        closure stored in ``precomputed.gradient_standard``.

        Args:
            state: Current simulation :class:`State`.
            precomputed: :class:`ElectricParams` from :meth:`build`.

        Returns:
            Electric force field, shape ``(nx, ny, 1, 2)``.
        """
        grad = precomputed.gradient_standard

        rho_2d = jnp.sum(state.f, axis=2)[:, :, 0]
        epsilon_2d = _rho_to_phi(
            rho_2d,
            precomputed.permittivity_liquid,
            precomputed.permittivity_vapour,
        )

        potential_2d = jnp.sum(state.h, axis=2)[:, :, 0]

        # Gradient of potential → electric field
        grad_potential = grad(potential_2d[:, :, None, None])
        du_dx = grad_potential[:, :, 0, 0]
        du_dy = grad_potential[:, :, 0, 1]
        ex = -du_dx
        ey = -du_dy

        # Gradient of permittivity
        grad_eps = grad(epsilon_2d[:, :, None, None])
        deps_dx = grad_eps[:, :, 0, 0]
        deps_dy = grad_eps[:, :, 0, 1]

        # Divergence of (epsilon * E):  d(eps*ex)/dx + d(eps*ey)/dy
        eps_ex = epsilon_2d * ex
        eps_ey = epsilon_2d * ey
        grad_eps_ex = grad(eps_ex[:, :, None, None])
        grad_eps_ey = grad(eps_ey[:, :, None, None])
        d_eps_ex_dx = grad_eps_ex[:, :, 0, 0]
        d_eps_ey_dy = grad_eps_ey[:, :, 0, 1]
        rho_e = -(d_eps_ex_dx + d_eps_ey_dy)

        e2 = ex * ex + ey * ey
        fx = rho_e * ex - 0.5 * e2 * deps_dx
        fy = rho_e * ey - 0.5 * e2 * deps_dy
        return jnp.concatenate([fx[:, :, None, None], fy[:, :, None, None]], axis=-1)
