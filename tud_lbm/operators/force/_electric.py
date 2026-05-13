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
from typing import TYPE_CHECKING
from typing import NamedTuple
import jax.numpy as jnp
from tud_lbm.registry import force_model

if TYPE_CHECKING:
    from collections.abc import Callable
    from tud_lbm import Lattice
    from tud_lbm.config import SimulationConfig
    from tud_lbm.pipeline.state import State

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
        potential: Macroscopic potential, shape ``(nx, ny, nz, 1, 1)``.
        w: Lattice weights, shape ``(1, 1, 1, q, 1)``.

    Returns:
        Equilibrium ``hi``, shape ``(nx, ny, nz, q, 1)``.
    """
    return w * potential


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
        _grid_shape: tuple[int, ...],
        config: SimulationConfig,
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
        from tud_lbm.operators.differential import build_diff_ops

        gradient_standard, _, _, _, _ = build_diff_ops(config, mp_params=None, lattice=lattice)
        return ElectricParams(**params, gradient_standard=gradient_standard)

    @staticmethod
    def compute(
        state: State,
        precomputed: ElectricParams,
        **_kwargs: dict,
    ) -> jnp.ndarray:
        """Compute electric force (step-time, jittable).

        Reads ``state.f`` for density and ``state.h`` for the current
        electric potential distribution.  Uses the pre-built gradient
        closure stored in ``precomputed.gradient_standard``.

        Args:
            state: Current simulation :class:`State`.
            precomputed: :class:`ElectricParams` from :meth:`build`.
            **kwargs: Additional arguments (ignored).

        Returns:
            Electric force field, shape ``(nx, ny, nz, 1, d)``.
        """
        grad = precomputed.gradient_standard

        # Sum over q-axis (axis 3) to get density; extract z-slice to 2D
        rho_3d = jnp.sum(state.f, axis=3, keepdims=True)  # (nx, ny, nz, 1, 1)
        rho_2d = rho_3d[:, :, 0, 0, 0]  # (nx, ny) - squeeze all singleton dims
        epsilon_2d = _rho_to_phi(
            rho_2d,
            precomputed.permittivity_liquid,
            precomputed.permittivity_vapour,
        )

        # Sum over q-axis for potential; extract z-slice to 2D
        potential_3d = jnp.sum(state.h, axis=3, keepdims=True)  # (nx, ny, nz, 1, 1)
        potential_3d_for_grad = potential_3d[:, :, :1, :, :]  # (nx, ny, 1, 1, 1) - take z slice for gradient

        # Gradient of potential → electric field
        grad_potential = grad(potential_3d_for_grad)  # input: (nx, ny, 1, 1, 1) → output: (nx, ny, 1, 1, 2)
        du_dx = grad_potential[:, :, 0, 0, 0]
        du_dy = grad_potential[:, :, 0, 0, 1]
        ex = -du_dx
        ey = -du_dy

        # Gradient of permittivity - broadcast epsilon to (nx, ny, 1, 1, 1)
        epsilon_3d = epsilon_2d[:, :, None, None, None]  # (nx, ny, 1, 1, 1)
        grad_eps = grad(epsilon_3d)  # input: (nx, ny, 1, 1, 1) → output: (nx, ny, 1, 1, 2)
        deps_dx = grad_eps[:, :, 0, 0, 0]
        deps_dy = grad_eps[:, :, 0, 0, 1]

        # Divergence of (epsilon * E):  d(eps*ex)/dx + d(eps*ey)/dy
        eps_ex = epsilon_2d * ex
        eps_ey = epsilon_2d * ey

        # Broadcast for gradient
        eps_ex_3d = eps_ex[:, :, None, None, None]  # (nx, ny, 1, 1, 1)
        eps_ey_3d = eps_ey[:, :, None, None, None]  # (nx, ny, 1, 1, 1)

        grad_eps_ex = grad(eps_ex_3d)  # input: (nx, ny, 1, 1, 1) → output: (nx, ny, 1, 1, 2)
        grad_eps_ey = grad(eps_ey_3d)  # input: (nx, ny, 1, 1, 1) → output: (nx, ny, 1, 1, 2)
        d_eps_ex_dx = grad_eps_ex[:, :, 0, 0, 0]
        d_eps_ey_dy = grad_eps_ey[:, :, 0, 0, 1]
        rho_e = -(d_eps_ex_dx + d_eps_ey_dy)

        e2 = ex * ex + ey * ey
        fx = rho_e * ex - 0.5 * e2 * deps_dx
        fy = rho_e * ey - 0.5 * e2 * deps_dy

        # Reshape to (nx, ny, nz, 1, 2) for 3D compatibility
        # fx, fy have shape (nx, ny), add 3 dims to get (nx, ny, 1, 1, 1)
        fx_expanded = fx[:, :, None, None, None]
        fy_expanded = fy[:, :, None, None, None]
        return jnp.concatenate([fx_expanded, fy_expanded], axis=-1)
