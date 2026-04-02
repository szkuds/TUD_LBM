"""Electric (leaky-dielectric) force module.

Implements the leaky-dielectric model for electric-field-induced
forces in multiphase flows.  The electric potential is solved via
a secondary lattice Boltzmann sub-step (distribution ``hi``).

The registry-backed :class:`ElectricForceModule` exposes setup-time
``build`` and step-time ``compute`` / ``init_state`` / ``update_state``
methods directly.

Usage::

    # Via registry (preferred)
    from operators.force import build_force_fn

    module = build_force_fn("electric_force")
    params = module.build(config_dict, (64, 64))
    force = module.compute(state, params, diff_ops=diff_ops)
    extra = module.init_state((64, 64), lattice, params)
    state = module.update_state(state, params, lattice, stream_fn)

    # Direct (internal / testing)
    from operators.force._electric import ElectricForceModule

    params = ElectricForceModule.build(config_dict, (64, 64))
    force = ElectricForceModule.compute(state, params, diff_ops=diff_ops)
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
    ) -> ElectricParams:
        """Build electric parameters (setup-time, non-jitted).

        Args:
            params: Config dict from ``[electric_force]`` TOML section.
                Required keys: ``permittivity_liquid``, ``permittivity_vapour``,
                ``conductivity_liquid``, ``conductivity_vapour``.
                Optional keys: ``applied_voltage``, ``voltage_top``,
                ``voltage_bottom``.
            grid_shape: Spatial dimensions (unused, but required by protocol).
            lattice: Simulation lattice (unused, but required by protocol).

        Returns:
            :class:`ElectricParams` NamedTuple (valid JAX pytree).
        """
        return ElectricParams(**params)

    @staticmethod
    def compute(
        state,
        precomputed: ElectricParams,
        *,
        diff_ops=None,
    ) -> jnp.ndarray:
        """Compute electric force (step-time, jittable).

        Reads ``state.f`` for density and ``state.h`` for the current
        electric potential distribution.  Uses the pre-built differential
        operators from the differential package for spatial gradients.

        Args:
            state: Current simulation :class:`State`.
            precomputed: :class:`ElectricParams` from :meth:`build`.
            diff_ops: Pre-built
                :class:`~operators.differential.operators.DifferentialOperators`.
                ``diff_ops.grad_standard`` is used for all gradient
                computations.

        Returns:
            Electric force field, shape ``(nx, ny, 1, 2)``.
        """
        rho_2d = jnp.sum(state.f, axis=2)[:, :, 0]
        epsilon_2d = _rho_to_phi(
            rho_2d,
            precomputed.permittivity_liquid,
            precomputed.permittivity_vapour,
        )

        potential_2d = jnp.sum(state.h, axis=2)[:, :, 0]

        # Gradient of potential → electric field  (via differential package)
        grad_potential = diff_ops.grad_standard(potential_2d[:, :, None, None])
        du_dx = grad_potential[:, :, 0, 0]
        du_dy = grad_potential[:, :, 0, 1]
        ex = -du_dx
        ey = -du_dy

        # Gradient of permittivity
        grad_eps = diff_ops.grad_standard(epsilon_2d[:, :, None, None])
        deps_dx = grad_eps[:, :, 0, 0]
        deps_dy = grad_eps[:, :, 0, 1]

        # Divergence of (epsilon * E):  d(eps*ex)/dx + d(eps*ey)/dy
        eps_ex = epsilon_2d * ex
        eps_ey = epsilon_2d * ey
        grad_eps_ex = diff_ops.grad_standard(eps_ex[:, :, None, None])
        grad_eps_ey = diff_ops.grad_standard(eps_ey[:, :, None, None])
        d_eps_ex_dx = grad_eps_ex[:, :, 0, 0]
        d_eps_ey_dy = grad_eps_ey[:, :, 0, 1]
        rho_e = -(d_eps_ex_dx + d_eps_ey_dy)

        e2 = ex * ex + ey * ey
        fx = rho_e * ex - 0.5 * e2 * deps_dx
        fy = rho_e * ey - 0.5 * e2 * deps_dy
        return jnp.concatenate([fx[:, :, None, None], fy[:, :, None, None]], axis=-1)

    @staticmethod
    def init_state(
        grid_shape: tuple[int, ...],
        lattice: Lattice,
        precomputed: ElectricParams,
    ) -> dict[str, jnp.ndarray]:
        """Create initial electric potential distribution ``h``.

        Sets up a linear voltage profile from bottom to top using the
        voltages stored in *precomputed*.

        Args:
            grid_shape: Spatial dimensions ``(nx, ny, ...)``.
            lattice: Simulation lattice (weights for equilibrium ``h``).
            precomputed: :class:`ElectricParams` from :meth:`build`.

        Returns:
            ``{"h": array}`` — initial potential distribution,
            shape ``(nx, ny, q, 1)``.  Merged into the initial
            :class:`State` by ``init_state()``.
        """
        nx, ny = grid_shape[:2]
        y_vals = jnp.linspace(precomputed.voltage_bottom, precomputed.voltage_top, ny)
        potential = jnp.broadcast_to(y_vals[None, :], (nx, ny))[:, :, None, None]
        return {
            "h": _equilibrium_h(potential, lattice.w),
        }

    @staticmethod
    def update_state(
        state,
        precomputed: ElectricParams,
        lattice: Lattice,
        stream_fn: Callable,
    ):
        """Evolve electric potential distribution by one LBM sub-step.

        Called every time step after :meth:`compute`, inside
        ``_compute_total_force_ext``.

        Args:
            state: Current simulation :class:`State` (reads ``state.f``
                and ``state.h``).
            precomputed: :class:`ElectricParams` from :meth:`build`.
            lattice: Simulation lattice.
            stream_fn: Streaming function ``stream(f, lattice) -> f``.

        Returns:
            Updated :class:`State` with ``h`` replaced.
        """
        rho_2d = jnp.sum(state.f, axis=2)[:, :, 0]
        conductivity = _rho_to_phi(
            rho_2d,
            precomputed.conductivity_liquid,
            precomputed.conductivity_vapour,
        )

        potential = jnp.sum(state.h, axis=2, keepdims=True)
        h_eq = _equilibrium_h(potential, lattice.w)

        tau_e = 3.0 * conductivity[:, :, None, None] + 0.5
        omega_e = 1.0 / tau_e
        h_col = (1.0 - omega_e) * state.h + omega_e * h_eq

        top_potential = jnp.full((state.h.shape[0], 1, 1, 1), precomputed.voltage_top)
        h_col = h_col.at[:, -1:, :, :].set(_equilibrium_h(top_potential, lattice.w))
        bottom_potential = jnp.full((state.h.shape[0], 1, 1, 1), precomputed.voltage_bottom)
        h_col = h_col.at[:, :1, :, :].set(_equilibrium_h(bottom_potential, lattice.w))

        return state._replace(h=stream_fn(h_col, lattice))
