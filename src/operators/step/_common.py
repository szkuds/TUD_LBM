"""Shared step helper — equilibrium → collision → streaming → BCs.

Internal module used by ``_single_phase`` and ``_multiphase``.
Not registered in the operator registry; imported directly by siblings.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax.numpy as jnp
    from operators.protocols import DifferentialOperator
    from setup import SimulationSetup
    from state.state import State


def _apply_common_step(
    setup: SimulationSetup,
    state: State,
    rho: jnp.ndarray,
    u: jnp.ndarray,
    force_tot: jnp.ndarray,
    gradient_density: DifferentialOperator = None,
) -> State:
    """Apply equilibrium → collision (+source) → streaming → BCs.

    Called by both step_single_phase and step_multiphase after macroscopic
    fields have been computed. Returns the updated State with f, rho, u replaced.

    Args:
        setup: Closed-over :class:`~setup.simulation_setup.SimulationSetup`.
        state: Current :class:`~state.state.State`.
        rho: Density field, shape ``(nx, ny, 1, 1)``.
        u: Velocity field, shape ``(nx, ny, 1, 2)``.
        force_tot: Total force or None.
        gradient_density: Optional wetting-corrected density gradient closure.
            If None, uses setup.gradient_density.

    Returns:
        Updated :class:`~state.state.State` with f, rho, u, t updated.
    """
    lattice = setup.lattice

    # 3. Equilibrium
    feq = setup.equilibrium_fn(rho, u, lattice)

    # 4. Collision (with or without source term)
    if force_tot is not None and setup.forces is not None:
        # Use provided gradient_density if available (for wetting), else use setup default
        grad = gradient_density if gradient_density is not None else setup.gradient_density
        src = setup.forces.source_term(rho, u, force_tot, lattice, gradient=grad)
        f_col = setup.collision_fn(state.f, feq, setup.tau, src)
    else:
        f_col = setup.collision_fn(state.f, feq, setup.tau)

    # 5. Streaming
    f_stream = setup.streaming_fn(f_col, lattice)

    # 6. Boundary conditions
    f_bc = setup.bc_fn(f_stream, f_col, setup.bc_masks)

    return state._replace(
        f=f_bc,
        rho=rho,
        u=u,
        t=state.t + 1,
    )
