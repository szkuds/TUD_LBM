"""Shared step helper — equilibrium → collision → streaming → BCs.

Internal module used by ``_single_phase`` and ``_multiphase``.
Not registered in the operator registry; imported directly by siblings.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from src.operators.equilibrium._viscous_stress import viscous_stress

if TYPE_CHECKING:
    from src.operators.protocols import DifferentialOperator
    from src.pipeline.setup import SimulationSetup
    from src.pipeline.state.state import State


def _multiphase_pipeline(
    setup: SimulationSetup,
    f_t: jnp.ndarray,
    force_ext: jnp.ndarray | None,
    gradient_density: DifferentialOperator,
    laplacian_density: DifferentialOperator,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray]:
    """Run one multiphase physics pass — the single canonical implementation.

    This function encapsulates the core multiphase LBM pipeline:
    1. Compute macroscopic fields (rho, u, force) from populations
    2. Apply common step (equilibrium → collision → streaming → BCs)
    3. Recompute macroscopic fields (rho, u, pressure) from updated populations

    This is the single place where multiphase physics runs. All step functions
    and trial steps (hysteresis, etc.) route through here.

    Args:
        setup: Closed-over :class:`~src.pipeline.setup.SimulationSetup`.
        f_t: Current population distribution, shape ``(nx, ny, nz, q, 1)``.
        force_ext: External force field, shape ``(nx, ny, nz, 1, 2)`` or ``None``.
        gradient_density: Density gradient operator ``(grid) -> result``.
            Used in source term and passed to common step.
        laplacian_density: Laplacian of density operator ``(grid) -> result``.
            Used in macroscopic computation.

    Returns:
        ``(f_out, rho, u, force_tot, pressure)`` where:
        - ``f_out``: Post-BC populations, shape ``(nx, ny, nz, q, 1)``
        - ``rho``: Updated density field
        - ``u``: Updated velocity field
        - ``force_tot``: Total interaction force (or None if no forces)
        - ``pressure``: Bulk pressure of the updated density field
    """
    from src.pipeline.state.state import State

    if setup.macroscopic_fn is None:
        msg = "macroscopic_fn is required for multiphase pipeline"
        raise TypeError(msg)
    if setup.multiphase_params is None:
        msg = "multiphase_params is required for multiphase pipeline"
        raise TypeError(msg)
    if setup.gradient_standard is None:
        msg = "gradient_standard is required for multiphase pipeline"
        raise TypeError(msg)

    lattice = setup.lattice

    # 1. Compute macroscopic fields from current populations
    rho, u, force_tot, _ = setup.macroscopic_fn(
        f_t,
        lattice,
        setup.multiphase_params,
        force_ext,
        gradient_standard=setup.gradient_standard,
        laplacian_density=laplacian_density,
    )

    # 2. Create temporary state for common step
    temp_state = State(
        f=f_t,
        rho=rho,
        u=u,
        t=jnp.array(0),
        wetting=None,
    )

    # 3. Apply common step (equilibrium → collision → streaming → BCs)
    next_state = _apply_common_step(setup, temp_state, rho, u, force_tot, gradient_density=gradient_density)

    # 4. Recompute macroscopic fields from updated populations (for next step)
    rho_next, u_next, _, pressure_next = setup.macroscopic_fn(
        next_state.f,
        lattice,
        setup.multiphase_params,
        force_ext=force_ext,
        gradient_standard=setup.gradient_standard,
        laplacian_density=laplacian_density,
    )

    return next_state.f, rho_next, u_next, force_tot, pressure_next


def _viscous_stress(setup: SimulationSetup, rho: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray | None:
    """``A*S`` for the equilibrium, or ``None`` when the viscosity is not decoupled."""
    if setup.viscosity_params is None:
        return None
    if setup.gradient_standard is None:
        msg = "gradient_standard is required for the viscous-stress term"
        raise TypeError(msg)
    return viscous_stress(rho, u, setup.gradient_standard, setup.viscosity_params)


def _apply_common_step(
    setup: SimulationSetup,
    state: State,
    rho: jnp.ndarray,
    u: jnp.ndarray,
    force_tot: jnp.ndarray | None,
    gradient_density: DifferentialOperator | None = None,
) -> State:
    """Apply equilibrium → collision (+source) → streaming → BCs.

    Called by both step_single_phase and step_multiphase after macroscopic
    fields have been computed. Returns the updated State with f, rho, u replaced.

    Args:
        setup: Closed-over :class:`~src.pipeline.setup.SimulationSetup`.
        state: Current :class:`~src.pipeline.state.state.State`.
        rho: Density field, shape ``(nx, ny, 1, 1)``.
        u: Velocity field, shape ``(nx, ny, 1, 2)``.
        force_tot: Total force or None.
        gradient_density: Optional wetting-corrected density gradient closure.
            If None, uses setup.gradient_density.

    Returns:
        Updated :class:`~src.pipeline.state.state.State` with f, rho, u, t updated.
    """
    if setup.equilibrium_fn is None:
        msg = "equilibrium_fn is required"
        raise TypeError(msg)
    if setup.collision_fn is None:
        msg = "collision_fn is required"
        raise TypeError(msg)
    if setup.streaming_fn is None:
        msg = "streaming_fn is required"
        raise TypeError(msg)
    if setup.bc_fn is None:
        msg = "bc_fn is required"
        raise TypeError(msg)

    lattice = setup.lattice

    # 3. Equilibrium (+ the viscous-stress term A*S when the viscosity is decoupled)
    reference = setup.reference_pressure
    feq = setup.equilibrium_fn(
        rho,
        u,
        lattice,
        viscous_stress=_viscous_stress(setup, rho, u),
        reference_pressure=None if reference is None else reference.field,
    )

    # 4. Collision (with or without source term)
    if force_tot is not None:
        if setup.source_fn is None:
            msg = "source_fn is required when force_tot is active"
            raise TypeError(msg)
        # Use provided gradient_density if available (for wetting), else use setup default
        grad = gradient_density if gradient_density is not None else setup.gradient_density
        if grad is None:
            msg = "gradient_density is required when force_tot is active"
            raise TypeError(msg)
        src = setup.source_fn(
            rho,
            u,
            force_tot,
            lattice,
            gradient=grad,
            reference_gradient=None if reference is None else reference.gradient,
        )
        f_col = setup.collision_fn(state.f, feq, setup.tau, src)
    else:
        f_col = setup.collision_fn(state.f, feq, setup.tau)

    # 5. Streaming
    f_stream = setup.streaming_fn(f_col, lattice)

    # 5b. Interior obstacle bounce-back (before edge BCs)
    if setup.obstacle_fn is not None:
        f_stream = setup.obstacle_fn(f_stream, f_col)

    # 6. Boundary conditions
    f_bc = setup.bc_fn(f_stream, f_col)

    return state._replace(
        f=f_bc,
        rho=rho,
        u=u,
        t=state.t + 1,
    )
