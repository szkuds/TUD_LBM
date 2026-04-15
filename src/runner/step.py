"""Jitted core step functions for TUD-LBM.

Provides :func:`step_single_phase` and :func:`step_multiphase` — pure
functions of the form ``(setup, state) → state`` that encapsulate one
complete LBM time step.

These functions replace the mutable ``Update.__call__`` /
``run_timestep`` chain with a functional interface suitable for
``jax.lax.scan``.

Design
~~~~~~
* ``setup`` (:class:`~setup.simulation_setup.SimulationSetup`) is
  **closed over** when building the scan body — JAX treats it as a
  compile-time constant.
* ``state`` (:class:`~state.state.State`) is the scan carry — a pure
  pytree of JAX arrays.
* All operators are prebuilt in :class:`~setup.simulation_setup.SimulationSetup`
  at setup time, eliminating trace-time factory calls.

Usage::

    from runner.run import init_state, run

    state = init_state(setup)
    final_state, trajectory = run(setup, state)
"""

from __future__ import annotations
import jax.numpy as jnp
from operators.force import compute_total_force_ext
from operators.wetting import build_wetting_fn
from registry import update_timestep_operator
from state.state import State

# ── Step functions ───────────────────────────────────────────────────


def _apply_common_step(
    setup,
    state: State,
    rho,
    u,
    force_tot,
    gradient_density=None,
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


def _make_wetting_differential_ops(setup, wetting_state):
    """Build grid-only gradient and laplacian shims from live wetting params.

    Extracts (phi_l, phi_r, d_rho_l, d_rho_r) from wetting_state,
    closes over them together with setup.gradient_density / setup.laplacian_density
    (the wetting-aware factory closures with baked static params), and returns
    two callables each with the signature ``grid -> result`` expected by
    :func:`~operators.macroscopic.compute_macroscopic_multiphase`.

    Args:
        setup: Closed-over :class:`~setup.simulation_setup.SimulationSetup`.
        wetting_state: Current :class:`~state.state.WettingState`.

    Returns:
        ``(gradient_density_shim, laplacian_density_shim)`` — two callable wrappers,
        each ``(grid) → result``.
    """
    # Extract live wetting parameters from the state
    _resolve_wetting_fields = build_wetting_fn("resolve_wetting_fields")
    phi_l, phi_r, d_rho_l, d_rho_r = _resolve_wetting_fields(
        {
            "phi_l": wetting_state.phi_left,
            "phi_r": wetting_state.phi_right,
            "d_rho_l": wetting_state.d_rho_left,
            "d_rho_r": wetting_state.d_rho_right,
        }
    )

    def gradient_density_shim(grid: jnp.ndarray) -> jnp.ndarray:
        """Gradient shim that injects live wetting parameters."""
        return setup.gradient_density(grid, phi_l, phi_r, d_rho_l, d_rho_r)

    def laplacian_density_shim(grid: jnp.ndarray) -> jnp.ndarray:
        """Laplacian shim that injects live wetting parameters."""
        return setup.laplacian_density(grid, phi_l, phi_r, d_rho_l, d_rho_r)

    return gradient_density_shim, laplacian_density_shim


# ── Step functions ───────────────────────────────────────────────────


@update_timestep_operator(name="single_phase")
def step_single_phase(setup, state: State) -> State:
    """Single-phase LBM step using prebuilt operator closures from setup.

    Args:
        setup: Closed-over :class:`~setup.simulation_setup.SimulationSetup`.
        state: Current :class:`~state.state.State`.

    Returns:
        Updated :class:`~state.state.State` after one time step.
    """
    lattice = setup.lattice

    # 1. External forces
    force_ext, state = compute_total_force_ext(setup, state, setup.forces, setup.streaming_fn)

    # 2. Macroscopic fields
    if force_ext is not None:
        rho, u, force_tot = setup.macroscopic_fn(state.f, lattice, force=force_ext)
    else:
        rho, u = setup.macroscopic_fn(state.f, lattice)
        force_tot = None

    # 3–6. Equilibrium → collision → streaming → BCs (shared)
    return _apply_common_step(setup, state, rho, u, force_tot)


@update_timestep_operator(name="multiphase")
def step_multiphase(setup, state: State) -> State:
    """Multiphase LBM step for both wetting and non-wetting simulations.

    When wetting is not active (state.wetting is None), uses the prebuilt
    gradient_density and laplacian_density closures from setup.

    When wetting is active (state.wetting is not None), builds wetting shims
    that inject live wetting parameters into the density operators at each step.

    Args:
        setup: Closed-over :class:`~setup.simulation_setup.SimulationSetup`.
        state: Current :class:`~state.state.State`.

    Returns:
        Updated :class:`~state.state.State` after one time step.
    """
    lattice = setup.lattice
    mp = setup.multiphase_params

    # 1. External forces
    force_ext, state = compute_total_force_ext(setup, state, setup.forces, setup.streaming_fn)

    # 1.1. Resolve density operators (wetting shims if applicable)
    # Wetting shims should only be applied if BOTH the state has wetting AND setup was built with wetting config
    if state.wetting is not None and setup.config.wetting_config is not None:
        gradient_density, laplacian_density = _make_wetting_differential_ops(setup, state.wetting)
    else:
        gradient_density = setup.gradient_density
        laplacian_density = setup.laplacian_density

    # 2. Multiphase macroscopic (chemical potential force, wetting-corrected operators)
    rho, u, force_tot = setup.macroscopic_fn(
        state.f,
        lattice,
        mp,
        force_ext,
        gradient_standard=setup.gradient_standard,
        laplacian_density=laplacian_density,
    )

    # 3–6. Shared pipeline (equilibrium → collision (+source with grad_density) → streaming → BCs)
    new_state = _apply_common_step(setup, state, rho, u, force_tot, gradient_density=gradient_density)

    # 7. Hysteresis update (if applicable)
    _update_wetting_state = build_wetting_fn("hysteresis")
    new_wetting = new_state.wetting
    if (
        new_state.wetting is not None
        and setup.config.wetting_config is not None
        and setup.config.hysteresis_config is not None
    ):
        new_wetting = _update_wetting_state(
            new_state.wetting,
            rho,
            setup,
            new_state.f,
            force_tot,
            gradient_density=gradient_density,
            laplacian_density=laplacian_density,
        )

    return new_state._replace(
        force=force_tot,
        wetting=new_wetting,
    )
