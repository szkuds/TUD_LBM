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
* All operators are resolved from the ``operators/`` package at
  trace time — no legacy class instances are used.

Usage::

    from runner.step import step_single_phase, get_step_fn
    from runner.run import init_state

    step_fn = get_step_fn(setup)
    state = init_state(setup)
    new_state = step_fn(state)
"""

from __future__ import annotations

import jax.numpy as jnp

from state.state import State
from operators.macroscopic.single_phase import compute_macroscopic
from operators.equilibrium.equilibrium import compute_equilibrium
from operators.streaming.streaming import stream
from operators.collision.factory import build_collision_fn
from operators.boundary.composite import build_composite_bc
from operators.force.source_term import source as compute_source
from operators.force.gravity import compute_gravity_force
from operators.force.electric import compute_electric_force, update_hi
from operators.macroscopic.multiphase import compute_macroscopic_multiphase
from operators.wetting.hysteresis import update_wetting_state

# ── Step functions ───────────────────────────────────────────────────


def step_single_phase(setup, state: State) -> State:
    """Single-phase LBM step using pure-function operators.

    Args:
        setup: Closed-over :class:`~setup.simulation_setup.SimulationSetup`.
        state: Current :class:`~state.state.State`.

    Returns:
        Updated :class:`~state.state.State` after one time step.
    """
    lattice = setup.lattice
    collision_fn = build_collision_fn(setup.collision_scheme)
    bc_fn = build_composite_bc(setup.bc_config, lattice)

    # 1. Macroscopic fields
    if setup.force_enabled and state.force_ext is not None:
        # Apply gravity force template if present
        force_ext = state.force_ext
        if setup.gravity_template is not None:
            rho_pre = jnp.sum(state.f, axis=2, keepdims=True)
            grav_force = compute_gravity_force(setup.gravity_template, rho_pre)
            force_ext = force_ext + grav_force

        rho, u, force_tot = compute_macroscopic(state.f, lattice, force=force_ext)
        # 2. Equilibrium
        feq = compute_equilibrium(rho, u, lattice)
        # 3. Source term + collision
        src = compute_source(rho, u, force_tot, lattice, diff_ops=setup.diff_ops)
        f_col = collision_fn(state.f, feq, setup.tau, src)
    else:
        rho, u = compute_macroscopic(state.f, lattice)
        # 2. Equilibrium
        feq = compute_equilibrium(rho, u, lattice)
        # 3. Collision (no source)
        f_col = collision_fn(state.f, feq, setup.tau)

    # 4. Streaming
    f_stream = stream(f_col, lattice)

    # 5. Boundary conditions
    f_bc = bc_fn(f_stream, f_col, setup.bc_masks)

    return state._replace(
        f=f_bc,
        rho=rho,
        u=u,
        t=state.t + 1,
    )


def step_multiphase(setup, state: State) -> State:
    """Multiphase LBM step using pure-function operators.

    All differential-operator branching (wetting / no-wetting) is resolved
    at setup time inside :class:`~operators.differential.operators.DifferentialOperators`.
    This function contains **no** ``if wetting_enabled`` guards.

    Args:
        setup: Closed-over :class:`~setup.simulation_setup.SimulationSetup`.
        state: Current :class:`~state.state.State`.

    Returns:
        Updated :class:`~state.state.State` after one time step.
    """
    lattice = setup.lattice
    mp = setup.multiphase_params
    diff_ops = setup.diff_ops
    collision_fn = build_collision_fn(setup.collision_scheme)
    bc_fn = build_composite_bc(setup.bc_config, lattice)

    # 1. Multiphase macroscopic (includes chemical potential, gradient, Laplacian)
    #    Start with external force from gravity / electric if applicable.
    force_ext = state.force_ext

    # Gravity force contribution
    if setup.gravity_template is not None:
        # We need rho first for gravity — use a quick density computation
        rho_pre = jnp.sum(state.f, axis=2, keepdims=True)
        grav_force = compute_gravity_force(setup.gravity_template, rho_pre)
        if force_ext is not None:
            force_ext = force_ext + grav_force
        else:
            force_ext = grav_force

    # Electric force contribution
    new_h = state.h
    if setup.electric_params is not None and state.h is not None:
        rho_pre = jnp.sum(state.f, axis=2, keepdims=True)
        elec_force = compute_electric_force(
            rho_pre,
            state.h,
            setup.electric_params,
            lattice,
        )
        new_h = update_hi(
            state.h,
            rho_pre,
            setup.electric_params,
            lattice,
            stream,
        )
        if force_ext is not None:
            force_ext = force_ext + elec_force
        else:
            force_ext = elec_force

    rho, u, force_tot = compute_macroscopic_multiphase(
        state.f,
        lattice,
        mp,
        force_ext=force_ext,
        diff_ops=diff_ops,
    )

    # 2. Equilibrium
    feq = compute_equilibrium(rho, u, lattice)

    # 3. Source term + collision
    src = compute_source(rho, u, force_tot, lattice, diff_ops=diff_ops)
    f_col = collision_fn(state.f, feq, setup.tau, src)

    # 4. Streaming
    f_stream = stream(f_col, lattice)

    # 5. Boundary conditions
    f_bc = bc_fn(f_stream, f_col, setup.bc_masks)

    # 6. Wetting / hysteresis state update (if enabled)
    new_wetting = state.wetting
    if state.wetting is not None and setup.hysteresis_config is not None:
        new_wetting = update_wetting_state(
            state.wetting,
            rho,
            setup,
            f_bc,
            force_tot,
        )

    return state._replace(
        f=f_bc,
        rho=rho,
        u=u,
        force=force_tot,
        t=state.t + 1,
        h=new_h,
        wetting=new_wetting,
    )


def get_step_fn(setup):
    """Return the appropriate step function with *setup* closed over.

    Args:
        setup: :class:`~setup.simulation_setup.SimulationSetup`.

    Returns:
        A callable ``step_fn(state) → state``.
    """
    if setup.multiphase_params is not None:

        def _step(state: State) -> State:
            return step_multiphase(setup, state)

    else:

        def _step(state: State) -> State:
            return step_single_phase(setup, state)

    return _step
