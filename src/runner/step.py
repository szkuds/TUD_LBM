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
from typing import Any, cast
from operators.boundary.composite import build_composite_bc
from operators.collision import build_collision_fn
from operators.equilibrium import build_equilibrium_fn
from operators.force.source_term import source as compute_source
from operators.macroscopic import build_macroscopic_fn
from operators.streaming import build_streaming_fn
from operators.wetting.hysteresis import update_wetting_state
from state.state import State

# ── Step functions ───────────────────────────────────────────────────


def _compute_total_force_ext(setup, state: State, streaming_fn):
    """Compute the summed external force contribution and update stateful hooks."""
    total_force = state.force_ext

    if not getattr(setup, "forces", ()):
        return total_force, state

    for spec in setup.forces:
        contribution = spec.compute_fn(state, spec.precomputed, diff_ops=setup.diff_ops)
        total_force = contribution if total_force is None else total_force + contribution
        state = spec.update_state_fn(state, spec.precomputed, setup.lattice, streaming_fn)

    return total_force, state


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
    equilibrium_fn = build_equilibrium_fn("wb")  # Default to weakly-compressible
    streaming_fn = build_streaming_fn("standard")
    macroscopic_fn = build_macroscopic_fn("standard")  # Single-phase
    bc_fn = build_composite_bc(setup.bc_config, lattice)

    # 1. External forces
    force_ext, state = _compute_total_force_ext(setup, state, streaming_fn)

    # 2. Macroscopic fields
    if force_ext is not None:
        rho, u, force_tot = macroscopic_fn(state.f, lattice, force=force_ext)
        # 2. Equilibrium
        feq = equilibrium_fn(rho, u, lattice)
        # 3. Source term + collision
        src = compute_source(rho, u, force_tot, lattice, diff_ops=setup.diff_ops)
        f_col = collision_fn(state.f, feq, setup.tau, src)
    else:
        rho, u = macroscopic_fn(state.f, lattice)
        # 2. Equilibrium
        feq = equilibrium_fn(rho, u, lattice)
        # 3. Collision (no source)
        f_col = collision_fn(state.f, feq, setup.tau)

    # 4. Streaming
    f_stream = streaming_fn(f_col, lattice)

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
    equilibrium_fn = build_equilibrium_fn()
    streaming_fn = build_streaming_fn()
    macroscopic_fn = build_macroscopic_fn(setup.multiphase_params.eos)  # Multiphase uses double-well
    bc_fn = build_composite_bc(setup.bc_config, lattice)

    # 1. Multiphase macroscopic (includes chemical potential, gradient, Laplacian)
    force_ext, state = _compute_total_force_ext(setup, state, streaming_fn)

    rho, u, force_tot = cast(Any, macroscopic_fn)(
        state.f,
        lattice,
        mp,
        force_ext,
        diff_ops=diff_ops,  # type: ignore[call-arg]
    )

    # 2. Equilibrium
    feq = equilibrium_fn(rho, u, lattice)

    # 3. Source term + collision
    src = compute_source(rho, u, force_tot, lattice, diff_ops=diff_ops)
    f_col = collision_fn(state.f, feq, setup.tau, src)

    # 4. Streaming
    f_stream = streaming_fn(f_col, lattice)

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
