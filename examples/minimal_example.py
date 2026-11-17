#!/usr/bin/env python3
"""Minimal example: Run a single-phase D2Q9 simulation with default parameters.

This example demonstrates the simplest way to run a simulation using TUD-LBM.
Every parameter has a default; this script sets only the grid, tau and nt.

Physics:
    - 2D Lattice Boltzmann Method (D2Q9 lattice)
    - Single-phase fluid (no [multiphase] section)
    - Periodic boundary conditions (all sides)
    - Rest equilibrium initialisation
    - BGK collision operator

Usage:
    python examples/minimal_example.py
"""

from src import SimulationConfig
from src import build_setup
from src import init_state
from src import observe
from src import run


def main():
    """Run a minimal simulation."""
    # Create a configuration. Fields not given here keep their defaults:
    # - lattice_type: "D2Q9"
    # - collision_scheme: "bgk"
    # - boundary_conditions: every edge "periodic"
    # - init_type: "standard" (rest equilibrium)
    config = SimulationConfig(
        grid_shape=(64, 64),  # 64x64 lattice, stored as (64, 64, 1)
        tau=0.8,  # Relaxation time (kinematic viscosity: nu = (0.8-0.5)/3 = 0.1)
        nt=1000,  # Number of timesteps
    )

    # Build the simulation setup: the registry resolves every operator once.
    setup = build_setup(config)

    # Initialise the state. It carries only the populations f, the time t
    # and the addon substates ext.
    state = init_state(setup)

    # Run nt timesteps with jax.lax.scan. Returns (final_state, trajectory),
    # where trajectory stacks the state of every step. For long runs use the
    # streaming mode instead (see example_run_simple.py).
    final_state, _trajectory = run(setup, state, nt=config.nt)

    # Density, velocity, forces and pressure are derived from a state.
    fields = observe(setup, final_state)
    print(f"t = {int(final_state.t)}, mean density = {float(fields.rho.mean()):.6f}")


if __name__ == "__main__":
    main()
