#!/usr/bin/env python3
"""Minimal example: Run a single-phase D2Q9 simulation with default parameters.

This example demonstrates the simplest way to run a simulation using TUD-LBM.
All parameters use sensible defaults — you only need to specify grid dimensions
if you want something other than 64×64.

Physics:
    - 2D Lattice Boltzmann Method (D2Q9 lattice)
    - Single-phase fluid
    - Periodic boundary conditions (all sides)
    - Rest equilibrium initialization
    - BGK collision operator

Usage:
    python examples/minimal_example.py
"""

from tud_lbm import SimulationConfig
from tud_lbm import build_setup
from tud_lbm import run
from tud_lbm.pipeline.runner import init_state


def main():
    """Run a minimal simulation."""
    # Create configuration with default parameters
    # All parameters have sensible defaults:
    # - grid_shape: (64, 64)
    # - tau: 1.0 (stable collision time)
    # - nt: 1000 (timesteps)
    # - lattice_type: "D2Q9"
    # - collision_scheme: "bgk"
    # - bc_config: all periodic
    config = SimulationConfig(
        grid_shape=(64, 64),  # 64x64 lattice
        tau=0.8,  # Relaxation time (kinematic viscosity: nu = (0.8-0.5)/3 = 0.1)
        nt=1000,  # Number of timesteps
    )

    # Build simulation setup from configuration
    # This resolves operators from registry and prepares immutable setup
    setup = build_setup(config)

    # Initialize state (rest equilibrium + zeros for velocity/forces)
    state = init_state(setup)

    # Run simulation for nt timesteps
    # run() uses jax.lax.scan for efficient JAX compilation
    # Returns: (final_state, trajectory)
    # - final_state: State after nt steps
    # - trajectory: All intermediate states (sampled by save_interval)
    _final_state, _trajectory = run(setup, state, nt=config.nt)


if __name__ == "__main__":
    main()
