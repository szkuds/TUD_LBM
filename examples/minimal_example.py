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

from tud_lbm import SimulationConfig, State, build_setup, run
from tud_lbm.pipeline.runner import init_state


def main():
    """Run a minimal simulation."""
    print("=" * 70)
    print("TUD-LBM Minimal Example")
    print("=" * 70)

    # Create configuration with default parameters
    # All parameters have sensible defaults:
    # - grid_shape: (64, 64)
    # - tau: 1.0 (stable collision time)
    # - nt: 1000 (timesteps)
    # - lattice_type: "D2Q9"
    # - collision_scheme: "bgk"
    # - bc_config: all periodic
    config = SimulationConfig(
        grid_shape=(64, 64),  # 64×64 lattice
        tau=0.8,  # Relaxation time (kinematic viscosity: nu = (0.8-0.5)/3 = 0.1)
        nt=1000,  # Number of timesteps
    )

    print(f"\nConfiguration:")
    print(f"  Grid shape:        {config.grid_shape}")
    print(f"  Lattice type:      {config.lattice_type}")
    print(f"  Tau (relaxation):  {config.tau}")
    print(f"  Timesteps:         {config.nt}")
    print(f"  Collision scheme:  {config.collision_scheme}")
    print(f"  Boundary conds:    {config.bc_config}")

    # Build simulation setup from configuration
    # This resolves operators from registry and prepares immutable setup
    print("\nBuilding simulation setup...")
    setup = build_setup(config)
    print(
        f"  Setup complete: lattice={setup.lattice.name}, " f"grid={setup.grid_shape}"
    )

    # Initialize state (rest equilibrium + zeros for velocity/forces)
    print("\nInitializing state...")
    state = init_state(setup)
    print(
        f"  State shape: f={state.f.shape}, rho={state.rho.shape}, "
        f"u={state.u.shape}"
    )

    # Run simulation for nt timesteps
    # run() uses jax.lax.scan for efficient JAX compilation
    # Returns: (final_state, trajectory)
    # - final_state: State after nt steps
    # - trajectory: All intermediate states (sampled by save_interval)
    print("\nRunning simulation...")
    final_state, trajectory = run(setup, state, nt=config.nt)
    print(f"  ✓ Simulation complete!")
    print(
        f"  Final state: rho range = [{final_state.rho.min():.4f}, "
        f"{final_state.rho.max():.4f}]"
    )
    print(f"  Velocity magnitude: {(final_state.u**2).sum(axis=-1).mean()**.5:.6f}")

    # To visualize or save results, use output adapters:
    # from tud_lbm.io.output_data import write_vtk, write_numpy
    # write_vtk(final_state, "output.vtk")          # ParaView compatible
    # write_numpy(final_state, "output.npz")        # Post-processing

    print("\n" + "=" * 70)
    print("Example complete! You can now:")
    print("  1. Modify grid_shape, tau, nt above for different parameters")
    print("  2. Add forces: gravity_force={...} or electric_force={...}")
    print("  3. Load from TOML: from tud_lbm.readers import toml")
    print("  4. Save results: from tud_lbm.io.output_data import write_*")
    print("=" * 70)


if __name__ == "__main__":
    main()
