#!/usr/bin/env python3
"""Example: Load simulation config from TOML file.

This example shows how to use the TOML input adapter to load simulation
parameters from a configuration file. This is useful for:
  - Reproducibility (store parameters in version control)
  - Parameter sweeps (create TOML files for each parameter set)
  - Collaboration (share TOML configs with others)

Usage:
    python examples/from_toml_example.py examples/config_simple.toml
"""

import sys
from pathlib import Path

from tud_lbm import build_setup, run
from tud_lbm.pipeline.runner import init_state
from tud_lbm.readers import toml


def main(config_path: str):
    """Load config from TOML and run simulation."""
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("\nUsage: python examples/from_toml_example.py <path_to_config.toml>")
        print(
            "Example: python examples/from_toml_example.py examples/config_simple.toml"
        )
        return

    print("=" * 70)
    print("TUD-LBM: Load Config from TOML")
    print("=" * 70)

    # Load configuration from TOML file using the TOML adapter
    print(f"\nLoading configuration from: {config_file}")
    config = toml.load_simulation_config(str(config_file))

    print(f"\nConfiguration loaded:")
    print(f"  Simulation name:   {config.simulation_name or '(unnamed)'}")
    print(f"  Simulation type:   {config.sim_type}")
    print(f"  Grid shape:        {config.grid_shape}")
    print(f"  Lattice type:      {config.lattice_type}")
    print(f"  Tau (relaxation):  {config.tau}")
    print(f"  Timesteps:         {config.nt}")
    print(f"  Collision scheme:  {config.collision_scheme}")
    print(f"  Save interval:     {config.save_interval}")
    print(f"  Results directory: {config.results_dir}")

    # Build simulation setup
    print("\nBuilding simulation setup...")
    setup = build_setup(config)

    # Initialize state
    print("Initializing state...")
    state = init_state(setup)

    # Run simulation
    print("Running simulation...")
    final_state, trajectory = run(setup, state, nt=config.nt)

    print(f"✓ Simulation complete!")
    print(
        f"  Final state: rho range = [{final_state.rho.min():.4f}, "
        f"{final_state.rho.max():.4f}]"
    )

    # Optionally save results
    # from tud_lbm.io.output_data import write_vtk, write_numpy
    # write_numpy(final_state, f"{config.results_dir}/final_state.npz")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Modify the TOML file to change parameters")
    print("  2. Save results: uncomment write_numpy/write_vtk above")
    print("  3. Create new TOML configs for different simulations")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("examples/config_simple.toml")
