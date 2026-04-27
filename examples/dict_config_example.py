#!/usr/bin/env python3
"""Example: Use dict adapter for programmatic configuration (Jupyter-friendly).

This example shows how to create simulations programmatically using Python
dictionaries. This is useful for:
  - Jupyter notebooks (interactive exploration)
  - Parameter sweeps (loop over dict configurations)
  - Unit tests (create configs on the fly)

The dict adapter converts Python dictionaries to SimulationConfig objects
with full validation.

Usage in Jupyter:
    from tud_lbm.readers import dict as dict_reader
    from tud_lbm import build_setup, run
    from tud_lbm.pipeline.runner import init_state

    # Create config as dict
    config_dict = {
        "grid_shape": (128, 128),
        "tau": 0.8,
        "nt": 2000
    }

    # Convert to SimulationConfig
    config = dict_reader.load_simulation_config(config_dict)

    # Run simulation
    setup = build_setup(config)
    state = init_state(setup)
    final_state, _ = run(setup, state, nt=config.nt)

Usage as script:
    python examples/dict_config_example.py
"""

from tud_lbm import build_setup, run
from tud_lbm.pipeline.runner import init_state
from tud_lbm.readers import dict as dict_reader


def example_basic():
    """Most basic example: minimal config dict."""
    print("\n" + "=" * 70)
    print("Example 1: Minimal Configuration")
    print("=" * 70)

    # Minimal config - uses all defaults for unspecified parameters
    config_dict = {
        "grid_shape": (64, 64),
        "tau": 0.8,
        "nt": 100,  # short run for quick demo
    }

    config = dict_reader.load_simulation_config(config_dict)
    print(
        f"Config created from dict: {config.grid_shape}, tau={config.tau}, "
        f"nt={config.nt}"
    )

    setup = build_setup(config)
    state = init_state(setup)
    final_state, _ = run(setup, state, nt=config.nt)
    print("✓ Simulation complete!")


def example_with_forces():
    """Example with body force (gravity-like)."""
    print("\n" + "=" * 70)
    print("Example 2: Simulation with Gravity Force")
    print("=" * 70)

    config_dict = {
        "grid_shape": (64, 32),
        "tau": 0.8,
        "nt": 200,
        "gravity_force": {
            "force_g": 1e-6,  # Small force for stability
            "inclination_angle_deg": 0.0,  # Horizontal force
        },
        "bc_config": {
            "left": "periodic",
            "right": "periodic",
            "top": "bounce-back",  # Wall
            "bottom": "bounce-back",  # Wall
        },
    }

    config = dict_reader.load_simulation_config(config_dict)
    print(f"Config with gravity force: {config.gravity_force}")

    setup = build_setup(config)
    state = init_state(setup)
    final_state, _ = run(setup, state, nt=config.nt)
    print("✓ Simulation with forces complete!")


def example_parameter_sweep():
    """Example: Simple parameter sweep."""
    print("\n" + "=" * 70)
    print("Example 3: Parameter Sweep")
    print("=" * 70)

    tau_values = [0.6, 0.8, 1.0]
    grid_size = 32

    for tau in tau_values:
        config_dict = {
            "grid_shape": (grid_size, grid_size),
            "tau": tau,
            "nt": 50,  # short runs for demo
        }

        config = dict_reader.load_simulation_config(config_dict)
        setup = build_setup(config)
        state = init_state(setup)
        final_state, _ = run(setup, state, nt=config.nt)

        # Extract some diagnostic
        rho_variance = (final_state.rho - final_state.rho.mean()).std()
        print(f"  tau={tau}: rho variance = {rho_variance:.6f}")

    print("✓ Parameter sweep complete!")


def main():
    """Run all examples."""
    print("=" * 70)
    print("TUD-LBM: Dictionary-Based Configuration Examples")
    print("=" * 70)
    print("\nThese examples show how to create simulations programmatically")
    print("using Python dictionaries — ideal for Jupyter notebooks and")
    print("parameter sweeps.")

    example_basic()
    example_with_forces()
    example_parameter_sweep()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nYou can now:")
    print("  1. Use dict_reader.load_simulation_config(dict) in Jupyter")
    print("  2. Create parameter sweeps by looping over dicts")
    print("  3. Programmatically explore different physical parameters")
    print("  4. Save results using write_vtk() or write_numpy()")
    print("\nFor more info, see:")
    print("  - examples/minimal_example.py (basic usage)")
    print("  - examples/from_toml_example.py (TOML config files)")
    print("  - docs/architecture.rst (hexagonal architecture pattern)")


if __name__ == "__main__":
    main()
