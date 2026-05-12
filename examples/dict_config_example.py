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

from tud_lbm import build_setup
from tud_lbm import run
from tud_lbm.pipeline.runner import init_state
from tud_lbm.readers import dict as dict_reader


def example_basic():
    """Most basic example: minimal config dict."""
    # Minimal config - uses all defaults for unspecified parameters
    config_dict = {
        "grid_shape": (64, 64),
        "tau": 0.8,
        "nt": 100,  # short run for quick demo
    }

    config = dict_reader.load_simulation_config(config_dict)

    setup = build_setup(config)
    state = init_state(setup)
    _final_state, _ = run(setup, state, nt=config.nt)


def example_with_forces():
    """Example with body force (gravity-like)."""
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

    setup = build_setup(config)
    state = init_state(setup)
    _final_state, _ = run(setup, state, nt=config.nt)


def example_parameter_sweep():
    """Example: Simple parameter sweep."""
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
        (final_state.rho - final_state.rho.mean()).std()


def main():
    """Run all examples."""
    example_basic()
    example_with_forces()
    example_parameter_sweep()


if __name__ == "__main__":
    main()
