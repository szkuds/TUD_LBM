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
from tud_lbm import build_setup
from tud_lbm import run
from tud_lbm.pipeline.runner import init_state
from tud_lbm.readers import TomlAdapter


def main(config_path: str):
    """Load config from TOML and run simulation."""
    config_file = Path(config_path)

    if not config_file.exists():
        return

    # Load configuration from TOML file using the TOML adapter
    config = TomlAdapter().load(str(config_file))

    # Build simulation setup
    setup = build_setup(config)

    # Initialize state
    state = init_state(setup)

    # Run simulation
    _final_state, _trajectory = run(setup, state, nt=config.nt)

    # Optionally save results
    # from tud_lbm.io.output_data import write_vtk, write_numpy
    # write_numpy(final_state, f"{config.results_dir}/final_state.npz")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("examples/config_simple.toml")
