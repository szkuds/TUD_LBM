"""Example: Load simulation config from TOML file.

This example shows how to use the TOML input adapter to load simulation
parameters from a configuration file. This is useful for:
  - Reproducibility (store parameters in version control)
  - Parameter sweeps (``tud-lbm sweep stage`` writes one TOML per combination)
  - Collaboration (share TOML configs with others)

Usage:
    python examples/from_toml_example.py examples/config_simple.toml

Without an argument, it loads ``config_simple.toml`` next to this script.
For long runs that write snapshots to disk, see ``example_run_simple.py``.
"""

import sys
from pathlib import Path
from src import build_setup
from src import init_state
from src import observe
from src import run
from src.simulation_io.readers import TomlAdapter

DEFAULT_CONFIG = Path(__file__).parent / "config_simple.toml"


def main(config_path: Path) -> None:
    """Load config from TOML and run simulation."""
    if not config_path.is_file():
        sys.exit(f"Config file not found: {config_path}")

    # Load configuration from TOML file using the TOML adapter
    config = TomlAdapter().load(str(config_path))

    # Build simulation setup
    setup = build_setup(config)

    # Initialise state
    state = init_state(setup)

    # Run simulation. The trajectory holds every step in memory.
    final_state, _trajectory = run(setup, state, nt=config.nt)

    fields = observe(setup, final_state)
    print(f"{config.simulation_name}: t = {int(final_state.t)}, mean density = {float(fields.rho.mean()):.6f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG)
