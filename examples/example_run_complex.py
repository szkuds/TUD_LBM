"""Multiphase LBM wetting/hysteresis simulation example_for_test.

Uses the streaming I/O path to write snapshots to disk during the
``jax.lax.scan`` loop via ``jax.debug.callback``, then plots them
post-run using the registered plot operators.

Configuration is loaded from config_complex.toml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config.adapter_toml import TomlAdapter
from config.jax_config import configure_jax
from runner.run import init_state
from runner.run import run
from setup.simulation_setup import build_setup
from util.io import SimulationIO
from util.plotting import FigureBuilder

# Configure JAX (64-bit precision, JIT enabled).
configure_jax()


def wetting_hysteresis_simulation():
    """Run a multiphase wetting simulation with streaming I/O + plotting."""
    # Load configuration from TOML file.
    config_path = Path(__file__).parent / "config_complex.toml"
    adapter = TomlAdapter()
    config = adapter.load(str(config_path))

    simulation_setup = build_setup(config)
    state = init_state(simulation_setup)

    # Create the I/O handler — makes the timestamped run directory.
    io = SimulationIO(
        base_dir=config.results_dir,
        config=config,
        simulation_name=config.simulation_name,
    )

    # Stream snapshots to disk during the lax.scan loop.
    final_state, _ = run(
        simulation_setup,
        state,
        nt=config.nt,
        save_interval=config.save_interval,
        io_handler=io,
        save_fields=tuple(config.save_fields) if config.save_fields else None,
    )

    # Render one composite PNG per saved snapshot.
    builder = FigureBuilder(config=config, run_dir=io.run_dir)
    builder.build_all()

    return final_state


if __name__ == "__main__":
    wetting_hysteresis_simulation()
