"""Multiphase LBM wetting/hysteresis simulation example.

Uses the streaming I/O path to write snapshots to disk during the
``jax.lax.scan`` loop via ``jax.debug.callback``, then plots them
post-run using the registered plot operators.

Configuration is loaded from config_complex.toml.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config.jax_config import configure_jax
from config.adapter_toml import TomlAdapter
from setup.simulation_setup import build_setup
from runner.run import init_state, run
from util.io import SimulationIO
from util.plotting import FigureBuilder

# Configure JAX (64-bit precision, JIT enabled).
configure_jax()


def wetting_hysteresis_simulation():
    """Run a multiphase wetting simulation with streaming I/O + plotting."""
    print("\n=== Multiphase Wetting — Streaming I/O + Plotting ===")

    # Load configuration from TOML file.
    config_path = Path(__file__).parent / "config_complex.toml"
    adapter = TomlAdapter()
    config = adapter.load(str(config_path))

    simulation_setup = build_setup(config)
    state = init_state(simulation_setup)

    print(f"  Config loaded from : {config_path.name}")
    print(f"  Grid               : {config.grid_shape}")
    print(f"  Steps              : {config.nt}  (save every {config.save_interval})")
    print(f"  Gravity present    : {simulation_setup.gravity_template is not None}")

    # Create the I/O handler — makes the timestamped run directory.
    io = SimulationIO(
        base_dir=config.results_dir,
        config=config.to_dict(),
        simulation_name=config.simulation_name,
    )
    print(f"  Run directory      : {io.run_dir}")

    # Stream snapshots to disk during the lax.scan loop.
    final_state, _ = run(
        simulation_setup,
        state,
        nt=config.nt,
        save_interval=config.save_interval,
        io_handler=io,
        save_fields=tuple(config.save_fields) if config.save_fields else None,
    )

    print(f"  Final timestep     : {int(final_state.t)}")
    print(f"  Snapshots saved to : {io.data_dir}")

    # Render one composite PNG per saved snapshot.
    builder = FigureBuilder(config=config.to_dict(), run_dir=io.run_dir)
    saved_plots = builder.build_all()
    print(f"  Plots saved        : {len(saved_plots)} PNG(s) in {io.run_dir}/plots/")

    return final_state


if __name__ == "__main__":
    print("TUD-LBM  —  Multiphase Wetting Example")
    print("=" * 50)

    wetting_hysteresis_simulation()

    print("\nDone.")
