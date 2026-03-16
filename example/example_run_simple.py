"""Single-phase LBM simulation example using the functional API.

Demonstrates the streaming I/O mode — snapshots are saved to disk at
each save_interval via jax.debug.callback, then plotted post-run.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config.jax_config import configure_jax
from config.simulation_config import SimulationConfig
from setup.simulation_setup import build_setup
from runner.run import init_state, run
from util.io import SimulationIO
from util.plotting import FigureBuilder

# Configure JAX (64-bit precision, JIT enabled).
configure_jax()


def run_and_save():
    """Run a simulation, stream snapshots to disk, then plot them."""
    print("\n=== Single-Phase Simulation — Streaming I/O + Plotting ===")

    config = SimulationConfig(
        simulation_name="single_phase_demo",
        grid_shape=(100, 100),
        lattice_type="D2Q9",
        tau=0.6,
        nt=500,
        save_interval=100,
        init_type="standard",
        results_dir="~/TUD_LBM_data/results",
        save_fields=["rho", "u"],
        plot_fields=["density", "velocity"],
    )

    setup = build_setup(config)
    state = init_state(setup)

    print(f"  Grid            : {config.grid_shape}")
    print(f"  Steps           : {config.nt}  (save every {config.save_interval})")
    print(f"  Results dir     : {config.results_dir}")

    # Create the I/O handler — this makes the timestamped run directory.
    io = SimulationIO(
        base_dir=config.results_dir,
        config=config.to_dict(),
        simulation_name=config.simulation_name,
    )
    print(f"  Run directory   : {io.run_dir}")

    # Stream snapshots to disk while the lax.scan loop runs.
    final_state, _ = run(
        setup,
        state,
        nt=config.nt,
        save_interval=config.save_interval,
        io_handler=io,
        save_fields=tuple(config.save_fields),
    )

    print(f"  Final timestep  : {int(final_state.t)}")
    print(f"  Snapshots saved to: {io.data_dir}")

    # Generate one PNG per saved snapshot.
    builder = FigureBuilder(config=config.to_dict(), run_dir=io.run_dir)
    saved_plots = builder.build_all()
    print(f"  Plots saved     : {len(saved_plots)} PNG(s) in {io.run_dir}/plots/")

    return final_state


if __name__ == "__main__":
    print("TUD-LBM  —  Single-Phase Example")
    print("=" * 50)

    run_and_save()

    print("\nDone.")
