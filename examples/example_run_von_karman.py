"""Von Karman vortex street example_for_test using the functional API.

Single-phase channel flow past a circular cylinder obstacle, with a
velocity-inlet on the left and an outlet on the right. Streaming I/O mode —
snapshots are saved to disk at each save_interval via jax.debug.callback,
then plotted post-run.

Configuration is loaded from config_von_karman.toml.
"""

from pathlib import Path
from tud_lbm.config.adapter_toml import TomlAdapter
from tud_lbm.config.jax_config import configure_jax
from tud_lbm.io.plotting.figure_builder import FigureBuilder
from tud_lbm.io.save import SimulationIO
from tud_lbm.pipeline.runner import init_state
from tud_lbm.pipeline.runner import run
from tud_lbm.pipeline.setup import build_setup

# Configure JAX (64-bit precision, JIT enabled).
configure_jax()


def run_von_karman():
    """Run the Von Karman vortex street simulation and plot the result."""
    # Load configuration from TOML file.
    config_path = Path(__file__).parent / "config_von_karman.toml"
    adapter = TomlAdapter()
    config = adapter.load(str(config_path))

    setup = build_setup(config)
    state = init_state(setup)

    # Create the I/O handler — this makes the timestamped run directory.
    io = SimulationIO(
        base_dir=config.results_dir,
        config=config,
        simulation_name=config.simulation_name,
        output_format=config.output_format,
    )

    # Stream snapshots to disk while the lax.scan loop runs.
    final_state, _ = run(
        setup,
        state,
        nt=config.nt,
        save_interval=config.save_interval,
        io_handler=io,
        save_fields=tuple(config.save_fields) if config.save_fields else None,
    )

    # Generate one PNG per saved snapshot.
    builder = FigureBuilder(config=config, run_dir=io.run_dir)
    builder.build_all()

    return final_state


if __name__ == "__main__":
    run_von_karman()
