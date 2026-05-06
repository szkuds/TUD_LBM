"""Multiphase LBM wetting_t/hysteresis simulation example_for_test.

Uses the streaming I/O path to write snapshots to disk during the
``jax.lax.scan`` loop via ``jax.debug.callback``, then plots them
post-run using the registered plot operators.

Configuration is loaded from config_complex.toml.
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


def wetting_hysteresis_simulation():
    """Run a multiphase wetting_t simulation with streaming I/O + plotting."""
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
