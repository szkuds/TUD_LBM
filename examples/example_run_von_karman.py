"""Von Karman vortex street: channel flow past a cylinder at Re ~ 100.

Single-phase D2Q9 flow enters on the left through a parabolic velocity inlet,
leaves on the right through a zero-gradient outlet, and sheds an alternating
vortex street behind a circular cylinder. Configuration is loaded from
``config_von_karman.toml``.

Run it with::

    python examples/example_run_von_karman.py

It produces, inside the timestamped run directory it prints:

* ``data/timestep_*.npz``: the saved snapshots (streamed to disk while the
  ``lax.scan`` loop runs);
* ``plots/snapshots/timestep_<nt>.png``: velocity magnitude above vorticity at
  the final step;
* ``plots/von_karman.gif``: the same two panels animated over four shedding
  periods of the developed street, looping without a seam.

The same run is available from the command line::

    tud-lbm run examples/config_von_karman.toml
    tud-lbm animate <run_dir> --fields velocity,vorticity --dpi 80 --output anim.gif
"""

from pathlib import Path
from src.config.adapter_toml import TomlAdapter
from src.config.jax_config import configure_jax
from src.pipeline.runner import init_state
from src.pipeline.runner import run
from src.pipeline.setup import build_setup
from src.simulation_io.plotting.animator import Animator
from src.simulation_io.plotting.figure_builder import FigureBuilder
from src.simulation_io.save import SimulationIO

# Configure JAX (64-bit precision, JIT enabled).
configure_jax()

#: GIF settings: 10 fps plays one shedding period (~11 frames) in about a second,
#: and 80 dpi keeps the file small enough to embed in a README (~4-5 MB).
GIF_FPS = 10
GIF_DPI = 80


def run_von_karman() -> Path:
    """Run the vortex street, plot the final step and animate it.

    Returns:
        Path to the run directory.
    """
    config_path = Path(__file__).parent / "config_von_karman.toml"
    config = TomlAdapter().load(str(config_path))

    setup = build_setup(config)
    state = init_state(setup)

    # Create the I/O handler — this makes the timestamped run directory.
    io = SimulationIO(
        base_dir=config.results_dir,
        config=config,
        simulation_name=config.simulation_name,
        output_format=config.output_format,
    )

    # Stream snapshots to disk while the lax.scan loop runs; nothing before
    # skip_interval is saved, so only the developed street is kept.
    run(
        setup,
        state,
        nt=config.nt,
        save_interval=config.save_interval,
        skip_interval=config.skip_interval,
        io_handler=io,
        save_fields=tuple(config.save_fields) if config.save_fields else None,
    )

    builder = FigureBuilder(config=config, run_dir=io.run_dir)
    _, last_snapshot = builder.sorted_timed_files()[-1]
    still = builder.build_single(last_snapshot)
    print(f"Final snapshot: {still}")

    gif = Animator(config, io.run_dir, fps=GIF_FPS, dpi=GIF_DPI).create(builder.plot_dir / "von_karman.gif")
    print(f"Animation:      {gif}")

    return Path(io.run_dir)


if __name__ == "__main__":
    run_von_karman()
