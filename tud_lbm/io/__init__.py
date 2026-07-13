"""Input/Output operations for LBM simulations (public API).

This module consolidates all I/O functionality: configuration readers, output
writers, and visualization tools.

Main Classes
------------

SimulationIO
    Manage simulation output to disk. Provides convenient interface for
    saving trajectories in multiple formats.

    Example::

        io = SimulationIO(base_dir="results", config=config)
        final_state, trajectory = run(setup, state, nt=1000)
        io.save(final_state, trajectory)

output_writers
    Registry of output format handlers (NumPy, VTK).

    Example::

        writers = output_writers.available()  # ['numpy', 'vtk']
        numpy_writer = output_writers['numpy']()

Submodules
----------

readers
    Configuration adapters for loading settings from files.

    - DictAdapter : Load from Python dicts
    - TomlAdapter : Load from TOML files

    Example::

        from tud_lbm.io.readers import TomlAdapter
        config = TomlAdapter().load("config.toml")

plotting
    Visualization operators and figure builders.

    - FigureBuilder : Assemble multi-panel figures from simulation data
    - Animator      : Encode saved timestep frames into mp4/gif
    - PlotOperator  : Base class for custom visualization panels

    Example::

        from tud_lbm.io.plotting import FigureBuilder
        builder = FigureBuilder(config, "results/run_001/")
        builder.build_all()

callbacks
    Host-callback-based streaming I/O for use inside a ``lax.scan`` body.

    - make_save_callback : Build a ``do_save(state, t)`` callback that
      writes snapshots to disk via ``jax.debug.callback`` without
      breaking the JIT trace. Used internally by
      :func:`~tud_lbm.pipeline.runner.run` when ``io_handler`` is given.
"""

from .output_data import output_writers
from .report import HtmlReport
from .save import SimulationIO

__all__ = ["HtmlReport", "SimulationIO", "output_writers"]
