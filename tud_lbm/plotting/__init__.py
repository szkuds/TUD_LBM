"""Visualization utilities for LBM simulation results (public API).

Re-exports plotting tools from tud_lbm.io.plotting for convenient access.
Provides matplotlib-based visualization of density, velocity, and other fields.

Main Classes
------------

FigureBuilder
    Assemble multi-panel composite figures from simulation config and data.
    Supports custom panel operators for flexible visualization.

    Example::

        from tud_lbm.plotting import FigureBuilder
        builder = FigureBuilder(config=config, run_dir="results/run_001/")
        builder.build_all()  # Save all figures

PlotOperator
    Abstract base class for custom visualization panels. Extend this to
    create custom plot types that integrate with FigureBuilder.

visualise()
    Backward-compatible entry point for rendering all timesteps in a run
    directory. Automatically discovers and uses registered plot operators.

    Example::

        from tud_lbm.plotting import visualise
        visualise("results/run_001/", skip=10)

Available Plot Operators
------------------------

The framework includes registered operators for:
- Density field (heatmap)
- Velocity magnitude (heatmap)
- Velocity field (vector plot)
- Streamlines
- Custom forces analysis

Register new operators using @plotting_operator decorator in tud_lbm.registry.

Complete Example
----------------

::

    from tud_lbm import SimulationConfig, build_setup, run, init_state
    from tud_lbm.plotting import visualise

    config = SimulationConfig(grid_shape=(64, 64), tau=0.8, nt=100)
    setup = build_setup(config)
    state = init_state(setup)
    final_state, _ = run(setup, state, nt=100)

    # Render figures for a saved run directory
    visualise("results/my_run/", skip=5)
"""

from tud_lbm.io.plotting import FigureBuilder
from tud_lbm.io.plotting import PlotOperator
from tud_lbm.io.plotting import visualise

__all__ = ["FigureBuilder", "PlotOperator", "visualise"]
