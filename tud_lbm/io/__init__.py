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
    
        from tud_lbm.readers import TomlAdapter
        config = TomlAdapter().load("config.toml")

plotting
    Visualization operators and figure builders.
    
    - FigureBuilder : Assemble multi-panel figures from simulation data
    - PlotOperator  : Base class for custom visualization panels
    - visualise()   : Backward-compatible entry point for rendering results
    
    Example::
    
        from tud_lbm.plotting import visualise
        visualise("results/run_001/")
"""

from .output_data import output_writers
from .save import SimulationIO

__all__ = ["SimulationIO", "output_writers"]


