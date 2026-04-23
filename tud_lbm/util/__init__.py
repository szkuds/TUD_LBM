"""I/O and visualisation utilities for LBM simulations.

Provides I/O operations for saving and loading simulation_type data,
as well as visualisation utilities for plotting results.

Classes:
    SimulationIO: Handles saving, plotting and analysis of simulations.

Functions:
    visualise: Generates visualisations of simulation_type fields.
"""

from tud_lbm.util.io import SimulationIO
from tud_lbm.util.plotting import FigureBuilder
from tud_lbm.util.plotting import PlotOperator
from tud_lbm.util.plotting import visualise

__all__ = ["FigureBuilder", "PlotOperator", "SimulationIO", "visualise"]
