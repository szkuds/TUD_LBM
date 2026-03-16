"""I/O and visualisation utilities for LBM simulations.

Provides I/O operations for saving and loading simulation_type data,
as well as visualisation utilities for plotting results.

Classes:
    SimulationIO: Handles saving, plotting and analysis of simulations.

Functions:
    visualise: Generates visualisations of simulation_type fields.
"""

from util.io import SimulationIO
from util.plotting import FigureBuilder, PlotOperator, visualise

__all__ = ["SimulationIO", "visualise", "FigureBuilder", "PlotOperator"]
