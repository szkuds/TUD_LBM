"""I/O and visualization utilities for LBM simulations.

Provides I/O operations for saving and loading simulation data,
as well as visualization utilities for plotting results.

Classes:
    SimulationIO: Handles saving, plotting and analysis of simulations.

Functions:
    visualise: Generates visualizations of simulation fields.
"""

from .io import SimulationIO
from .plotting import visualise

__all__ = ["SimulationIO", "visualise"]
