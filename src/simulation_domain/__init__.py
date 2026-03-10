"""Domain definitions for LBM simulations.

This package provides the spatial and velocity space discretizations
required for lattice Boltzmann simulations.

Sub-packages:
    grid: Spatial grid definitions for 2D and 3D domains.
    lattice: Lattice velocity models (D2Q9, D3Q19, etc.).

Classes:
    Grid: Rectangular grid with configurable dimensions.
    Lattice: Lattice velocity model with discrete velocity sets and weights.
"""

from .grid import Grid
from .lattice import Lattice

__all__ = ["Grid", "Lattice"]
