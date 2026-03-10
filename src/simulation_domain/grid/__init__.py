"""Grid definition for LBM simulations.

Provides the spatial discretization of the simulation_type simulation_domain,
including 2D and 3D grid support and edge extraction utilities.

Classes:
    Grid: Rectangular grid with configurable dimensions.
"""

from .grid import Grid

__all__ = [
    "Grid",
]
