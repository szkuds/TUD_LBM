"""TUD-LBM: Lattice Boltzmann Method package from Delft University of Technology."""

from tud_lbm.config import load
from tud_lbm.core.run import Run

__all__ = ["load", "Run"]
