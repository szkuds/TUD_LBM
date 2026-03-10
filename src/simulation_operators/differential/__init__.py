"""Differential simulation_operators for LBM simulations.

Provides finite-difference simulation_operators for computing spatial derivatives
of scalar and vector fields on the LBM grid.

Classes:
    Gradient: First-order spatial gradient operator.
    Laplacian: Second-order Laplacian operator.
"""

from .gradient import Gradient
from .laplacian import Laplacian
