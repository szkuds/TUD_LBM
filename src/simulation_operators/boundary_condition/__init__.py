"""Boundary condition simulation_operators for LBM simulations.

Provides boundary condition implementations for handling simulation_domain edges,
including periodic, bounce-back, symmetry and wetting boundary conditions.

Classes:
    BoundaryConditionBase: Abstract base class for boundary conditions.
    BoundaryCondition: Composite dispatcher that chains per-edge BC operators (``"standard"``).
    BounceBackBoundaryCondition: Half-way bounce-back BC (``"bounce-back"``).
    SymmetryBoundaryCondition: Mirror-symmetry BC (``"symmetry"``).
    PeriodicBoundaryCondition: No-op periodic boundary condition (``"periodic"``).
    WettingBoundaryCondition: Wetting BC — bounce-back variant (``"wetting"``).
"""

from .base import BoundaryConditionBase
from .bounce_back import BounceBackBoundaryCondition
from .symmetry import SymmetryBoundaryCondition
from .periodic import PeriodicBoundaryCondition
from .wetting import WettingBoundaryCondition
from .boundary_condition import BoundaryCondition
