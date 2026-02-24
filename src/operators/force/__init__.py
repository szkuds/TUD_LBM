"""Force models for LBM simulations.

This sub-package provides external force implementations that can be
composed via :class:`CompositeForce` and injected into any simulation.

Available forces:
    - :class:`Force` — abstract base class for custom force implementations
    - :class:`CompositeForce` — helper class to combine multiple force models
    - :class:`GravityForceMultiphase` — constant gravitational body force
    - :class:`ElectricForce` — leaky-dielectric electrostatic force
"""

from .gravitational_force import GravityForceMultiphase
from .composite_force import CompositeForce
from .electric_force import ElectricForce
from .force_base import Force
