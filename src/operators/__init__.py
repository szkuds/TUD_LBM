"""Operators for LBM simulations.

This package provides all operators needed for lattice Boltzmann simulations,
including boundary conditions, equilibrium distributions, differential operators,
macroscopic field computation, wetting models, external forces, and initialisation.

All concrete operator classes are registered in the global operator registry
upon import via the ``@register_operator`` decorator.

Sub-packages:
    boundary_condition: Boundary condition implementations.
    collision: BGK, MRT collision operators and source term.
    equilibrium: EquilibriumWB distribution functions.
    differential: Gradient and Laplacian operators.
    macroscopic: Density and velocity field computation.
    stream: Streaming operator.
    wetting: Contact angle and wetting boundary conditions.
    force: External force models (gravity, electric, composite).
    initialise: Field initialisation routines.
"""

from .boundary_condition import BoundaryCondition
from .collision import CollisionBGK, CollisionMRT, SourceTerm
from .equilibrium import EquilibriumWB
from .differential import Laplacian, Gradient
from .macroscopic import Macroscopic, MacroscopicMultiphaseCS, MacroscopicMultiphaseDW
from .stream import Streaming
from .wetting import ContactAngle, ContactLineLocation, determine_padding_modes, wetting_1d, apply_wetting_to_all_edges, has_wetting_bc, WettingParameters
from .force import Force, CompositeForce, ElectricForce, GravityForceMultiphase
from .initialise import Initialise
