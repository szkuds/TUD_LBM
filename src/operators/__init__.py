"""Operators for LBM simulations.

This package provides all operators needed for lattice Boltzmann simulations,
including boundary conditions, equilibrium distributions, differential operators,
macroscopic field computation, wetting models, external forces, and initialisation.

Sub-packages:
    boundary_condition: Boundary condition implementations.
    equilibrium: EquilibriumWB distribution functions.
    differential: Gradient and Laplacian operators.
    macroscopic: Density and velocity field computation.
    wetting: Contact angle and wetting boundary conditions.
    force: External force models (gravity, electric, composite).
    initialise: Field initialisation routines.
"""

from .boundary_condition import BoundaryCondition
from .equilibrium import EquilibriumWB
from .differential import Laplacian, Gradient
from .macroscopic import Macroscopic, MacroscopicMultiphaseCS, MacroscopicMultiphaseDW
from .wetting import ContactAngle, ContactLineLocation, determine_padding_modes, wetting_1d, apply_wetting_to_all_edges, has_wetting_bc, WettingParameters
from .force import Force, CompositeForce, ElectricForce, GravityForceMultiphase
from .initialise import Initialise
