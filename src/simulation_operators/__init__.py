"""Operators for LBM simulations.

This package provides all simulation_operators needed for lattice Boltzmann simulations,
including boundary conditions, equilibrium distributions, differential simulation_operators,
macroscopic field computation, wetting models, external forces, and initialisation.

All concrete operator classes self-register in the global operator registry
(``src/registry.py``) upon import via the ``@register_operator`` decorator.
Importing this package triggers registration of every operator listed below.

Adding a new operator
~~~~~~~~~~~~~~~~~~~~~
1. Create a class with ``@register_operator("<kind>")`` and ``name = "..."``.
2. Import it in the relevant sub-package ``__init__.py``.

That's it — app_setup validation, factories, and the CLI pick it up
automatically.  See ``dev_notes/OperatorRegistry.md`` for the full guide.

Sub-packages:
    boundary_condition: Boundary condition implementations.
    collision_models: BGK, MRT collision_models simulation_operators and source term.
    equilibrium: EquilibriumWB distribution functions.
    differential: Gradient and Laplacian simulation_operators.
    macroscopic: Density and velocity field computation.
    stream: Streaming operator.
    wetting: Contact angle and wetting boundary conditions.
    force: External force models (gravity, electric, composite).
    initialise: Field initialisation routines.
"""

from .boundary_condition import BoundaryCondition
from .collision_models import CollisionBGK, CollisionMRT
from .equilibrium import EquilibriumWB
from .differential import Laplacian, Gradient
from .macroscopic import Macroscopic, MacroscopicMultiphaseCS, MacroscopicMultiphaseDW
from .stream import Streaming
from .wetting import ContactAngle, ContactLineLocation, determine_padding_modes, wetting_1d, apply_wetting_to_all_edges, has_wetting_bc, WettingParameters
from .force import Force, CompositeForce, ElectricForce, GravityForceMultiphase, SourceTerm
from .initialise import Initialise
