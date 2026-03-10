"""Collision simulation_operators for the lattice Boltzmann method.

Provides BGK (single relaxation time) and MRT (multiple relaxation time)
collision_models schemes, plus the source term for the well-balanced forcing scheme.

Classes:
    CollisionBGK: Standard BGK collision_models with optional source term.
    CollisionMRT: MRT collision_models with configurable relaxation rates.
    SourceTerm: Forcing source term.
"""

from .collision_base import CollisionBase
from .collision_bgk import CollisionBGK
from .collision_mrt import CollisionMRT
