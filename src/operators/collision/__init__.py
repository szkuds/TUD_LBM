"""Collision operators for the lattice Boltzmann method.

Provides BGK (single relaxation time) and MRT (multiple relaxation time)
collision schemes, plus the source term for the well-balanced forcing scheme.

Classes:
    CollisionBGK: Standard BGK collision with optional source term.
    CollisionMRT: MRT collision with configurable relaxation rates.
    SourceTerm: Forcing source term.
"""

from .collision_base import CollisionBase
from .collision_bgk import CollisionBGK
from .collision_mrt import CollisionMRT
from .source import SourceTerm
