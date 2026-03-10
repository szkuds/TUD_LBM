"""Wetting boundary condition operator.

Wetting boundaries use the same half-way bounce-back rule as
:class:`BounceBackBoundaryCondition`.  This subclass exists so that
users can write ``bottom = "wetting"`` in config files and the name
is resolved from the operator registry without any hardcoded mapping.
"""

from __future__ import annotations

from app_setup.registry import register_operator
from .bounce_back import BounceBackBoundaryCondition


@register_operator("boundary_condition")
class WettingBoundaryCondition(BounceBackBoundaryCondition):
    """Wetting boundary condition (bounce-back variant).

    Registered as ``"wetting"`` so that config files using
    ``bottom = "wetting"`` are resolved via the registry.
    The streaming-level behaviour is identical to bounce-back;
    additional wetting physics (contact-angle imposition, etc.)
    is handled by separate wetting operators.
    """

    name = "wetting"

