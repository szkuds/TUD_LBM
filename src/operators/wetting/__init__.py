"""Wetting and hysteresis operators — pure functions.

Provides JAX-compatible pure-function equivalents of the legacy
:class:`~simulation_operators.wetting.ContactAngle`,
:class:`~simulation_operators.wetting.ContactLineLocation`, and
the hysteresis optimisation from
:class:`~update_timestep.UpdateMultiphaseHysteresis`.
"""

from operators.wetting.contact_angle import compute_contact_angle
from operators.wetting.contact_line import compute_contact_line_location
from operators.wetting.hysteresis import update_wetting_state, WettingParams
from operators.wetting.wetting_util import (
    resolve_wetting_fields,
    apply_wetting_to_all_edges,
)

__all__ = [
    "compute_contact_angle",
    "compute_contact_line_location",
    "update_wetting_state",
    "WettingParams",
    "resolve_wetting_fields",
    "apply_wetting_to_all_edges",
]
