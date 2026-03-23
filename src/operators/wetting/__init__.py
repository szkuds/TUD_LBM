"""Wetting and hysteresis operators — pure functions.

Provides JAX-compatible pure-function equivalents of the legacy
:class:`~simulation_operators.wetting.ContactAngle`,
:class:`~simulation_operators.wetting.ContactLineLocation`, and
the hysteresis optimisation from
:class:`~update_timestep.UpdateMultiphaseHysteresis`.
"""

from operators.wetting.contact_angle import compute_contact_angle
from operators.wetting.contact_line import compute_contact_line_location
from operators.wetting.hysteresis import WettingParams
from operators.wetting.hysteresis import update_wetting_state
from operators.wetting.wetting_util import apply_wetting_to_all_edges
from operators.wetting.wetting_util import resolve_wetting_fields

__all__ = [
    "WettingParams",
    "apply_wetting_to_all_edges",
    "compute_contact_angle",
    "compute_contact_line_location",
    "resolve_wetting_fields",
    "update_wetting_state",
]
