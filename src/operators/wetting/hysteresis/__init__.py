"""Wetting and hysteresis operators — implementations of wetting protocols.

Public API: build_wetting_fn()

Implementation modules (_contact_angle.py, _contact_line.py, hysteresis.py)
are internal; use the factory to access.

Utility helpers (build_wetting_applicator) and the
WettingParams data class are re-exported for convenience.

Example:
    from operators.wetting import build_wetting_fn

    contact_angle_fn = build_wetting_fn("contact_angle")
    ca_left, ca_right = contact_angle_fn(rho, rho_mean)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import cast
from src.operators._loader import auto_load_operators
from src.operators.factory import build_operator
from src.operators.wetting._params import WettingParams
from .hysteresis import _D_RHO_NEUTRAL
from .hysteresis import _PHI_NEUTRAL
from .hysteresis import _clamp_params
from .hysteresis import _cost_ca
from .hysteresis import _cost_cll
from .hysteresis import _optimise_single_param
from .hysteresis import _phi_is_active
from .hysteresis import update_wetting_state
from .hysteresis import update_wetting_state_chemical_step

if TYPE_CHECKING:
    from src.operators.protocols import HysteresisOperator

# Auto-discover and import private operator modules for registry registration
auto_load_operators("src.operators.wetting")


def build_wetting_fn(scheme: str = "contact_angle") -> HysteresisOperator:
    """Return a wetting operator looked up from the registry.

    Args:
        scheme: Wetting operator name ("contact_angle",
                "contact_line_location", "hysteresis", or others).
                Defaults to "contact_angle".

    Returns:
        A callable satisfying the HysteresisOperator protocol,
        registered under the "wetting" kind.

    Raises:
        ValueError: If scheme is not registered.

    Examples:
        >>> from src.operators.wetting import build_wetting_fn
        >>> ca_fn = build_wetting_fn("contact_angle")
        >>> ca_left, ca_right = ca_fn(rho, rho_mean)
    """
    return cast("HysteresisOperator", build_operator("wetting", scheme))


__all__ = [
    "_D_RHO_NEUTRAL",
    "_PHI_NEUTRAL",
    "WettingParams",
    "_clamp_params",
    "_cost_ca",
    "_cost_cll",
    "_optimise_single_param",
    "_phi_is_active",
    "build_wetting_fn",
    "update_wetting_state",
    "update_wetting_state_chemical_step",
]
