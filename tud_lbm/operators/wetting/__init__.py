"""Wetting and hysteresis operators — implementations of wetting protocols.

Public API: build_wetting_fn()

Implementation modules (_contact_angle.py, _contact_line.py, hysteresis.py)
are internal; use the factory to access.

Utility helpers (build_wetting_applicator, resolve_wetting_fields) and the
WettingParams data class are re-exported for convenience.

Example:
    from operators.wetting import build_wetting_fn

    contact_angle_fn = build_wetting_fn("contact_angle")
    ca_left, ca_right = contact_angle_fn(rho, rho_mean)
"""

from __future__ import annotations
from tud_lbm.operators._loader import auto_load_operators
from tud_lbm.operators.factory import build_operator
from tud_lbm.operators.wetting._params import WettingParams
from tud_lbm.operators.wetting.hysteresis import update_wetting_state as _hysteresis_update  # noqa: F401

# Auto-discover and import private operator modules for registry registration
auto_load_operators("operators.wetting")


def build_wetting_fn(scheme: str = "contact_angle"):
    """Return a wetting operator looked up from the registry.

    Args:
        scheme: Wetting operator name ("contact_angle",
                "contact_line_location", "hysteresis", or others).
                Defaults to "contact_angle".

    Returns:
        A callable registered under the "wetting" kind.

    Raises:
        ValueError: If scheme is not registered.

    Examples:
        >>> from operators.wetting import build_wetting_fn
        >>> ca_fn = build_wetting_fn("contact_angle")
        >>> ca_left, ca_right = ca_fn(rho, rho_mean)
    """
    return build_operator("wetting", scheme)


__all__ = [
    "WettingParams",  # Data class for optimisation params
    "build_wetting_fn",
]
