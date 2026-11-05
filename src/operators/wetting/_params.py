"""Wetting containers: the four wall parameters and the per-step wetting state.

:class:`WettingState` is the wetting addon's substate, ``State.ext["wetting"]``.
It carries the same four parameter names as :class:`WettingParams`, so the wall
ghost fill reads either one: the live state, or the hysteresis optimiser's
candidate parameters.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import NamedTuple
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Mapping


class WettingParams(NamedTuple):
    """Optimisation wetting boundary parameters for hysteresis optimiser.

    Four scalar fields representing wetting behaviour at left and right contact lines.
    Used only for non-chemical-step simulations. Chemical step cases are extended with per-region pre/post variants.
    """

    d_rho_left: jnp.ndarray
    d_rho_right: jnp.ndarray
    phi_left: jnp.ndarray
    phi_right: jnp.ndarray

    @classmethod
    def from_defaults(cls, defaults: Mapping[str, float]) -> WettingParams:
        """Build the parameters from ``src.config.wetting.wetting_scalars``.

        Each of the four scalars becomes a 0-d array.
        """
        return cls(**{name: jnp.array(defaults[name]) for name in cls._fields})


class WettingState(NamedTuple):
    """Dynamic wetting / hysteresis parameters, updated every step.

    All fields are ``jax.Array`` scalars so that the state can flow through
    ``jax.lax.scan`` as part of the carry.

    Attributes:
        phi_left: Wetting potential parameter (left).
        phi_right: Wetting potential parameter (right).
        d_rho_left: Density offset parameter (left contact line).
        d_rho_right: Density offset parameter (right contact line).
        ca_left: Current contact angle, left.
        ca_right: Current contact angle, right.
        cll_left: Contact-line location, left. A coordinate *tangential* to the
            wetting wall: x for bottom/top walls, y for left/right walls.
        cll_right: Contact-line location, right. Tangential to the wetting wall
            (see ``cll_left``).

    ``cll_left`` / ``cll_right`` are the hysteresis *anchors*: they follow the
    measured contact line only while it moves the way its exceeded window bound
    allows, and hold still while it is pinned.
    """

    phi_left: jnp.ndarray
    phi_right: jnp.ndarray
    d_rho_left: jnp.ndarray
    d_rho_right: jnp.ndarray
    ca_left: jnp.ndarray
    ca_right: jnp.ndarray
    cll_left: jnp.ndarray
    cll_right: jnp.ndarray
