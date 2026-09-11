"""Shared wetting parameter containers and config reading."""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple

if TYPE_CHECKING:
    import jax.numpy as jnp

#: The four wetting scalars, each as ``(canonical_key, legacy_key, default)``.
#:
#: One table rather than a default dict plus a per-parameter default: the two
#: drifted apart the moment they were written twice, and nothing checks that a
#: neutral config built in one module agrees with a reader defaulting in another.
WETTING_SCALARS: tuple[tuple[str, str, float], ...] = (
    ("phi_left", "phi_l", 1.0),
    ("phi_right", "phi_r", 1.0),
    ("d_rho_left", "d_rho_l", 0.0),
    ("d_rho_right", "d_rho_r", 0.0),
)

#: Wetting configuration with no wall modification, for a run that asks for
#: hysteresis without declaring a ``[wetting]`` section.
NEUTRAL_WETTING_CONFIG: dict[str, float] = {name: default for name, _legacy, default in WETTING_SCALARS}


def wetting_scalar(cfg: dict[str, Any], name: str, legacy_name: str, *, default: float) -> float:
    """Read one wetting scalar, accepting the legacy short key.

    ``wetting_config`` is a free-form ``dict[str, Any]`` straight off the TOML,
    so a lookup is ``Any`` and a missing key is ``None`` -- neither of which
    ``float()`` accepts. Both the state initialiser and the differential-operator
    closures need exactly this narrowing, and a key present but null must mean
    the same to both.
    """
    for key in (name, legacy_name):
        value = cfg.get(key)
        if value is not None:
            return float(value)
    return default


class WettingParams(NamedTuple):
    """Optimisation wetting boundary parameters for hysteresis optimiser.

    Four scalar fields representing wetting behaviour at left and right contact lines.
    Used only for non-chemical-step simulations. Chemical step cases are extended with per-region pre/post variants.
    """

    d_rho_left: jnp.ndarray
    d_rho_right: jnp.ndarray
    phi_left: jnp.ndarray
    phi_right: jnp.ndarray
