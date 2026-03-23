"""Initialisation factory — maps init_type strings to pure functions.

Uses the global operator registry to resolve initialisation type names
to their corresponding pure functions.

Usage::

    from operators.initialise.factory import get_init_fn

    init_fn = get_init_fn("multiphase_bubble")
    f = init_fn(64, 64, lattice, rho_l=1.0, rho_v=0.33, interface_width=4)
"""

from __future__ import annotations

from typing import Callable

from registry import get_operators

# Ensure all init modules are imported so decorators fire
from operators.initialise import standard as _std  # noqa: F401
from operators.initialise import multiphase_bubble as _mb  # noqa: F401
from operators.initialise import multiphase_bubble_bot as _mbb  # noqa: F401
from operators.initialise import multiphase_bubble_bubble as _mbbb  # noqa: F401
from operators.initialise import multiphase_droplet as _md  # noqa: F401
from operators.initialise import multiphase_droplet_top as _mdt  # noqa: F401
from operators.initialise import (
    multiphase_droplet_variable_radius as _mdvr,
)  # noqa: F401
from operators.initialise import multiphase_lateral_bubble as _mlb  # noqa: F401
from operators.initialise import wetting as _wet  # noqa: F401
from operators.initialise import wetting_chemical_step as _wcs  # noqa: F401
from operators.initialise import from_file as _ff  # noqa: F401


def get_init_fn(init_type: str) -> Callable:
    """Return the initialisation function for the given *init_type*.

    Args:
        init_type: One of the registered initialisation operator names.

    Returns:
        A callable ``init_fn(nx, ny, lattice, **kwargs) -> jnp.ndarray``.

    Raises:
        ValueError: If *init_type* is not registered.
    """
    init_ops = get_operators("initialise")
    if init_type not in init_ops:
        raise ValueError(
            f"Unknown init_type '{init_type}'. " f"Available: {sorted(init_ops)}"
        )
    return init_ops[init_type].target
