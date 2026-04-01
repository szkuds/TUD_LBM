"""Initialisation operators — implementations of InitialiserOperator protocol.

Public API: build_initialise_fn()

Implementation modules are internal; use the factory to access.

Example:
    from operators.initialise import build_initialise_fn

    init_fn = build_initialise_fn("standard")
    f = init_fn(64, 64, lattice, density=1.0)
"""

from __future__ import annotations

from operators.protocols import InitialiserOperator
from operators.factory import build_operator
from operators._loader import auto_load_operators

# Auto-discover and import private operator modules for registry registration
auto_load_operators('operators.initialise')


def _import_initialise_modules() -> None:
    """Import concrete initialiser modules so decorators register them."""
    from operators.initialise import from_file as _ff_impl  # noqa: F401
    from operators.initialise import multiphase_bubble as _mb_impl  # noqa: F401
    from operators.initialise import multiphase_bubble_bot as _mbb_impl  # noqa: F401
    from operators.initialise import multiphase_bubble_bubble as _mbbb_impl  # noqa: F401
    from operators.initialise import multiphase_droplet as _md_impl  # noqa: F401
    from operators.initialise import multiphase_droplet_top as _mdt_impl  # noqa: F401
    from operators.initialise import multiphase_droplet_variable_radius as _mdvr_impl  # noqa: F401
    from operators.initialise import multiphase_lateral_bubble as _mlb_impl  # noqa: F401
    from operators.initialise import standard as _std_impl  # noqa: F401
    from operators.initialise import wetting as _wet_impl  # noqa: F401
    from operators.initialise import wetting_chemical_step as _wcs_impl  # noqa: F401


_import_initialise_modules()


def build_initialise_fn(scheme: str = "standard") -> InitialiserOperator:
    """Return an initialisation operator satisfying InitialiserOperator protocol.

    Args:
        scheme: Initialisation type name ("standard", "multiphase_bubble", etc).
                Defaults to "standard".

    Returns:
        A callable satisfying the InitialiserOperator protocol.
        Can be called as: operator(nx, ny, lattice, **kwargs) → f

        Type-checkers see this as an InitialiserOperator.

    Raises:
        ValueError: If scheme is not registered.

    Examples:
        >>> from operators.initialise import build_initialise_fn
        >>> init = build_initialise_fn("standard")
        >>> f = init(64, 64, lattice, density=1.0)
    """
    _import_initialise_modules()

    return build_operator("initialise", scheme)


__all__ = ["build_initialise_fn"]
