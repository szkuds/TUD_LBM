"""Initialisation operators — implementations of InitialiserOperator protocol.

Public API: build_f(), build_initialise_fn()

Implementation modules (_standard.py, _multiphase_bubble.py) are internal; use the factory to access.

Example:
    from operators.initialise import build_f

    # High-level: builds f with all setup parameters
    f = build_f(setup, init_kwargs={"density": 1.0})

    # Low-level: direct initialiser access
    from operators.initialise import build_initialise_fn
    init_fn = build_initialise_fn("standard")
    f = init_fn(64, 64, lattice, density=1.0)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
from operators._loader import auto_load_operators
from operators.factory import build_operator
from operators.initialise._kwargs import _build_init_kwargs
from operators.protocols import InitialiserOperator

if TYPE_CHECKING:
    from config.simulation_config import SimulationSetup

# Auto-discover and import private operator modules for registry registration
auto_load_operators("operators.initialise")


def build_initialise_fn(scheme: str = "standard") -> InitialiserOperator:
    """Return an initialisation operator satisfying InitialiserOperator protocol.

    Args:
        scheme: Initialisation type name ("standard", "multiphase_bubble", etc).
                Defaults to "standard".

    Returns:
        A callable satisfying the InitialiserOperator protocol.

        Can be called as::

            operator(nx, ny, lattice, **kwargs) → f

        Type-checkers see this as an InitialiserOperator.

    Raises:
        ValueError: If scheme is not registered.

    Examples:
        >>> from operators.initialise import build_initialise_fn
        >>> init = build_initialise_fn("standard")
        >>> f = init(64, 64, lattice, density=1.0)
    """
    return build_operator("initialise", scheme)


def build_f(
    setup: SimulationSetup,
    init_kwargs: dict | None = None,
) -> jnp.ndarray:
    """Build the initial population distribution *f* for *setup*.

    Selects the initialiser indicated by ``setup.config.init_type``,
    resolves all required keyword arguments (including multiphase
    parameters and file paths), and returns ``f`` with shape
    ``(nx, ny, q, 1)``.

    Args:
        setup: :class:`~setup.simulation_setup.SimulationSetup`.
        init_kwargs: Optional caller overrides forwarded to the
            initialiser (e.g. ``density``, ``rho_l``, ``npz_path``).

    Returns:
        ``jnp.ndarray`` of shape ``(nx, ny, q, 1)``.

    Examples:
        >>> from operators.initialise import build_f
        >>> f = build_f(setup)
        >>> f = build_f(setup, init_kwargs={"density": 0.9})
    """
    init_type = setup.config.init_type
    kw = _build_init_kwargs(setup, init_type, init_kwargs)
    init_fn = build_initialise_fn(init_type)
    nx, ny = setup.grid_shape[0], setup.grid_shape[1]
    return init_fn(nx, ny, setup.lattice, **kw)


__all__ = ["build_f", "build_initialise_fn"]
