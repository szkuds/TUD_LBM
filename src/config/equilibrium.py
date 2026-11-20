"""The equilibrium a run uses, and the source term registered as its pair.

Which source term belongs to which equilibrium is physics, so each equilibrium
declares it at registration (``@equilibrium_operator(name=..., source_term=...)``).
The pairing is about where the pressure lives: the standard equilibrium carries
``cs^2*rho`` and takes the plain Guo source; the well-balanced one carries none
and takes the source with the ``grad(rho)`` corrections. Mixing them loses the
pressure or counts it twice.

The configuration is the one place that asks the registry, so the metadata is
read and checked here, and ``build_setup`` receives plain names.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.registry import OperatorEntry


def _registry() -> dict[str, OperatorEntry]:
    """The ``equilibrium`` registry, after importing the package for its registration side effect."""
    import src.operators.equilibrium  # noqa: F401
    from src.registry import get_operators

    return get_operators("equilibrium")


def validate_equilibrium(name: str, *, multiphase: bool) -> None:
    """Reject an unregistered equilibrium, one without a source term, or one the phase cannot supply.

    Raises:
        ValueError: If *name* is not registered, has no ``source_term`` pair, or
            needs a reference density on a single-phase run.
    """
    registry = _registry()
    if name not in registry:
        msg = f"equilibrium must be one of {sorted(registry)}, got '{name}'"
        raise ValueError(msg)
    source_term_for(name)
    if uses_reference_density(name) and not multiphase:
        msg = f"equilibrium '{name}' takes its pressure from the chemical potential, so it needs a [multiphase] section"
        raise ValueError(msg)


def uses_reference_density(name: str | None) -> bool:
    """Whether the equilibrium *name* carries a pressure built on a reference density.

    Declared at registration (``reference_density=True``). Such an equilibrium needs
    ``[multiphase] equilibrium_reference_density``, and the macroscopic operator
    then supplies its pressure. ``False`` for an unregistered name, which
    :func:`validate_equilibrium` reports.
    """
    entry = _registry().get(name) if name is not None else None
    return entry is not None and bool((entry.metadata or {}).get("reference_density"))


def source_term_for(name: str) -> str:
    """The source term the equilibrium *name* registered as its pair.

    Raises:
        ValueError: If the equilibrium declares no ``source_term``.
    """
    source_term = (_registry()[name].metadata or {}).get("source_term")
    if not isinstance(source_term, str):
        msg = f"equilibrium '{name}' registers no source_term pair"
        # A configuration error, not a type error.
        raise ValueError(msg)  # noqa: TRY004
    return source_term
