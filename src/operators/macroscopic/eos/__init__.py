"""Equation-of-state operators for multiphase macroscopic computation.

Public API: build_eos_fn(), build_pressure_fn(), analytical_surface_tension(),
has_analytical_surface_tension()

Implementation modules (_double_well.py, _carnahan_starling.py) are internal;
use the factories to access them. Each module registers its EOS under the
``"eos"`` kind and, if it has them, its bulk pressure under ``"pressure"`` and
its closed-form surface tension under ``"surface_tension"``, using the same
name in each. To ask whether an EOS has one of those, query the registry —
``eos in get_operator_names("pressure")``,
:func:`has_analytical_surface_tension` — rather than any list maintained here.

Example:
    from src.operators.macroscopic.eos import build_eos_fn, build_pressure_fn

    eos_fn = build_eos_fn("carnahan-starling", mp)
    mu_0 = eos_fn(rho)

    pressure_fn = build_pressure_fn(mp)
    p_0 = pressure_fn(rho)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators._loader import auto_load_operators
from src.registry import get_operator_names
from src.registry import get_operators

if TYPE_CHECKING:
    from src.config.multiphase_params import MultiphaseParams
    from src.operators.protocols import EosOperator

# Auto-discover EOS modules in this subpackage.
auto_load_operators("src.operators.macroscopic.eos")


def build_eos_fn(eos: str, mp: MultiphaseParams) -> EosOperator:
    """Build an EOS callable with bound parameters.

    Args:
        eos: EOS scheme name from config (e.g. ``"double-well"``).
        mp: :class:`~src.operators.macroscopic.MultiphaseParams` carrying
            all EOS-specific scalars.

    Returns:
        A bound :class:`~src.operators.protocols.EosOperator`
        ``eos_fn(rho) -> mu_0``.
    """
    return get_operators("eos")[eos].target(mp)


def build_pressure_fn(mp: MultiphaseParams) -> EosOperator:
    """Build the bulk-pressure callable for the EOS bound in *mp*.

    This is the bulk thermodynamic pressure only: the interfacial ``-kappa``
    terms are not included, so ``p_0`` swings across a diffuse interface.
    Consumers that need the full normal pressure add
    ``-kappa * (rho * laplacian(rho) + |grad rho|^2 / 2)`` themselves.

    Args:
        mp: :class:`~src.operators.macroscopic.MultiphaseParams` carrying the
            EOS name and its scalars.

    Returns:
        A bound :class:`~src.operators.protocols.EosOperator`
        ``pressure_fn(rho) -> p_0``. The multiphase macroscopic operator
        evaluates it every step and returns it as the pressure field.

    Raises:
        ValueError: If the EOS's own parameters are missing.
    """
    return get_operators("pressure")[mp.eos].target(mp)


def has_analytical_surface_tension(eos: str | None) -> bool:
    """Whether *eos* has a closed-form surface tension, so needs no calibration.

    The inverse is the question the analysis layer actually asks: an EOS that
    is absent from the ``"surface_tension"`` kind has no closed form, and its
    sigma must be measured by
    :mod:`src.simulation_io.analysis.surface_tension`. Membership of the kind
    is the whole check — do not reintroduce a list of EOS names.
    """
    return eos is not None and eos in get_operator_names("surface_tension")


def analytical_surface_tension(mp: MultiphaseParams) -> float | None:
    """Closed-form liquid-gas surface tension for the EOS bound in *mp*.

    Returns ``None`` when the EOS has no closed form (Young-Laplace
    calibration is the only way to a sigma for it) and when the EOS has one
    but its parameters cannot produce a value. Both cases mean the same thing
    to every caller — fall back to a measured value — so they are not
    distinguished.
    """
    if not has_analytical_surface_tension(mp.eos):
        return None
    sigma = get_operators("surface_tension")[mp.eos].target(mp)
    return None if sigma is None else float(sigma)


__all__ = [
    "analytical_surface_tension",
    "build_eos_fn",
    "build_pressure_fn",
    "has_analytical_surface_tension",
]
