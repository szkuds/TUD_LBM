"""Equation-of-state operators for multiphase macroscopic computation.

Public API: build_eos_fn(), build_pressure_fn()

Implementation modules (_double_well.py, _carnahan_starling.py) are internal;
use the factories to access them. Each module registers its EOS under the
``"eos"`` kind and, if it has one, its bulk pressure under the ``"pressure"``
kind using the same name. To ask whether an EOS has a bulk pressure, query the
registry — ``eos in get_operator_names("pressure")`` — rather than any list
maintained here.

Example:
    from src.operators.macroscopic.eos import build_eos_fn, build_pressure_fn

    eos_fn = build_eos_fn("carnahan-starling", mp)
    mu_0 = eos_fn(rho)

    pressure_fn = build_pressure_fn(mp)
    p_0 = pressure_fn(rho)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import cast
from src.operators._loader import auto_load_operators
from src.operators.factory import build_operator

if TYPE_CHECKING:
    from src.operators.macroscopic import MultiphaseParams
    from src.operators.protocols import EOSFunction
    from src.operators.protocols import PressureFunction

# Auto-discover EOS modules in this subpackage.
auto_load_operators("src.operators.macroscopic.eos")


def build_eos_fn(eos: str, mp: MultiphaseParams) -> EOSFunction:
    """Build an EOS callable with bound parameters.

    Args:
        eos: EOS scheme name from config (e.g. ``"double-well"``).
        mp: :class:`~src.operators.macroscopic.MultiphaseParams` carrying
            all EOS-specific scalars.

    Returns:
        A bound :class:`~src.operators.protocols.EOSFunction`
        ``eos_fn(rho) -> mu_0``.

    Raises:
        ValueError: If *eos* is not registered in the EOS registry.
    """
    eos_builder = build_operator("eos", eos)
    return cast("EOSFunction", eos_builder(mp))


def build_pressure_fn(mp: MultiphaseParams) -> PressureFunction:
    """Build the bulk-pressure callable for the EOS bound in *mp*.

    This is the bulk thermodynamic pressure only: the interfacial ``-kappa``
    terms are not included, so ``p_0`` swings across a diffuse interface.
    Consumers that need the full normal pressure add
    ``-kappa * (rho * laplacian(rho) + |grad rho|^2 / 2)`` themselves.

    Args:
        mp: :class:`~src.operators.macroscopic.MultiphaseParams` carrying the
            EOS name and its scalars.

    Returns:
        A bound :class:`~src.operators.protocols.PressureFunction`
        ``pressure_fn(rho) -> p_0``.

    Raises:
        ValueError: If ``mp.eos`` has no registered pressure implementation, or
            if the EOS's own parameters are missing.
    """
    pressure_builder = build_operator("pressure", mp.eos)
    return cast("PressureFunction", pressure_builder(mp))


__all__ = [
    "build_eos_fn",
    "build_pressure_fn",
]
