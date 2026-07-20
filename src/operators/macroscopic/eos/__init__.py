"""Equation-of-state operators for multiphase macroscopic computation.

Public API: build_eos_fn()
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import cast
from src.operators._loader import auto_load_operators
from src.operators.factory import build_operator
from src.operators.macroscopic.eos._carnahan_starling import carnahan_starling_pressure
from src.operators.macroscopic.eos._double_well import double_well_pressure

if TYPE_CHECKING:
    from src.operators.macroscopic import MultiphaseParams
    from src.operators.protocols import EOSFunction

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


__all__ = ["build_eos_fn", "carnahan_starling_pressure", "double_well_pressure"]
