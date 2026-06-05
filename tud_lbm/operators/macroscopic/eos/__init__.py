"""Equation-of-state operators for multiphase macroscopic computation.

Public API: build_eos_fn()
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from tud_lbm.operators._loader import auto_load_operators
from tud_lbm.operators.factory import build_operator

if TYPE_CHECKING:
    from tud_lbm.operators.macroscopic import MultiphaseParams
    from tud_lbm.operators.protocols import EOSFunction
    from tud_lbm.operators.protocols import EOSOperator

# Auto-discover EOS modules in this subpackage.
auto_load_operators("tud_lbm.operators.macroscopic.eos")


def build_eos_fn(eos: str, mp: MultiphaseParams) -> EOSFunction:
    """Build an EOS callable with bound parameters.

    Args:
        eos: EOS scheme name from config (e.g. ``"double-well"``).
        mp: :class:`~tud_lbm.operators.macroscopic.MultiphaseParams` carrying
            all EOS-specific scalars.

    Returns:
        A bound :class:`~tud_lbm.operators.protocols.EOSFunction`
        ``eos_fn(rho) -> mu_0``.

    Raises:
        ValueError: If *eos* is not registered in the EOS registry.
    """
    eos_builder: EOSOperator = build_operator("eos", eos)
    return eos_builder(mp)


__all__ = ["build_eos_fn"]
