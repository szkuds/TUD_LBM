"""Wetting operators — hysteresis implementations of HysteresisOperator protocol.

Public API: build_hysteresis_fn()

The ``wetting`` registry kind holds only the per-step hysteresis updates
(_hysteresis.py). The fixed-name helpers — ``compute_contact_angle``
(_contact_angle.py), ``compute_contact_line_location`` (_contact_line.py) and
``build_wetting_applicator`` (_applicator.py) — are imported directly from
their modules.

Example:
    from src.operators.wetting import build_hysteresis_fn

    hysteresis_fn = build_hysteresis_fn("hysteresis")
    wetting_next = hysteresis_fn(wetting, rho, setup, trial_step_fn=trial_step)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from src.operators._loader import auto_load_operators
from src.registry import get_operators

if TYPE_CHECKING:
    from src.operators.protocols import HysteresisOperator

# Auto-discover and import private operator modules for registry registration
auto_load_operators("src.operators.wetting")


def build_hysteresis_fn(scheme: str) -> HysteresisOperator:
    """Return a hysteresis operator satisfying HysteresisOperator protocol.

    Args:
        scheme: Hysteresis operator name ("hysteresis" or "chemical_step_hysteresis").

    Returns:
        The registered wetting-state update for *scheme*.
    """
    return get_operators("wetting")[scheme].target


__all__ = [
    "build_hysteresis_fn",
]
