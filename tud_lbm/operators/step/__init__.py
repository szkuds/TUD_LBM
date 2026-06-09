"""Step operators — implementations of update_timestep protocols.

Public API: build_step_fn()

Implementation modules (_single_phase.py, _multiphase.py, _multiphase_wetting.py) are internal;
use the factory to access. Helper modules (_common.py) provide shared logic.

Example:
    from tud_lbm.operators.step import build_step_fn

    step = build_step_fn("single_phase")
    new_state = step(setup, state)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import cast
from tud_lbm.operators._loader import auto_load_operators
from tud_lbm.operators.factory import build_operator

if TYPE_CHECKING:
    from tud_lbm.operators.protocols import StepOperator

# Auto-discover and import private operator modules for registry registration
auto_load_operators("tud_lbm.operators.step")
from tud_lbm.operators.step import _multiphase_hysteresis_chemical_step  # noqa: F401, E402


def build_step_fn(scheme: str = "single_phase") -> StepOperator:
    """Return a step operator looked up from the registry.

    Args:
        scheme: Step operator name ("single_phase", "multiphase", "multiphase_wetting").
                Defaults to "single_phase".

    Returns:
        A callable satisfying the StepOperator protocol,
        registered under the "update_timestep" kind.

    Raises:
        ValueError: If scheme is not registered.

    Examples:
        >>> from tud_lbm.operators.step import build_step_fn
        >>> step = build_step_fn("single_phase")
        >>> new_state = step(setup, state)
    """
    return cast("StepOperator", build_operator("update_timestep", scheme))


__all__ = [
    "build_step_fn",
]
