"""Step operators — implementations of update_timestep protocols.

Public API: build_step_fn()

Implementation modules (_single_phase.py, _multiphase.py) are internal;
use the factory to access. Helper modules (_common.py, _wetting_shims.py)
provide shared logic.

Example:
    from operators.step import build_step_fn

    step = build_step_fn("single_phase")
    new_state = step(setup, state)
"""

from __future__ import annotations
from operators._loader import auto_load_operators
from operators.factory import build_operator
from operators.step._multiphase import step_multiphase
from operators.step._single_phase import step_single_phase

# Auto-discover and import private operator modules for registry registration
auto_load_operators("operators.step")


def build_step_fn(scheme: str = "single_phase"):
    """Return a step operator looked up from the registry.

    Args:
        scheme: Step operator name ("single_phase" or "multiphase").
                Defaults to "single_phase".

    Returns:
        A callable registered under the "update_timestep" kind.

    Raises:
        ValueError: If scheme is not registered.

    Examples:
        >>> from operators.step import build_step_fn
        >>> step = build_step_fn("single_phase")
        >>> new_state = step(setup, state)
    """
    return build_operator("update_timestep", scheme)


__all__ = [
    "build_step_fn",
    "step_multiphase",
    "step_single_phase",
]
