"""Simulation runner for TUD-LBM.

Functional API — all orchestration uses pure functions and
``jax.lax.scan``.  No mutable classes are involved.

Public API::

    from runner import run, init_state
    from runner import step_single_phase, step_multiphase, get_step_fn
"""

from .step import (
    step_single_phase,
    step_multiphase,
    get_step_fn,
)
from .run import run, init_state

__all__ = [
    "step_single_phase",
    "step_multiphase",
    "get_step_fn",
    "run",
    "init_state",
]
