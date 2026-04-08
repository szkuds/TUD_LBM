"""Simulation runner for TUD-LBM.

Functional API — all orchestration uses pure functions and
``jax.lax.scan``.  No mutable classes are involved.

Public API::

    from runner import run, init_state
    from runner import step_single_phase, step_multiphase
"""

from .run import init_state
from .run import run
from .step import step_multiphase
from .step import step_single_phase

__all__ = [
    "init_state",
    "run",
    "step_multiphase",
    "step_single_phase",
]
